import dotenv
dotenv.load_dotenv()

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
import hydra
from pathlib import Path
from tqdm import tqdm
import wandb
import gc

from omegaconf import DictConfig, OmegaConf

from clarification_trees_v3.config import schema
from clarification_trees_v3.config.sft_tree_schema import SFTTreeConfig, parse_sft_tree_config, get_base_config
from clarification_trees_v3.models.transformers_model_v2 import TransformersModelV2
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, DialogTrajectory, DialogNode
from clarification_trees_v3.utils import set_seed
from clarification_trees_v3.dataset.dataset import ClearVQADataset, ClearVQASample, SFTClarificationTreeDataset, SFTClarificationTreeSample

from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH, GENERATED_TREES_PATH

from logging import getLogger
logger = getLogger(Path(__file__).name)

def get_collate_fn(model: TransformersModelV2):
    def clarification_sample_collate(batch: list[ClearVQASample | SFTClarificationTreeSample]):
        processed_samples = []
        
        for sample in batch:
            if isinstance(sample, ClearVQASample):
                image = sample.image
                assert image is not None, "ClearVQADataset was created without image loading enabled."
                ambiguous_question = sample.blurred_question
                clarifying_question = sample.clarification_question
                image_path = sample.image_path
                
                # Construct tree to get the trajectory
                tree = DialogTree(ambiguous_question, image, image_path)
                # Add the target node (Assistant's response)
                cq = tree.add_node(DialogTree.ROOT, NodeType.CLARIFICATION_QUESTION, clarifying_question)
                
                # Get trajectory specifically ending at the target
                trajectory = tree.get_trajectory(cq)

                # Process to tokens in such a way that all labels are masked except the clarifying question
                tokenized = model.preprocess_sft_training_inputs(trajectory, role="user")
                processed_samples.append(tokenized)
            elif isinstance(sample, SFTClarificationTreeSample):
                trajectory = DialogTrajectory()
                trajectory.trajectory = list(sample.trajectory.trajectory)
                
                target_node = DialogNode(NodeType.CLARIFICATION_QUESTION, None, None, sample.target)
                trajectory.trajectory.append(target_node)
                
                tokenized = model.preprocess_sft_training_inputs(trajectory, role="user")
                processed_samples.append(tokenized)
            else:
                raise ValueError(f"Unknown sample type: {type(sample)}")
    
        input_ids = [s["input_ids"] for s in processed_samples]
        labels = [s["labels"] for s in processed_samples]

        pad_token_id = model.processor.tokenizer.pad_token_id

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        attention_mask = [s["attention_mask"] for s in processed_samples]
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        pixel_values = torch.cat([s["pixel_values"] for s in processed_samples], dim=0)
        grid_thw = torch.cat([s["image_grid_thw"] for s in processed_samples], dim=0)

        processed_batch = {
            "input_ids": input_ids_padded,  # (batch_size, max_seq_length)
            "labels": labels_padded,  # (batch_size, max_seq_length). -100 masks out labels that are not part of the target
            "attention_mask": attention_mask_padded,  # (batch_size, max_seq_length)
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw
        }

        if "mm_token_type_ids" in processed_samples[0]:
            mm_token_type_ids = [s["mm_token_type_ids"] for s in processed_samples]
            mm_token_type_ids_padded = pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)
            processed_batch["mm_token_type_ids"] = mm_token_type_ids_padded

        return processed_batch

    return clarification_sample_collate

def evaluate(model: TransformersModelV2, val_loader: DataLoader, device: str, step_id: int):
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    model.peft_model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation Loss")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "labels": labels
            }
            if "mm_token_type_ids" in batch:
                kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

            outputs = model.peft_model(**kwargs)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.4f}")

    wandb.log({"val/loss": avg_loss, "val/step": step_id})

    model.peft_model.train()
    return avg_loss

def generate_samples(model: TransformersModelV2, val_loader: DataLoader, device: str, step_id: int, n_samples: int = 20):
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    model.peft_model.eval()

    logger.info(f"Generating {n_samples} samples for step {step_id}")

    table_data = []
    columns = ["Image Index", "Image", "Question Id", "Ambiguous Question", "Ground Truth CQ", "Model Prediction", "Answer"]

    with torch.no_grad():
        dataset = val_loader.dataset
        # Handle Subset specifically if we need to access attributes
        vqa_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        assert isinstance(vqa_dataset, ClearVQADataset), "Dataset is not a ClearVQADataset"
        
        progress = tqdm(range(min(n_samples, len(dataset))), desc="Generating Samples")
        for i in progress:
            sample = dataset[i]
            image = sample.image
            assert image is not None, "ClearVQADataset was created without image loading enabled."
            ambiguous_q = sample.blurred_question
            gt_clarification = sample.clarification_question
            question_id = sample.question_id
            answer = sample.gold_answer

            tree = DialogTree(ambiguous_q, image, sample.image_path)
            trajectory = tree.get_trajectory(DialogTree.ROOT)

            inputs = model.preprocess_generation_inputs(trajectory, role="user")
            prediction = model.generate(inputs)
            prediction_text = prediction[0] if isinstance(prediction, list) else prediction
            
            table_data.append([
                i,
                wandb.Image(image),
                question_id,
                ambiguous_q,
                gt_clarification,
                prediction_text,
                answer
            ])

    wandb.log({"val/predictions": wandb.Table(data=table_data, columns=columns), "val/step": step_id})

    model.peft_model.train()

def save_checkpoint(
    save_dir: Path,
    model: TransformersModelV2,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step_id: int,
    epoch: int,
    is_best: bool = False
):
    checkpoint_dir = save_dir / f"epoch_{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_adapter(checkpoint_dir / "adapter", adapter_name="default")

    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step_id,
        "epoch": epoch,
        "is_best": is_best
    }, checkpoint_dir / "state.pt")

    if is_best:
        model.save_adapter(save_dir / "best_adapter", adapter_name="default")

def train_loop(model: TransformersModelV2, train_loader: DataLoader, val_loader: DataLoader, cfg: SFTTreeConfig):
    model_config = cfg.clarification_model
    lora_config = model_config.lora_config

    training_config = lora_config.training_config
    assert training_config is not None, "Training config not found."

    # Training config
    set_seed(training_config.seed)
    epochs = training_config.epochs
    evaluate_first = training_config.evaluate_first
    device = training_config.device
    lr = training_config.lr
    weight_decay = training_config.weight_decay
    gradient_accumulation_steps = training_config.gradient_accumulation_steps
    max_grad_norm = training_config.max_grad_norm
    warmup_ratio = training_config.warmup_ratio
    patience = training_config.patience

    # Get the save dir
    assert BASE_WEIGHTS_PATH is not None, "BASE_WEIGHTS_PATH is not defined."
    lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
    lora_id = lora_config.lora_id
    save_dir = lora_checkpoint_path / lora_id
    if save_dir.exists():
        logger.warning(f"LoRA checkpoint directory {save_dir} already exists. Overwrite?")
        if not input("Overwrite? (y/n): ").lower() == "y":
            return
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving LoRA checkpoints to {save_dir}")
    
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    trainable_params = [p for p in model.peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    logger.info(f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch")
    global_step = 0
    
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    if evaluate_first:
        logger.info("Evaluating before training...")
        best_val_loss = evaluate(model, val_loader, device, global_step)
        generate_samples(model, val_loader, device, global_step)

    for epoch in range(epochs):
        model.peft_model.train()
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        epoch_loss = 0.0

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "labels": labels
            }
            if "mm_token_type_ids" in batch:
                kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

            outputs = model.peft_model(**kwargs)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/lr": current_lr,
                    "train/step": global_step,
                    "train/epoch": epoch + (step / len(train_loader))
                })
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
        logger.info(f"End of Epoch {epoch+1}. Running validation...")
        val_loss = evaluate(model, val_loader, device, global_step)
        generate_samples(model, val_loader, device, global_step)

        if val_loss < best_val_loss:
            logger.info(f"New best validation loss: {val_loss}")
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            save_checkpoint(save_dir, model, optimizer, scheduler, global_step, epoch, is_best=True)
        else:
            logger.info(f"Validation loss did not improve from {best_val_loss}")
            save_checkpoint(save_dir, model, optimizer, scheduler, global_step, epoch, is_best=False)
            if epoch - best_val_loss_epoch >= patience:
                logger.info(f"No improvement for {patience} epochs. Early stopping.")
                break


def construct_model_with_lora(model_config: schema.HuggingfaceClarificationModelConfig) -> TransformersModelV2:
    lora_training_config = model_config.lora_config.training_config
    device = lora_training_config.device

    logger.info("Loading model...")
    model = TransformersModelV2(model_config, device)
    
    print("Adding LoRA to model...")
    model.construct_lora_adapter(model_config.lora_config.peft_config, adapter_name="default")
    assert model.peft_model is not None, "No adapter was constructed."
    
    train_p, tot_p = model.peft_model.get_nb_trainable_parameters()
    print(f'Trainable parameters:      {train_p/1e6:.2f}M')
    print(f'Total parameters:          {tot_p/1e6:.2f}M')
    print(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')

    return model

@hydra.main(config_path="../../config", config_name="sft_tree_config", version_base=None)
def main(raw_cfg: DictConfig):
    cfg: SFTTreeConfig = parse_sft_tree_config(raw_cfg)
    base_cfg: schema.Config = get_base_config(cfg)
    print(f"Training with config:\n{cfg.model_dump_json(indent=2)}")

    model_config = cfg.clarification_model
    training_config = model_config.lora_config.training_config
    lora_id = model_config.lora_config.lora_id

    logger.info("Starting SFT training for clarification LORA using Tree Dataset")
    logger.info(f"Model config: {model_config}")

    model = construct_model_with_lora(model_config)
    collate_fn = get_collate_fn(model)

    assert GENERATED_TREES_PATH is not None, "GENERATED_TREES_PATH is required to load tree dataset"
    trees_path = GENERATED_TREES_PATH / cfg.paths.data.trees_subpath
    
    train_ds = SFTClarificationTreeDataset(
        cfg=base_cfg,
        trees_path=trees_path,
        load_images=cfg.sft_dataset.load_images,
        advantage_threshold=cfg.sft_dataset.advantage_threshold,
        top_n=cfg.sft_dataset.top_n
    )
    val_ds = ClearVQADataset(table_name="val_annotated.jsonl")

    # Aritifially limit the dataset to a small subset of samples for quick tests
    from torch.utils.data import Subset
    # train_ds = Subset(train_ds, range(100))
    val_ds = Subset(val_ds, range(1000))

    train_loader = DataLoader(
        train_ds, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    wandb.init(
        project="clarification-sft-tree",
        config=cfg.model_dump(),
        name=lora_id
    )

    train_loop(model, train_loader, val_loader, cfg)

    wandb.finish()

if __name__ == "__main__":
    main()
