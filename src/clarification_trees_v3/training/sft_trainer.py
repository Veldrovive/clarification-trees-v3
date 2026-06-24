import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import wandb
import gc
import random
import itertools

from clarification_trees_v3.config import schema
from clarification_trees_v3.config.sft_tree_schema import SFTTreeConfig
from clarification_trees_v3.models.transformers_model_v2 import TransformersModelV2
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, DialogTrajectory, DialogNode
from clarification_trees_v3.dataset.dataset import ClearVQASample, SFTClarificationTreeDataset, SFTClarificationTreeSample
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

from logging import getLogger
logger = getLogger(__name__)

class ClarificationSampleCollate:
    def __init__(self, model: TransformersModelV2):
        self.model = model

    def __call__(self, batch: list[ClearVQASample | SFTClarificationTreeSample]):
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
                tokenized = self.model.preprocess_sft_training_inputs(trajectory, role="user")
                processed_samples.append(tokenized)
            elif isinstance(sample, SFTClarificationTreeSample):
                trajectory = DialogTrajectory()
                trajectory.trajectory = list(sample.trajectory.trajectory)
                
                target_node = DialogNode(NodeType.CLARIFICATION_QUESTION, None, None, sample.target)
                trajectory.trajectory.insert(0, target_node)
                
                tokenized = self.model.preprocess_sft_training_inputs(trajectory, role="user")
                processed_samples.append(tokenized)
            else:
                raise ValueError(f"Unknown sample type: {type(sample)}")
    
        input_ids = [s["input_ids"] for s in processed_samples]
        labels = [s["labels"] for s in processed_samples]

        pad_token_id = self.model.processor.tokenizer.pad_token_id

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

def get_collate_fn(model: TransformersModelV2):
    return ClarificationSampleCollate(model)

def evaluate(model: TransformersModelV2, val_loader: DataLoader, device: str, step_id: int, eval_batches: int | None = None, iter_number: int | None = None):
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    model.peft_model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    total_loss = 0.0
    num_batches = 0

    total_batches = len(val_loader)
    if eval_batches is not None and eval_batches < total_batches:
        total_batches = eval_batches

    with torch.no_grad():
        progress = tqdm(itertools.islice(val_loader, total_batches), desc="Validation Loss", total=total_batches)
        for step, batch in enumerate(progress):
            
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

    log_dict = {"val/loss": avg_loss, "val/step": step_id}
    if iter_number is not None:
        log_dict[f"val/iter_{iter_number}/loss"] = avg_loss
    wandb.log(log_dict)

    model.peft_model.train()
    return avg_loss

def generate_samples(model: TransformersModelV2, val_loader: DataLoader, device: str, step_id: int, n_samples: int = 20, iter_number: int | None = None):
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    model.peft_model.eval()

    logger.info(f"Generating {n_samples} samples for step {step_id}")

    table_data = []
    columns = ["Image Index", "Image", "Question Id", "Trajectory", "Ground Truth CQ", "Model Prediction", "Answer"]

    with torch.no_grad():
        dataset = val_loader.dataset
        # Handle Subset specifically if we need to access attributes
        vqa_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        assert isinstance(vqa_dataset, SFTClarificationTreeDataset), "Dataset is not a SFTClarificationTreeDataset"
        
        indices = list(range(len(dataset)))
        # Use a deterministic random instance so we visualize the exact same validation samples every time
        rng = random.Random(42)
        rng.shuffle(indices)
        indices = indices[:min(n_samples, len(dataset))]

        progress = tqdm(indices, desc="Generating Samples")
        for i in progress:
            sample = dataset[i]
            
            orig_i = dataset.indices[i] if hasattr(dataset, 'indices') else i
            sample_info = vqa_dataset.samples[orig_i]
            tree_idx = sample_info["tree_idx"]
            tree = vqa_dataset.trees[tree_idx]

            image = tree.init_image
            assert image is not None, "SFTClarificationTreeDataset was created without image loading enabled."
            
            trajectory_text_parts = []
            for node in sample.trajectory.trajectory[::-1]:
                role = node.node_type_to_str[node.node_type].capitalize()
                trajectory_text_parts.append(f"**{role}:** {node.response}")
            trajectory_text = "\n".join(trajectory_text_parts)

            gt_clarification = sample.target
            question_id = str(tree_idx)
            answer = str(tree.gold_answer) if tree.gold_answer else "N/A"

            inputs = model.preprocess_generation_inputs(sample.trajectory, role="user")
            prediction = model.generate(inputs)
            prediction_text = prediction[0] if isinstance(prediction, list) else prediction
            
            table_data.append([
                i,
                wandb.Image(image),
                question_id,
                trajectory_text,
                gt_clarification,
                prediction_text,
                answer
            ])

    log_dict = {"val/predictions": wandb.Table(data=table_data, columns=columns), "val/step": step_id}
    if iter_number is not None:
        log_dict[f"val/iter_{iter_number}/predictions"] = log_dict["val/predictions"]
    wandb.log(log_dict)

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

def train_loop(
    model: TransformersModelV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: schema.HuggingfaceClarificationModelConfig,
    sft_dataset_config,
    save_dir: Path,
    override_epochs: int | None = None,
    iter_number: int | None = None
):
    lora_config = model_config.lora_config
    training_config = lora_config.training_config
    assert training_config is not None, "Training config not found."

    # Training config
    epochs = override_epochs if override_epochs is not None else training_config.epochs
    evaluate_first = training_config.evaluate_first
    device = training_config.device
    lr = training_config.lr
    weight_decay = training_config.weight_decay
    gradient_accumulation_steps = training_config.gradient_accumulation_steps
    max_grad_norm = training_config.max_grad_norm
    warmup_ratio = training_config.warmup_ratio
    patience = training_config.patience
    
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving LoRA checkpoints to {save_dir}")
    
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    trainable_params = [p for p in model.peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    batches_per_epoch = sft_dataset_config.batches_per_epoch
    num_batches_in_loader = len(train_loader)
    
    if batches_per_epoch is not None and batches_per_epoch < num_batches_in_loader:
        steps_per_epoch = batches_per_epoch
    else:
        steps_per_epoch = num_batches_in_loader

    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    logger.info(f"Starting training: {epochs} epochs, {steps_per_epoch} batches/epoch")
    global_step = 0
    
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    if evaluate_first:
        logger.info("Evaluating before training...")
        best_val_loss = evaluate(model, val_loader, device, global_step, sft_dataset_config.eval_batches_per_epoch, iter_number)
        generate_samples(model, val_loader, device, global_step, iter_number=iter_number)
        
        if training_config.fallback_on_no_improvement == "previous_lora":
            logger.info("Saving initial model as best_adapter fallback.")
            model.save_adapter(save_dir / "best_adapter", adapter_name="default")

    for epoch in range(epochs):
        model.peft_model.train()
        progress_bar = tqdm(itertools.islice(train_loader, steps_per_epoch), desc=f"Training Epoch {epoch+1}", total=steps_per_epoch)

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
                log_dict = {
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/lr": current_lr,
                    "train/step": global_step,
                    "train/epoch": epoch + (step / steps_per_epoch)
                }
                if iter_number is not None:
                    log_dict[f"train/iter_{iter_number}/loss"] = log_dict["train/loss"]
                    log_dict[f"train/iter_{iter_number}/lr"] = current_lr
                wandb.log(log_dict)
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
        logger.info(f"End of Epoch {epoch+1}. Running validation...")
        val_loss = evaluate(model, val_loader, device, global_step, sft_dataset_config.eval_batches_per_epoch, iter_number)
        generate_samples(model, val_loader, device, global_step, iter_number=iter_number)

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

    if best_val_loss_epoch == -1:
        fallback = training_config.fallback_on_no_improvement
        if fallback == "first_epoch":
            logger.info("Validation loss did not improve. Fallback to first epoch.")
            epoch_0_adapter = save_dir / "epoch_000" / "adapter"
            if epoch_0_adapter.exists():
                import shutil
                shutil.copytree(epoch_0_adapter, save_dir / "best_adapter", dirs_exist_ok=True)
            else:
                logger.warning("Epoch 0 adapter not found for fallback.")
        elif fallback == "last_epoch":
            logger.info("Validation loss did not improve. Fallback to last epoch.")
            last_epoch_adapter = save_dir / f"epoch_{epochs-1:03d}" / "adapter"
            if last_epoch_adapter.exists():
                import shutil
                shutil.copytree(last_epoch_adapter, save_dir / "best_adapter", dirs_exist_ok=True)
            else:
                logger.warning("Last epoch adapter not found for fallback.")

def construct_model_with_lora(model_config: schema.HuggingfaceClarificationModelConfig, paths_config: schema.PathsConfig, iter_number: int, postfix_pattern: str = "_rl_sft_iter_{}") -> TransformersModelV2:
    lora_training_config = model_config.lora_config.training_config
    device = lora_training_config.device

    logger.info("Loading model...")
    model = TransformersModelV2(model_config, paths_config, device)

    if iter_number == 0:
        logger.info(f"Iteration {iter_number}: Initializing a new LoRA adapter...")
        model.construct_lora_adapter(model_config.lora_config.peft_config, adapter_name="default")
    else:
        prev_iter = iter_number - 1
        assert BASE_WEIGHTS_PATH is not None, "BASE_WEIGHTS_PATH is not defined."
        lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(paths_config.checkpoints.loras_subpath)
        lora_id = model_config.lora_config.lora_id
        
        prev_adapter_path = lora_checkpoint_path / f"{lora_id}{postfix_pattern.format(prev_iter)}" / "best_adapter"
        
        logger.info(f"Iteration {iter_number}: Loading previous LoRA adapter from {prev_adapter_path}...")
        model.load_adapter(prev_adapter_path, adapter_name="default", is_trainable=True)

    assert model.peft_model is not None, "No adapter was constructed."
    
    train_p, tot_p = model.peft_model.get_nb_trainable_parameters()
    logger.info(f'Trainable parameters:      {train_p/1e6:.2f}M')
    logger.info(f'Total parameters:          {tot_p/1e6:.2f}M')
    logger.info(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')

    return model
