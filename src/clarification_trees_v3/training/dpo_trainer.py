import os
import torch
import torch.nn.functional as F
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
from clarification_trees_v3.config.iterative_rl_dpo_schema import DPOTreeDatasetConfig
from clarification_trees_v3.models.transformers_model_v2 import TransformersModelV2
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, DialogNode
from clarification_trees_v3.dataset.dataset import ClarificationTreeSample
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

from logging import getLogger
logger = getLogger(__name__)

def get_dpo_collate_fn(model: TransformersModelV2, dpo_dataset_config: DPOTreeDatasetConfig):
    def dpo_sample_collate(batch: list[ClarificationTreeSample]):
        chosen_samples = []
        rejected_samples = []
        chosen_ref_logps_list = []
        rejected_ref_logps_list = []
        
        for sample in batch:
            tree = sample.tree
            sidecar = sample.tree_sidecar
            parent_node_idx = sample.parent_node_idx
            child_idxs = sample.child_node_idxs
            
            rewards = [sidecar.reward_cache[idx] for idx in child_idxs]
            
            pairs = [] # List of (chosen_idx, rejected_idx)
            
            positive_threshold = dpo_dataset_config.positive_reward_threshold
            method = dpo_dataset_config.pair_selection_method
            
            if method == "top-bottom":
                max_reward = max(rewards)
                min_reward = min(rewards)
                if max_reward >= positive_threshold and max_reward > min_reward:
                    best_idx = child_idxs[rewards.index(max_reward)]
                    worst_idx = child_idxs[rewards.index(min_reward)]
                    pairs.append((best_idx, worst_idx))
            elif method == "top-random":
                max_reward = max(rewards)
                if max_reward >= positive_threshold:
                    best_idx = child_idxs[rewards.index(max_reward)]
                    lower_idxs = [idx for idx, r in zip(child_idxs, rewards) if r < max_reward]
                    if lower_idxs:
                        worst_idx = random.choice(lower_idxs)
                        pairs.append((best_idx, worst_idx))
            elif method == "top-all":
                max_reward = max(rewards)
                if max_reward >= positive_threshold:
                    best_idx = child_idxs[rewards.index(max_reward)]
                    lower_idxs = [idx for idx, r in zip(child_idxs, rewards) if r < max_reward]
                    for worst_idx in lower_idxs:
                        pairs.append((best_idx, worst_idx))
            elif method == "all-all":
                for i, c_idx in enumerate(child_idxs):
                    if rewards[i] >= positive_threshold:
                        for j, r_idx in enumerate(child_idxs):
                            if rewards[i] > rewards[j]:
                                pairs.append((c_idx, r_idx))
            
            for chosen_idx, rejected_idx in pairs:
                chosen_target = tree.get_node(chosen_idx).response
                rejected_target = tree.get_node(rejected_idx).response
                
                c_idx_in_child = child_idxs.index(chosen_idx)
                r_idx_in_child = child_idxs.index(rejected_idx)
                
                c_ref_logp_sum = sample.token_logprobs[c_idx_in_child].sum().item()
                r_ref_logp_sum = sample.token_logprobs[r_idx_in_child].sum().item()
                
                chosen_ref_logps_list.append(c_ref_logp_sum)
                rejected_ref_logps_list.append(r_ref_logp_sum)
                
                # Get base trajectory up to parent
                base_traj = tree.get_trajectory(parent_node_idx).trajectory
                
                # Chosen trajectory
                chosen_traj_obj = tree.get_trajectory(parent_node_idx) # Create a new instance
                chosen_node = DialogNode(NodeType.CLARIFICATION_QUESTION, None, None, chosen_target)
                chosen_traj_obj.trajectory.append(chosen_node)
                chosen_tokenized = model.preprocess_sft_training_inputs(chosen_traj_obj, role="user")
                
                # Rejected trajectory
                rejected_traj_obj = tree.get_trajectory(parent_node_idx)
                rejected_node = DialogNode(NodeType.CLARIFICATION_QUESTION, None, None, rejected_target)
                rejected_traj_obj.trajectory.append(rejected_node)
                rejected_tokenized = model.preprocess_sft_training_inputs(rejected_traj_obj, role="user")
                
                chosen_samples.append(chosen_tokenized)
                rejected_samples.append(rejected_tokenized)
                
        if len(chosen_samples) == 0:
            return None # Handle empty batch in training loop
            
        pad_token_id = model.processor.tokenizer.pad_token_id

        # Pad chosen
        c_input_ids = pad_sequence([s["input_ids"] for s in chosen_samples], batch_first=True, padding_value=pad_token_id)
        c_labels = pad_sequence([s["labels"] for s in chosen_samples], batch_first=True, padding_value=-100)
        c_attn = pad_sequence([s["attention_mask"] for s in chosen_samples], batch_first=True, padding_value=0)
        c_pixels = torch.cat([s["pixel_values"] for s in chosen_samples], dim=0)
        c_grid = torch.cat([s["image_grid_thw"] for s in chosen_samples], dim=0)
        
        # Pad rejected
        r_input_ids = pad_sequence([s["input_ids"] for s in rejected_samples], batch_first=True, padding_value=pad_token_id)
        r_labels = pad_sequence([s["labels"] for s in rejected_samples], batch_first=True, padding_value=-100)
        r_attn = pad_sequence([s["attention_mask"] for s in rejected_samples], batch_first=True, padding_value=0)
        r_pixels = torch.cat([s["pixel_values"] for s in rejected_samples], dim=0)
        r_grid = torch.cat([s["image_grid_thw"] for s in rejected_samples], dim=0)

        processed_batch = {
            "chosen_input_ids": c_input_ids,
            "chosen_labels": c_labels,
            "chosen_attention_mask": c_attn,
            "chosen_pixel_values": c_pixels,
            "chosen_image_grid_thw": c_grid,
            
            "rejected_input_ids": r_input_ids,
            "rejected_labels": r_labels,
            "rejected_attention_mask": r_attn,
            "rejected_pixel_values": r_pixels,
            "rejected_image_grid_thw": r_grid,
            
            "chosen_ref_logps": torch.tensor(chosen_ref_logps_list, dtype=torch.float),
            "rejected_ref_logps": torch.tensor(rejected_ref_logps_list, dtype=torch.float),
        }

        if "mm_token_type_ids" in chosen_samples[0]:
            c_mm = pad_sequence([s["mm_token_type_ids"] for s in chosen_samples], batch_first=True, padding_value=0)
            r_mm = pad_sequence([s["mm_token_type_ids"] for s in rejected_samples], batch_first=True, padding_value=0)
            processed_batch["chosen_mm_token_type_ids"] = c_mm
            processed_batch["rejected_mm_token_type_ids"] = r_mm

        return processed_batch

    return dpo_sample_collate

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor):
    """
    Computes log probabilities for tokens corresponding to non -100 labels.
    """
    # shift labels and logits
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    
    loss_mask = (labels != -100)
    
    # dummy label to avoid index out of bounds
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

def forward_pass_dpo(model: TransformersModelV2, batch: dict, device: str, is_train: bool, beta: float):
    # Prepare chosen kwargs
    c_kwargs = {
        "input_ids": batch["chosen_input_ids"].to(device),
        "attention_mask": batch["chosen_attention_mask"].to(device),
        "pixel_values": batch["chosen_pixel_values"].to(device),
        "image_grid_thw": batch["chosen_image_grid_thw"].to(device),
        "labels": batch["chosen_labels"].to(device)
    }
    if "chosen_mm_token_type_ids" in batch:
        c_kwargs["mm_token_type_ids"] = batch["chosen_mm_token_type_ids"].to(device)

    # Prepare rejected kwargs
    r_kwargs = {
        "input_ids": batch["rejected_input_ids"].to(device),
        "attention_mask": batch["rejected_attention_mask"].to(device),
        "pixel_values": batch["rejected_pixel_values"].to(device),
        "image_grid_thw": batch["rejected_image_grid_thw"].to(device),
        "labels": batch["rejected_labels"].to(device)
    }
    if "rejected_mm_token_type_ids" in batch:
        r_kwargs["mm_token_type_ids"] = batch["rejected_mm_token_type_ids"].to(device)

    with torch.set_grad_enabled(is_train):
        pol_c_outputs = model.peft_model(**c_kwargs)
        pol_r_outputs = model.peft_model(**r_kwargs)
        
        pol_c_logps = get_batch_logps(pol_c_outputs.logits, c_kwargs["labels"])
        pol_r_logps = get_batch_logps(pol_r_outputs.logits, r_kwargs["labels"])

    ref_c_logps = batch["chosen_ref_logps"].to(device)
    ref_r_logps = batch["rejected_ref_logps"].to(device)

    pi_logratios = pol_c_logps - pol_r_logps
    ref_logratios = ref_c_logps - ref_r_logps
    logits = pi_logratios - ref_logratios
    
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Compute accuracy for logging (chosen > rejected)
    chosen_rewards = beta * (pol_c_logps - ref_c_logps).detach()
    rejected_rewards = beta * (pol_r_logps - ref_r_logps).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, accuracy, chosen_rewards.mean(), rejected_rewards.mean()

def evaluate(model: TransformersModelV2, val_loader: DataLoader, device: str, step_id: int, beta: float, eval_batches: int | None = None, iter_number: int | None = None):
    assert model.peft_model is not None, "No adapter is currently loaded or constructed."
    model.peft_model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    total_batches = len(val_loader)
    if eval_batches is not None and eval_batches < total_batches:
        total_batches = eval_batches

    progress = tqdm(itertools.islice(val_loader, total_batches), desc="Validation Loss", total=total_batches)
    for step, batch in enumerate(progress):
        if batch is None:
            continue
            
        loss, accuracy, _, _ = forward_pass_dpo(model, batch, device, is_train=False, beta=beta)

        total_loss += loss.item()
        total_acc += accuracy.item()
        num_batches += 1

        progress.set_postfix({"loss": loss.item(), "acc": accuracy.item()})

    avg_loss = total_loss / max(1, num_batches)
    avg_acc = total_acc / max(1, num_batches)
    logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    log_dict = {"val/dpo_loss": avg_loss, "val/dpo_acc": avg_acc, "val/step": step_id}
    if iter_number is not None:
        log_dict[f"val/iter_{iter_number}/dpo_loss"] = avg_loss
    wandb.log(log_dict)

    model.peft_model.train()
    return avg_loss

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

def dpo_train_loop(
    model: TransformersModelV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: schema.HuggingfaceClarificationModelConfig,
    dpo_dataset_config: DPOTreeDatasetConfig,
    beta: float,
    save_dir: Path,
    override_epochs: int | None = None,
    iter_number: int | None = None
):
    lora_config = model_config.lora_config
    training_config = lora_config.training_config
    assert training_config is not None, "Training config not found."

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

    batches_per_epoch = dpo_dataset_config.batches_per_epoch
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

    logger.info(f"Starting DPO training: {epochs} epochs, {steps_per_epoch} batches/epoch")
    global_step = 0
    
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    if evaluate_first:
        logger.info("Evaluating before training...")
        best_val_loss = evaluate(model, val_loader, device, global_step, beta, dpo_dataset_config.eval_batches_per_epoch, iter_number)
        
        if training_config.fallback_on_no_improvement == "previous_lora":
            logger.info("Saving initial model as best_adapter fallback.")
            model.save_adapter(save_dir / "best_adapter", adapter_name="default")

    for epoch in range(epochs):
        model.peft_model.train()
        progress_bar = tqdm(itertools.islice(train_loader, steps_per_epoch), desc=f"Training Epoch {epoch+1}", total=steps_per_epoch)

        epoch_loss = 0.0

        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue
                
            loss, accuracy, chosen_rewards, rejected_rewards = forward_pass_dpo(model, batch, device, is_train=True, beta=beta)
            loss = loss / gradient_accumulation_steps
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
                    "train/dpo_loss": loss.item() * gradient_accumulation_steps,
                    "train/dpo_acc": accuracy.item(),
                    "train/chosen_rewards": chosen_rewards.item(),
                    "train/rejected_rewards": rejected_rewards.item(),
                    "train/lr": current_lr,
                    "train/step": global_step,
                    "train/epoch": epoch + (step / steps_per_epoch)
                }
                if iter_number is not None:
                    log_dict[f"train/iter_{iter_number}/dpo_loss"] = log_dict["train/dpo_loss"]
                wandb.log(log_dict)
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps, "acc": accuracy.item()})
    
        logger.info(f"End of Epoch {epoch+1}. Running validation...")
        val_loss = evaluate(model, val_loader, device, global_step, beta, dpo_dataset_config.eval_batches_per_epoch, iter_number)

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
