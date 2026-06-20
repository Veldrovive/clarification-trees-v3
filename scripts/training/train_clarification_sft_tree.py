import os
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
import random
import itertools

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("sub", lambda x, y: int(x) - int(y), replace=True)
OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y), replace=True)

from clarification_trees_v3.config import schema
from clarification_trees_v3.config.sft_tree_schema import SFTTreeConfig, parse_sft_tree_config
from clarification_trees_v3.models.transformers_model_v2 import TransformersModelV2
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, DialogTrajectory, DialogNode
from clarification_trees_v3.utils import set_seed
from clarification_trees_v3.dataset.dataset import ClearVQADataset, ClearVQASample, SFTClarificationTreeDataset, SFTClarificationTreeSample

from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH, GENERATED_TREES_PATH

from logging import getLogger
logger = getLogger(Path(__file__).name)

from clarification_trees_v3.training.sft_trainer import get_collate_fn, evaluate, generate_samples, save_checkpoint, train_loop, construct_model_with_lora

@hydra.main(config_path="../../config", config_name="sft_tree_config", version_base=None)
def main(raw_cfg: DictConfig):
    cfg: SFTTreeConfig = parse_sft_tree_config(raw_cfg)
    print(f"Training with config:\n{cfg.model_dump_json(indent=2)}")

    model_config = cfg.clarification_model
    training_config = model_config.lora_config.training_config
    lora_id = model_config.lora_config.lora_id

    set_seed(training_config.seed)

    logger.info("Starting SFT training for clarification LORA using Tree Dataset")
    logger.info(f"Model config: {model_config}")

    iter_number = int(os.environ.get("ITER_NUMBER", "0"))
    model = construct_model_with_lora(model_config, cfg.paths, iter_number)
    collate_fn = get_collate_fn(model)

    assert GENERATED_TREES_PATH is not None, "GENERATED_TREES_PATH is required to load tree dataset"
    trees_path = GENERATED_TREES_PATH / cfg.paths.data.trees_subpath
    
    tree_dirs = [d for d in trees_path.iterdir() if d.is_dir()]
    tree_dirs.sort()  # Sort to guarantee deterministic splits regardless of OS
    random.shuffle(tree_dirs)
    
    val_split_size = int(len(tree_dirs) * cfg.sft_dataset.val_split)
    val_tree_dirs = tree_dirs[:val_split_size]
    train_tree_dirs = tree_dirs[val_split_size:]
    
    logger.info(f"Split {len(tree_dirs)} trees into {len(train_tree_dirs)} train and {len(val_tree_dirs)} val.")
    
    train_ds = SFTClarificationTreeDataset(
        trees_path=None,
        tree_paths=train_tree_dirs,
        load_images=False,
        advantage_threshold=cfg.sft_dataset.advantage_threshold,
        min_reward_threshold=cfg.sft_dataset.min_reward_threshold,
        top_n=cfg.sft_dataset.top_n
    )
    val_ds = SFTClarificationTreeDataset(
        trees_path=None,
        tree_paths=val_tree_dirs,
        load_images=True,
        advantage_threshold=cfg.sft_dataset.advantage_threshold,
        min_reward_threshold=cfg.sft_dataset.min_reward_threshold,
        top_n=None
    )

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

    wandb_name = cfg.wandb.name if cfg.wandb.name else lora_id
    wandb.init(
        project=cfg.wandb.project,
        config=cfg.model_dump(),
        name=wandb_name
    )

    assert BASE_WEIGHTS_PATH is not None, "BASE_WEIGHTS_PATH is not defined."
    lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
    save_dir = lora_checkpoint_path / f"{lora_id}_rl_sft_iter_{iter_number}"
    train_loop(model, train_loader, val_loader, model_config, cfg.sft_dataset, save_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
