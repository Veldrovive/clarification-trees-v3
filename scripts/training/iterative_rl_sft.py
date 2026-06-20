import os
import shutil
import random
import asyncio
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import copy

from clarification_trees_v3.config.iterative_rl_sft_schema import IterativeRLSFTConfig, parse_iterative_rl_sft_config
from clarification_trees_v3.definitions import GENERATED_TREES_PATH, BASE_WEIGHTS_PATH
from clarification_trees_v3.utils import set_seed, SentenceAnalyzer
from clarification_trees_v3.dataset.dataset import ClearVQADataset, SFTClarificationTreeDataset
from clarification_trees_v3.models.utils import use_models
from clarification_trees_v3.models import construct_semantic_clusterer
from clarification_trees_v3.dataset.tree_generation import process_dataset_lazily, print_timer_tree
from clarification_trees_v3.training.sft_trainer import get_collate_fn, train_loop, construct_model_with_lora

from logging import getLogger
logger = getLogger(__name__)

def check_and_clean_malformed_trees(out_dir: Path):
    if not out_dir.exists():
        return
    deleted = 0
    for tree_dir in out_dir.iterdir():
        if tree_dir.is_dir():
            if not (tree_dir / "tree.json").exists() or not (tree_dir / "tree_sidecar.json").exists():
                logger.warning(f"Deleting malformed tree directory: {tree_dir}")
                shutil.rmtree(tree_dir)
                deleted += 1
    if deleted > 0:
        logger.info(f"Deleted {deleted} malformed tree directories.")

def get_completed_trees_count(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return sum(1 for d in out_dir.iterdir() if d.is_dir() and (d / "tree.json").exists() and (d / "tree_sidecar.json").exists())

def get_lora_path(cfg: IterativeRLSFTConfig, iter_number: int) -> Path | None:
    assert BASE_WEIGHTS_PATH is not None
    lora_id = cfg.clarification_model.lora_config.lora_id
    lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
    save_dir = lora_checkpoint_path / f"{lora_id}_rl_sft_iter_{iter_number}"
    
    if (save_dir / "best_adapter").exists():
        return save_dir / "best_adapter"
    return None

async def run_iterative_loop(cfg: IterativeRLSFTConfig, raw_cfg: DictConfig):
    assert GENERATED_TREES_PATH is not None
    assert BASE_WEIGHTS_PATH is not None

    sentence_analyzer = SentenceAnalyzer()
    ds = ClearVQADataset(load_images=False, table_name="train_annotated.jsonl")

    # PRE-STARTUP FIX: Configure the LoRA settings for vLLM startup
    if cfg.start_iter == 0:
        cfg.clarification_model.lora_config.use_lora = False
    else:
        cfg.clarification_model.lora_config.use_lora = True
        cfg.clarification_model.lora_config.lora_id_postfix = f"_rl_sft_iter_{cfg.start_iter - 1}"

    async with use_models(cfg) as (cq_model, answer_model):
        for iter_number in range(cfg.start_iter, cfg.max_iters):
            logger.info(f"=== Starting Iteration {iter_number} ===")
            
            # --- LoRA Swapping Phase ---
            if iter_number > cfg.start_iter:
                cfg.clarification_model.lora_config.use_lora = True
                cfg.clarification_model.lora_config.lora_id_postfix = f"_rl_sft_iter_{iter_number - 1}"
                lora_id = cfg.clarification_model.lora_config.lora_id
                lora_dir_name = f"{lora_id}{cfg.clarification_model.lora_config.lora_id_postfix}"
                
                logger.info(f"Swapping LoRA to {lora_dir_name} in vLLM server...")
                
                # Unload previous LoRAs to prevent OOM
                loaded_models = await cq_model._get_loaded_model_ids()
                for m in loaded_models:
                    if m != cq_model.base_model_path and m != lora_dir_name:
                        logger.info(f"Unloading previous LoRA adapter: {m}")
                        await cq_model.unload_lora_adapter(m)
                        
                await cq_model.load_lora_adapter(lora_id, allow_overwrite=True)
            elif iter_number == 0:
                cfg.clarification_model.lora_config.use_lora = False

            # Update the paths for this iteration
            iter_trees_subpath = f"{cfg.paths.data.trees_subpath}_iter_{iter_number}"
            out_dir = GENERATED_TREES_PATH / iter_trees_subpath
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this iteration's LoRA is already trained
            current_lora_path = get_lora_path(cfg, iter_number)
            if current_lora_path is not None:
                logger.info(f"LoRA for iteration {iter_number} already exists at {current_lora_path}. Skipping to next iteration.")
                continue

            # --- Phase 1: Tree Generation ---
            logger.info(f"Phase 1: Generating trees for iteration {iter_number}...")
            check_and_clean_malformed_trees(out_dir)
            existing_count = get_completed_trees_count(out_dir)
            
            trees_to_generate = cfg.trees_per_iteration - existing_count
            if trees_to_generate > 0:
                logger.info(f"Need to generate {trees_to_generate} more trees to reach {cfg.trees_per_iteration}.")
                
                # Sample random indices from the dataset
                # Note: We just sample random indices. In a perfect world, we'd exclude ones we already generated.
                # Since the dataset is large, a random subset is fine.
                indices = random.sample(range(len(ds)), trees_to_generate)

                logger.info("Loading semantic clusterer for tree generation...")
                clusterer_gpus = cfg.devices.semantic_cluster
                clusterer = construct_semantic_clusterer(raw_cfg.semantic_cluster_model, f"cuda:{clusterer_gpus[0]}")

                await process_dataset_lazily(
                    cfg=cfg,
                    dataset=ds,
                    indices=indices,
                    clusterer=clusterer,
                    cq_model=cq_model,
                    answer_model=answer_model,
                    sentence_analyzer=sentence_analyzer,
                    N_parallel_trees=25,
                    out_dir=out_dir
                )

                print_timer_tree()

                logger.info("Unloading semantic clusterer to free memory...")
                del clusterer
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.info(f"Already have {existing_count} trees for iteration {iter_number}. Skipping generation.")

            # --- Phase 2: SFT Training ---
            logger.info(f"Phase 2: SFT Training for iteration {iter_number}...")
            
            # Set the environment variable for construct_model_with_lora
            os.environ["ITER_NUMBER"] = str(iter_number)
            
            # Modify config paths to point to the trees we just generated
            sft_paths_config = copy.deepcopy(cfg.paths)
            sft_paths_config.data.trees_subpath = iter_trees_subpath
            
            logger.info("Initializing Transformers model for SFT...")
            # Set the device using the configured SFT GPU
            sft_gpus = cfg.devices.sft
            if sft_gpus is not None:
                cfg.clarification_model.lora_config.training_config.device = f"cuda:{sft_gpus[0]}"
            
            model = construct_model_with_lora(cfg.clarification_model, sft_paths_config, iter_number)
            collate_fn = get_collate_fn(model)

            tree_dirs = [d for d in out_dir.iterdir() if d.is_dir() and (d / "tree.json").exists()]
            tree_dirs.sort()
            random.shuffle(tree_dirs)
            
            val_split_size = int(len(tree_dirs) * cfg.sft_dataset.val_split)
            # Ensure at least 1 val sample if there are trees
            if val_split_size == 0 and len(tree_dirs) > 0:
                val_split_size = 1
                
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
                batch_size=cfg.clarification_model.lora_config.training_config.batch_size, 
                collate_fn=collate_fn, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, 
                batch_size=cfg.clarification_model.lora_config.training_config.batch_size, 
                collate_fn=collate_fn, 
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            lora_id = cfg.clarification_model.lora_config.lora_id
            lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
            save_dir = lora_checkpoint_path / f"{lora_id}_rl_sft_iter_{iter_number}"
            
            train_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=cfg.clarification_model,
                sft_dataset_config=cfg.sft_dataset,
                save_dir=save_dir,
                iter_number=iter_number
            )
            
            # Free the Transformers model and dataloaders to clear memory before next iteration
            del model, train_loader, val_loader, collate_fn, train_ds, val_ds
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"=== Completed Iteration {iter_number} ===")


@hydra.main(config_path="../../config", config_name="iterative_rl_sft", version_base=None)
def main(raw_cfg: DictConfig):
    cfg: IterativeRLSFTConfig = parse_iterative_rl_sft_config(raw_cfg)
    print(f"Running Iterative RL SFT with config:\n{cfg.model_dump_json(indent=2)}")
    
    set_seed(cfg.seed)
    
    # Enable W&B if needed, but since it loops, we can init here.
    # W&B can log multiple steps over time.
    import wandb
    wandb_name = cfg.wandb.name if cfg.wandb.name else f"iterative_rl_sft_{cfg.clarification_model.lora_config.lora_id}"
    wandb.init(
        project=cfg.wandb.project,
        config=cfg.model_dump(),
        name=wandb_name
    )
    
    asyncio.run(run_iterative_loop(cfg, raw_cfg))
    
    wandb.finish()

if __name__ == "__main__":
    main()
