import os
import random
import asyncio
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import copy

from clarification_trees_v3.config.iterative_rl_dpo_schema import IterativeRLDPOConfig, parse_iterative_rl_dpo_config
from clarification_trees_v3.definitions import GENERATED_TREES_PATH, BASE_WEIGHTS_PATH
from clarification_trees_v3.utils import set_seed, SentenceAnalyzer
from clarification_trees_v3.dataset.dataset import ClearVQADataset, ClarificationTreeDataset
from clarification_trees_v3.models.utils import use_models
from clarification_trees_v3.training.sft_trainer import construct_model_with_lora
from clarification_trees_v3.training.dpo_trainer import get_dpo_collate_fn, dpo_train_loop
from clarification_trees_v3.training.iterative_utils import get_lora_path, run_phase_1_tree_generation

from logging import getLogger
logger = getLogger(__name__)

async def run_iterative_loop(cfg: IterativeRLDPOConfig, raw_cfg: DictConfig):
    assert GENERATED_TREES_PATH is not None
    assert BASE_WEIGHTS_PATH is not None

    sentence_analyzer = SentenceAnalyzer()
    ds = ClearVQADataset(load_images=False, table_name="train_annotated.jsonl")
    val_ds = ClearVQADataset(load_images=False, table_name="val_annotated.jsonl")

    # PRE-STARTUP FIX: Configure the LoRA settings for vLLM startup
    if cfg.start_iter == 0:
        cfg.clarification_model.lora_config.use_lora = False
    else:
        cfg.clarification_model.lora_config.use_lora = True
        cfg.clarification_model.lora_config.lora_id_postfix = f"_rl_dpo_iter_{cfg.start_iter - 1}"

    async with use_models(cfg) as (cq_model, answer_model):
        if cfg.stop_vllm_during_dpo:
            if not cq_model.is_running_internally or not answer_model.is_running_internally:
                raise ValueError("Cannot use stop_vllm_during_dpo=True when vLLM servers are managed externally.")

        for iter_number in range(cfg.start_iter, cfg.max_iters):
            logger.info(f"=== Starting Iteration {iter_number} ===")
            
            if cfg.stop_vllm_during_dpo:
                if not cq_model.is_running:
                    logger.info("Restarting cq_model vLLM server...")
                    await cq_model.initialize_server()
                if not answer_model.is_running:
                    logger.info("Restarting answer_model vLLM server...")
                    await answer_model.initialize_server()
            
            # --- LoRA Swapping Phase ---
            if iter_number > cfg.start_iter:
                cfg.clarification_model.lora_config.use_lora = True
                cfg.clarification_model.lora_config.lora_id_postfix = f"_rl_dpo_iter_{iter_number - 1}"
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
            lora_id = cfg.clarification_model.lora_config.lora_id
            lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
            save_dir = lora_checkpoint_path / f"{lora_id}_rl_dpo_iter_{iter_number}"
            
            current_lora_path = get_lora_path(save_dir)
            if current_lora_path is not None:
                logger.info(f"LoRA for iteration {iter_number} already exists at {current_lora_path}. Skipping to next iteration.")
                continue

            # --- Phase 1: Tree Generation ---
            await run_phase_1_tree_generation(
                cfg=cfg,
                raw_cfg=raw_cfg,
                iter_number=iter_number,
                iter_trees_subpath=iter_trees_subpath,
                out_dir=out_dir,
                ds=ds,
                val_ds=val_ds,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer
            )

            # --- Phase 2: DPO Training ---
            logger.info(f"Phase 2: DPO Training for iteration {iter_number}...")
            
            if cfg.stop_vllm_during_dpo:
                logger.info("Stopping vLLM servers to free memory for DPO...")
                cq_model.stop_server()
                answer_model.stop_server()
                
                # Force cleanup
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Set the environment variable for construct_model_with_lora
            os.environ["ITER_NUMBER"] = str(iter_number)
            
            # Modify config paths to point to the trees we just generated
            dpo_paths_config = copy.deepcopy(cfg.paths)
            dpo_paths_config.data.trees_subpath = iter_trees_subpath
            
            logger.info("Initializing Transformers model for DPO...")
            # Set the device using the configured SFT GPU (re-used for DPO)
            sft_gpus = cfg.devices.sft
            if sft_gpus is not None:
                cfg.clarification_model.lora_config.training_config.device = f"cuda:{sft_gpus[0]}"
            
            model = construct_model_with_lora(cfg.clarification_model, dpo_paths_config, iter_number, postfix_pattern="_rl_dpo_iter_{}")
            collate_fn = get_dpo_collate_fn(model, cfg.dpo_dataset)

            tree_dirs = [d for d in out_dir.iterdir() if d.is_dir() and (d / "tree.json").exists()]
            tree_dirs.sort()
            rng = random.Random(cfg.seed + iter_number)
            rng.shuffle(tree_dirs)
            
            val_split_size = int(len(tree_dirs) * cfg.dpo_dataset.val_split)
            # Ensure at least 1 val sample if there are trees
            if val_split_size == 0 and len(tree_dirs) > 0:
                val_split_size = 1
                
            val_tree_dirs = tree_dirs[:val_split_size]
            train_tree_dirs = tree_dirs[val_split_size:]
            
            logger.info(f"Split {len(tree_dirs)} trees into {len(train_tree_dirs)} train and {len(val_tree_dirs)} val.")
            
            dpo_train_ds = ClarificationTreeDataset(
                trees_path=None,
                tree_paths=train_tree_dirs,
                load_images=False,
                precompute_rewards=True,
                positive_reward_threshold=cfg.dpo_dataset.positive_reward_threshold,
                require_multiple_children=True
            )
            dpo_val_ds = ClarificationTreeDataset(
                trees_path=None,
                tree_paths=val_tree_dirs,
                load_images=True,
                precompute_rewards=True,
                positive_reward_threshold=cfg.dpo_dataset.positive_reward_threshold,
                require_multiple_children=True
            )

            train_loader = DataLoader(
                dpo_train_ds, 
                batch_size=cfg.clarification_model.lora_config.training_config.batch_size, 
                collate_fn=collate_fn, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                dpo_val_ds, 
                batch_size=cfg.clarification_model.lora_config.training_config.batch_size, 
                collate_fn=collate_fn, 
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # save_dir already defined above

            
            dpo_train_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=cfg.clarification_model,
                dpo_dataset_config=cfg.dpo_dataset,
                beta=cfg.beta,
                save_dir=save_dir,
                iter_number=iter_number
            )
            
            # Free the Transformers model and dataloaders to clear memory before next iteration
            del model, train_loader, val_loader, collate_fn, dpo_train_ds, dpo_val_ds
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"=== Completed Iteration {iter_number} ===")


@hydra.main(config_path="../../config", config_name="iterative_rl_dpo", version_base=None)
def main(raw_cfg: DictConfig):
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    cfg: IterativeRLDPOConfig = parse_iterative_rl_dpo_config(raw_cfg)
    print(f"Running Iterative RL DPO with config:\n{cfg.model_dump_json(indent=2)}")
    
    set_seed(cfg.seed)
    
    import wandb
    wandb_name = cfg.wandb.name if cfg.wandb.name else f"iterative_rl_dpo_{cfg.clarification_model.lora_config.lora_id}"
    wandb.init(
        project=cfg.wandb.project,
        config=cfg.model_dump(),
        name=wandb_name
    )
    
    asyncio.run(run_iterative_loop(cfg, raw_cfg))
    
    wandb.finish()

if __name__ == "__main__":
    main()
