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
import torch
import wandb
import gc
import multiprocessing as mp
import logging

from clarification_trees_v3.config.iterative_rl_sft_schema import IterativeRLSFTConfig, parse_iterative_rl_sft_config
from clarification_trees_v3.definitions import GENERATED_TREES_PATH, BASE_WEIGHTS_PATH
from clarification_trees_v3.utils import set_seed, SentenceAnalyzer
from clarification_trees_v3.dataset.dataset import ClearVQADataset, SFTClarificationTreeDataset
from clarification_trees_v3.models.utils import use_models
from clarification_trees_v3.models import construct_semantic_clusterer
from clarification_trees_v3.dataset.tree_generation import process_dataset_lazily, print_timer_tree
from clarification_trees_v3.training.sft_trainer import get_collate_fn, train_loop, construct_model_with_lora
from clarification_trees_v3.training.eval_utils import gather_statistics, plot_metrics

from logging import getLogger
logger = getLogger(__name__)

from clarification_trees_v3.training.iterative_utils import (
    check_and_clean_malformed_trees,
    get_completed_trees_count,
    get_lora_path,
    run_train_tree_generation,
    run_eval_tree_generation_concurrent
)

def run_sft_training_process(
    cfg_json: str,
    iter_number: int,
    iter_trees_subpath: str,
    out_dir_str: str,
    save_dir_str: str,
    seed: int,
    wandb_run_id: str | None,
    wandb_project: str | None
):
    if wandb_run_id and wandb_project:
        wandb.init(project=wandb_project, id=wandb_run_id, resume="must")

    cfg = IterativeRLSFTConfig.model_validate_json(cfg_json)
    out_dir = Path(out_dir_str)
    save_dir = Path(save_dir_str)

    os.environ["ITER_NUMBER"] = str(iter_number)
    
    sft_paths_config = copy.deepcopy(cfg.paths)
    sft_paths_config.data.trees_subpath = iter_trees_subpath
    
    sft_gpus = cfg.devices.sft
    if sft_gpus is not None:
        cfg.clarification_model.lora_config.training_config.device = f"cuda:{sft_gpus[0]}"
    
    model = construct_model_with_lora(cfg.clarification_model, sft_paths_config, iter_number)
    collate_fn = get_collate_fn(model)

    tree_dirs = [d for d in out_dir.iterdir() if d.is_dir() and (d / "tree.json").exists()]
    tree_dirs.sort()
    rng = random.Random(seed + iter_number)
    rng.shuffle(tree_dirs)
    
    val_split_size = int(len(tree_dirs) * cfg.sft_dataset.val_split)
    if val_split_size == 0 and len(tree_dirs) > 0:
        val_split_size = 1
        
    val_tree_dirs = tree_dirs[:val_split_size]
    train_tree_dirs = tree_dirs[val_split_size:]
    
    sft_train_ds = SFTClarificationTreeDataset(
        trees_path=None,
        tree_paths=train_tree_dirs,
        load_images=False,
        advantage_threshold=cfg.sft_dataset.advantage_threshold,
        min_reward_threshold=cfg.sft_dataset.min_reward_threshold,
        top_n=cfg.sft_dataset.top_n
    )
    sft_val_ds = SFTClarificationTreeDataset(
        trees_path=None,
        tree_paths=val_tree_dirs,
        load_images=True,
        advantage_threshold=cfg.sft_dataset.advantage_threshold,
        min_reward_threshold=cfg.sft_dataset.min_reward_threshold,
        top_n=None
    )

    train_loader = DataLoader(
        sft_train_ds, 
        batch_size=cfg.clarification_model.lora_config.training_config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context="fork"
    )
    val_loader = DataLoader(
        sft_val_ds, 
        batch_size=cfg.clarification_model.lora_config.training_config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context="fork"
    )

    try:
        train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_config=cfg.clarification_model,
            sft_dataset_config=cfg.sft_dataset,
            save_dir=save_dir,
            iter_number=iter_number
        )
    finally:
        if wandb.run is not None:
            wandb.finish()

async def run_iterative_loop(cfg: IterativeRLSFTConfig, raw_cfg: DictConfig):
    assert GENERATED_TREES_PATH is not None
    assert BASE_WEIGHTS_PATH is not None

    if getattr(cfg, 'concurrent_eval_trees', False):
        if cfg.stop_vllm_during_sft:
            raise ValueError("Cannot use concurrent_eval_trees=True when stop_vllm_during_sft=True.")
        vllm_devices = set(cfg.devices.clarification) | set(cfg.devices.answer)
        if set(cfg.devices.sft).intersection(vllm_devices):
            raise ValueError("Cannot use concurrent_eval_trees=True when SFT shares GPUs with vLLM servers.")

    sentence_analyzer = SentenceAnalyzer()
    ds = ClearVQADataset(load_images=False, table_name="train_annotated.jsonl")
    val_ds = ClearVQADataset(load_images=False, table_name="val_annotated.jsonl")

    # PRE-STARTUP FIX: Configure the LoRA settings for vLLM startup
    if cfg.start_iter == 0:
        cfg.clarification_model.lora_config.use_lora = False
    else:
        cfg.clarification_model.lora_config.use_lora = True
        cfg.clarification_model.lora_config.lora_id_postfix = f"_rl_sft_iter_{cfg.start_iter - 1}"

    async with use_models(cfg) as (cq_model, answer_model):
        if cfg.stop_vllm_during_sft:
            if not cq_model.is_running_internally or not answer_model.is_running_internally:
                raise ValueError("Cannot use stop_vllm_during_sft=True when vLLM servers are managed externally.")

        for iter_number in range(cfg.start_iter, cfg.max_iters):
            logger.info(f"=== Starting Iteration {iter_number} ===")
            
            if cfg.stop_vllm_during_sft:
                if not cq_model.is_running:
                    logger.info("Restarting cq_model vLLM server...")
                    await cq_model.initialize_server()
                if not answer_model.is_running:
                    logger.info("Restarting answer_model vLLM server...")
                    await answer_model.initialize_server()
            
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
            lora_id = cfg.clarification_model.lora_config.lora_id
            lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
            save_dir = lora_checkpoint_path / f"{lora_id}_rl_sft_iter_{iter_number}"
            
            current_lora_path = get_lora_path(save_dir)
            if current_lora_path is not None:
                logger.info(f"LoRA for iteration {iter_number} already exists at {current_lora_path}. Skipping to next iteration.")
                continue

            # --- Phase 1: Train Tree Generation ---
            df_inf, df_qp, df_ent = await run_train_tree_generation(
                cfg=cfg,
                raw_cfg=raw_cfg,
                iter_number=iter_number,
                iter_trees_subpath=iter_trees_subpath,
                out_dir=out_dir,
                ds=ds,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer
            )

            if wandb.run is not None:
                train_metrics = {"iteration": iter_number}
                
                # Train metrics
                if df_inf is not None and not df_inf.empty:
                    for depth in sorted(df_inf['Depth'].unique()):
                        depth_scores = df_inf[df_inf['Depth'] == depth]['Score']
                        train_metrics[f"train/avg_reward_at_depth_{depth}"] = depth_scores.mean()
                        train_metrics[f"train/prob_correct_at_depth_{depth}"] = (float)((depth_scores >= 1.0).mean())
                if df_qp is not None and not df_qp.empty:
                    for depth in sorted(df_qp['Depth'].unique()):
                        depth_scores = df_qp[df_qp['Depth'] == depth]['Score']
                        train_metrics[f"train/avg_qp_cost_at_depth_{depth}"] = depth_scores.mean()
                if df_ent is not None and not df_ent.empty:
                    for depth in sorted(df_ent['Depth'].unique()):
                        depth_scores = df_ent[df_ent['Depth'] == depth]['Score']
                        train_metrics[f"train/avg_ent_cost_at_depth_{depth}"] = depth_scores.mean()
                        
                wandb.log(train_metrics, commit=False)

            # --- Setup Eval Tree Generation ---
            eval_trees_subpath = f"{cfg.paths.data.trees_subpath}_eval_iter_{iter_number}"
            eval_out_dir = GENERATED_TREES_PATH / eval_trees_subpath
            eval_out_dir.mkdir(parents=True, exist_ok=True)
            
            training_done_event = asyncio.Event()
            eval_task = None
            if getattr(cfg, 'concurrent_eval_trees', False):
                logger.info("Starting concurrent eval tree generation task...")
                eval_task = asyncio.create_task(run_eval_tree_generation_concurrent(
                    cfg=cfg,
                    raw_cfg=raw_cfg,
                    iter_number=iter_number,
                    eval_trees_subpath=eval_trees_subpath,
                    eval_out_dir=eval_out_dir,
                    val_ds=val_ds,
                    cq_model=cq_model,
                    answer_model=answer_model,
                    sentence_analyzer=sentence_analyzer,
                    training_done_event=training_done_event
                ))
            else:
                logger.info("Generating eval trees sequentially before SFT...")
                training_done_event.set()
                df_inf_val, df_qp_val, df_ent_val = await run_eval_tree_generation_concurrent(
                    cfg=cfg,
                    raw_cfg=raw_cfg,
                    iter_number=iter_number,
                    eval_trees_subpath=eval_trees_subpath,
                    eval_out_dir=eval_out_dir,
                    val_ds=val_ds,
                    cq_model=cq_model,
                    answer_model=answer_model,
                    sentence_analyzer=sentence_analyzer,
                    training_done_event=training_done_event
                )
                
                if wandb.run is not None:
                    val_metrics = {"iteration": iter_number}
                    if df_inf_val is not None and not df_inf_val.empty:
                        for depth in sorted(df_inf_val['Depth'].unique()):
                            depth_scores = df_inf_val[df_inf_val['Depth'] == depth]['Score']
                            val_metrics[f"val/avg_reward_at_depth_{depth}"] = depth_scores.mean()
                            val_metrics[f"val/prob_correct_at_depth_{depth}"] = (float)((depth_scores >= 1.0).mean())
                    if df_qp_val is not None and not df_qp_val.empty:
                        for depth in sorted(df_qp_val['Depth'].unique()):
                            depth_scores = df_qp_val[df_qp_val['Depth'] == depth]['Score']
                            val_metrics[f"val/avg_qp_cost_at_depth_{depth}"] = depth_scores.mean()
                    if df_ent_val is not None and not df_ent_val.empty:
                        for depth in sorted(df_ent_val['Depth'].unique()):
                            depth_scores = df_ent_val[df_ent_val['Depth'] == depth]['Score']
                            val_metrics[f"val/avg_ent_cost_at_depth_{depth}"] = depth_scores.mean()
                            
                    wandb.log(val_metrics)

            # --- Phase 2: SFT Training ---
            logger.info(f"Phase 2: SFT Training for iteration {iter_number}...")
            
            if cfg.stop_vllm_during_sft:
                logger.info("Stopping vLLM servers to free memory for SFT...")
                cq_model.stop_server()
                answer_model.stop_server()
                
                # Force cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                        
                logger.info("Waiting 5 seconds for vLLM processes to fully terminate...")
                await asyncio.sleep(5)
            
            # Set the environment variable for construct_model_with_lora
            os.environ["ITER_NUMBER"] = str(iter_number)
            
            wandb_run_id = wandb.run.id if wandb.run is not None else None
            wandb_project = cfg.wandb.project
            wandb_name = wandb.run.name if wandb.run is not None else None
            
            if wandb.run is not None:
                wandb.finish()
            
            logger.info("Starting SFT training in a separate process...")
            ctx = mp.get_context("spawn")
            p = ctx.Process(
                target=run_sft_training_process,
                args=(
                    cfg.model_dump_json(),
                    iter_number,
                    iter_trees_subpath,
                    str(out_dir),
                    str(save_dir),
                    cfg.seed,
                    wandb_run_id,
                    wandb_project
                )
            )
            p.start()
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, p.join)

            if p.exitcode != 0:
                raise RuntimeError(f"SFT training process failed with exit code {p.exitcode}")

            if wandb_run_id is not None:
                wandb.init(project=wandb_project, id=wandb_run_id, resume="must", name=wandb_name)

            if eval_task is not None:
                logger.info("SFT training process completed. Waiting for eval tree generation to finish...")
                training_done_event.set()
                df_inf_val, df_qp_val, df_ent_val = await eval_task
                
                if wandb.run is not None:
                    val_metrics = {"iteration": iter_number}
                    if df_inf_val is not None and not df_inf_val.empty:
                        for depth in sorted(df_inf_val['Depth'].unique()):
                            depth_scores = df_inf_val[df_inf_val['Depth'] == depth]['Score']
                            val_metrics[f"val/avg_reward_at_depth_{depth}"] = depth_scores.mean()
                            val_metrics[f"val/prob_correct_at_depth_{depth}"] = (float)((depth_scores >= 1.0).mean())
                    if df_qp_val is not None and not df_qp_val.empty:
                        for depth in sorted(df_qp_val['Depth'].unique()):
                            depth_scores = df_qp_val[df_qp_val['Depth'] == depth]['Score']
                            val_metrics[f"val/avg_qp_cost_at_depth_{depth}"] = depth_scores.mean()
                    if df_ent_val is not None and not df_ent_val.empty:
                        for depth in sorted(df_ent_val['Depth'].unique()):
                            depth_scores = df_ent_val[df_ent_val['Depth'] == depth]['Score']
                            val_metrics[f"val/avg_ent_cost_at_depth_{depth}"] = depth_scores.mean()
                            
                    wandb.log(val_metrics)
                
            if cfg.stop_vllm_during_sft:
                logger.info(f"Waiting {cfg.vllm_restart_delay} seconds for memory to clear before restarting vLLM...")
                await asyncio.sleep(cfg.vllm_restart_delay)
            
            logger.info(f"=== Completed Iteration {iter_number} ===")


@hydra.main(config_path="../../config", config_name="iterative_rl_sft", version_base=None)
def main(raw_cfg: DictConfig):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    cfg: IterativeRLSFTConfig = parse_iterative_rl_sft_config(raw_cfg)
    logger.info(f"Running Iterative RL SFT with config:\n{cfg.model_dump_json(indent=2)}")
    
    set_seed(cfg.seed)
    
    # Enable W&B if needed, but since it loops, we can init here.
    # W&B can log multiple steps over time.
    wandb_name = cfg.wandb.name if cfg.wandb.name else f"iterative_rl_sft_{cfg.clarification_model.lora_config.lora_id}"
    wandb.init(
        project=cfg.wandb.project,
        config=cfg.model_dump(),
        name=wandb_name
    )
    
    wandb.define_metric("iteration")
    wandb.define_metric("train/*", step_metric="iteration")
    wandb.define_metric("val/*", step_metric="iteration")
    
    asyncio.run(run_iterative_loop(cfg, raw_cfg))
    
    wandb.finish()

if __name__ == "__main__":
    main()
