import shutil
import random
import asyncio
from pathlib import Path
from logging import getLogger
from omegaconf import DictConfig

from clarification_trees_v3.config.iterative_rl_sft_schema import IterativeRLSFTConfig
from clarification_trees_v3.definitions import GENERATED_TREES_PATH, BASE_WEIGHTS_PATH
from clarification_trees_v3.dataset.dataset import ClearVQADataset
from clarification_trees_v3.utils import SentenceAnalyzer
from clarification_trees_v3.models import construct_semantic_clusterer
from clarification_trees_v3.dataset.tree_generation import process_dataset_lazily, print_timer_tree
from clarification_trees_v3.training.eval_utils import gather_statistics, plot_metrics

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

def get_lora_path(save_dir: Path) -> Path | None:
    if (save_dir / "best_adapter").exists():
        return save_dir / "best_adapter"
    return None

async def run_train_tree_generation(
    cfg,
    raw_cfg: DictConfig,
    iter_number: int,
    iter_trees_subpath: str,
    out_dir: Path,
    ds: ClearVQADataset,
    cq_model,
    answer_model,
    sentence_analyzer: SentenceAnalyzer
):
    assert GENERATED_TREES_PATH is not None

    logger.info(f"Phase 1: Generating train trees for iteration {iter_number}...")
    check_and_clean_malformed_trees(out_dir)
    existing_count = get_completed_trees_count(out_dir)
    trees_to_generate = cfg.trees_per_iteration - existing_count
    
    if trees_to_generate > 0:
        logger.info("Loading semantic clusterer for train tree generation...")
        clusterer_gpus = cfg.devices.semantic_cluster
        clusterer = construct_semantic_clusterer(raw_cfg.semantic_cluster_model, f"cuda:{clusterer_gpus[0]}")
        
        logger.info(f"Need to generate {trees_to_generate} more train trees to reach {cfg.trees_per_iteration}.")
        rng = random.Random(cfg.seed + iter_number)
        indices = rng.sample(range(len(ds)), trees_to_generate)
        async for _ in process_dataset_lazily(
            cfg=cfg,
            dataset=ds,
            indices=indices,
            clusterer=clusterer,
            cq_model=cq_model,
            answer_model=answer_model,
            sentence_analyzer=sentence_analyzer,
            N_parallel_trees=25,
            out_dir=out_dir,
            tqdm_desc="Generating Train Trees",
            tqdm_position=0
        ):
            pass
        print_timer_tree()

        logger.info("Unloading semantic clusterer to free memory...")
        del clusterer
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info(f"Already have {existing_count} train trees for iteration {iter_number}. Skipping generation.")

    logger.info(f"Generating eval visualizations for train trees iteration {iter_number}...")
    train_eval_output_dir = GENERATED_TREES_PATH / f"{iter_trees_subpath}_train_eval_visualizations"
    df_inf, df_qp, df_ent = gather_statistics(out_dir)
    if df_inf is not None:
        plot_metrics(df_inf, df_qp, df_ent, train_eval_output_dir)

    return df_inf, df_qp, df_ent

async def run_eval_tree_generation_concurrent(
    cfg,
    raw_cfg: DictConfig,
    iter_number: int,
    eval_trees_subpath: str,
    eval_out_dir: Path,
    val_ds: ClearVQADataset,
    cq_model,
    answer_model,
    sentence_analyzer: SentenceAnalyzer
):
    assert GENERATED_TREES_PATH is not None

    logger.info(f"Eval Tree Generation for iteration {iter_number}...")
    check_and_clean_malformed_trees(eval_out_dir)
    existing_count = get_completed_trees_count(eval_out_dir)
    
    eval_trees_needed = getattr(cfg, 'eval_trees_per_iteration', 25)
    trees_to_generate = min(eval_trees_needed - existing_count, len(val_ds) - existing_count)
    
    if trees_to_generate > 0:
        logger.info("Loading semantic clusterer for eval tree generation...")
        clusterer_gpus = cfg.devices.semantic_cluster
        clusterer = construct_semantic_clusterer(raw_cfg.semantic_cluster_model, f"cuda:{clusterer_gpus[0]}")
        
        rng = random.Random(42)
        # We need an order of indices to evaluate
        all_eval_indices = rng.sample(range(len(val_ds)), len(val_ds))
        
        # Determine indices to evaluate based on how many we already generated
        target_indices = all_eval_indices[existing_count:existing_count + trees_to_generate]
        
        if target_indices:
            logger.info(f"Generating eval trees. Current count: {existing_count}. Target: {eval_trees_needed}.")
            
            async for _ in process_dataset_lazily(
                cfg=cfg,
                dataset=val_ds,
                indices=target_indices,
                clusterer=clusterer,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer,
                N_parallel_trees=25,
                out_dir=eval_out_dir,
                tqdm_desc="Generating Eval Trees",
                tqdm_position=1
            ):
                pass
                
        logger.info("Unloading semantic clusterer from eval generation to free memory...")
        del clusterer
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print_timer_tree()
    else:
        logger.info(f"Already have {existing_count} val trees. Skipping eval generation.")

    logger.info(f"Generating eval visualizations for val trees iteration {iter_number}...")
    val_eval_output_dir = GENERATED_TREES_PATH / f"{eval_trees_subpath}_eval_visualizations"
    df_inf_val, df_qp_val, df_ent_val = gather_statistics(eval_out_dir)
    if df_inf_val is not None:
        plot_metrics(df_inf_val, df_qp_val, df_ent_val, val_eval_output_dir)

    return df_inf_val, df_qp_val, df_ent_val
