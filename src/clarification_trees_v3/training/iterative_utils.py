import shutil
import random
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

async def run_phase_1_tree_generation(
    cfg,
    raw_cfg: DictConfig,
    iter_number: int,
    iter_trees_subpath: str,
    out_dir: Path,
    ds: ClearVQADataset,
    val_ds: ClearVQADataset,
    cq_model,
    answer_model,
    sentence_analyzer: SentenceAnalyzer
):
    assert GENERATED_TREES_PATH is not None

    logger.info(f"Phase 1: Generating trees for iteration {iter_number}...")
    check_and_clean_malformed_trees(out_dir)
    existing_count = get_completed_trees_count(out_dir)
    trees_to_generate = cfg.trees_per_iteration - existing_count
    
    eval_trees_subpath = f"{cfg.paths.data.trees_subpath}_eval_iter_{iter_number}"
    eval_out_dir = GENERATED_TREES_PATH / eval_trees_subpath
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    check_and_clean_malformed_trees(eval_out_dir)
    eval_existing_count = get_completed_trees_count(eval_out_dir)
    eval_trees_to_generate = getattr(cfg, 'eval_trees_per_iteration', 50) - eval_existing_count

    if trees_to_generate > 0 or eval_trees_to_generate > 0:
        logger.info("Loading semantic clusterer for tree generation...")
        clusterer_gpus = cfg.devices.semantic_cluster
        clusterer = construct_semantic_clusterer(raw_cfg.semantic_cluster_model, f"cuda:{clusterer_gpus[0]}")
        
        if trees_to_generate > 0:
            logger.info(f"Need to generate {trees_to_generate} more train trees to reach {cfg.trees_per_iteration}.")
            rng = random.Random(cfg.seed + iter_number)
            indices = rng.sample(range(len(ds)), trees_to_generate)
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
        else:
            logger.info(f"Already have {existing_count} train trees for iteration {iter_number}. Skipping generation.")

        if eval_trees_to_generate > 0:
            logger.info(f"Need to generate {eval_trees_to_generate} more val trees.")
            rng = random.Random(42)
            eval_indices = rng.sample(range(len(val_ds)), min(getattr(cfg, 'eval_trees_per_iteration', 50), len(val_ds)))
            
            await process_dataset_lazily(
                cfg=cfg,
                dataset=val_ds,
                indices=eval_indices,
                clusterer=clusterer,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer,
                N_parallel_trees=25,
                out_dir=eval_out_dir
            )
            print_timer_tree()
        else:
            logger.info(f"Already have {eval_existing_count} val trees. Skipping.")

        logger.info("Unloading semantic clusterer to free memory...")
        del clusterer
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("All train and val trees already generated. Skipping.")

    logger.info(f"Generating eval visualizations for train trees iteration {iter_number}...")
    train_eval_output_dir = GENERATED_TREES_PATH / f"{iter_trees_subpath}_train_eval_visualizations"
    df_inf, df_qp, df_ent = gather_statistics(out_dir)
    if df_inf is not None:
        plot_metrics(df_inf, df_qp, df_ent, train_eval_output_dir)

    logger.info(f"Generating eval visualizations for val trees iteration {iter_number}...")
    val_eval_output_dir = GENERATED_TREES_PATH / f"{eval_trees_subpath}_eval_visualizations"
    df_inf_val, df_qp_val, df_ent_val = gather_statistics(eval_out_dir)
    if df_inf_val is not None:
        plot_metrics(df_inf_val, df_qp_val, df_ent_val, val_eval_output_dir)

    return df_inf, df_qp, df_ent, df_inf_val, df_qp_val, df_ent_val
