import os
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm
import hydra
from omegaconf import DictConfig
from logging import getLogger
import uuid

from clarification_trees_v3.config.mcts_eval_schema import MCTSEvalConfig, parse_mcts_eval_config
from clarification_trees_v3.definitions import GENERATED_TREES_PATH
from clarification_trees_v3.utils import set_seed, SentenceAnalyzer
from clarification_trees_v3.dataset.dataset import ClearVQADataset
from clarification_trees_v3.models.utils import use_models
from clarification_trees_v3.models import construct_semantic_clusterer
from clarification_trees_v3.dataset.tree_generation import expand_tree, process_dataset_lazily, print_timer_tree
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, visualize_tree, TreeSidecar

logger = getLogger(__name__)

def compute_spikiness_metrics(tree: DialogTree) -> tuple[float, float]:
    # Compute depths for all nodes
    depths = {DialogTree.ROOT: 0}
    
    # Process nodes in order of creation (which is topologically sorted since children are added after parents)
    for node_idx, (parent_idx, node) in enumerate(tree.nodes):
        if node_idx == DialogTree.ROOT:
            continue
        parent_depth = depths[parent_idx]
        new_depth = parent_depth + (1 if node.node_type == NodeType.CLARIFYING_ANSWER else 0)
        depths[node_idx] = new_depth
        
    # Find leaves
    is_parent = set(parent_idx for parent_idx, _ in tree.nodes)
    leaf_nodes = [i for i in range(len(tree.nodes)) if i not in is_parent]
    
    if not leaf_nodes:
        return 0.0, 0.0
        
    leaf_depths = [depths[i] for i in leaf_nodes]
    
    variance = float(np.var(leaf_depths))
    max_depth = float(np.max(leaf_depths))
    avg_depth = float(np.mean(leaf_depths))
    
    ratio = (max_depth / avg_depth) if avg_depth > 0 else 0.0
    
    return variance, ratio

async def run_evaluation(cfg: MCTSEvalConfig, raw_cfg: DictConfig):
    set_seed(cfg.seed)
    
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating c values: {cfg.c_values}")
    
    sentence_analyzer = SentenceAnalyzer()
    clusterer_gpus = cfg.devices.semantic_cluster
    clusterer = construct_semantic_clusterer(raw_cfg.semantic_cluster_model, f"cuda:{clusterer_gpus[0]}")
    
    dataset = ClearVQADataset(load_images=False, table_name="val_annotated.jsonl")
    
    results = []
    
    async with use_models(cfg) as (cq_model, answer_model):
        for c in cfg.c_values:
            logger.info(f"=== Evaluating c = {c} ===")
            
            # Create a copy of config and update c
            cfg.dialog_tree.mcts_exploration_constant = c
            
            c_out_dir = out_dir / f"c_{c}"
            c_out_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating {cfg.num_trees_per_c} trees for c={c} using lazy processing...")
            
            tree_count = 0
            async for tree_save_path, sidecar_save_path in process_dataset_lazily(
                cfg=cfg,
                dataset=dataset,
                indices=list(range(cfg.num_trees_per_c)),
                clusterer=clusterer,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer,
                out_dir=c_out_dir,
                N_parallel_trees=25,
                tqdm_desc=f"Expanding trees for c={c}",
                tqdm_position=0
            ):
                # Load tree and sidecar to compute metrics and visualize
                generated_tree = DialogTree.load(tree_save_path)
                sidecar = TreeSidecar.load(sidecar_save_path)
                
                variance, ratio = compute_spikiness_metrics(generated_tree)
                
                logger.info(f"Metrics (Tree {tree_count+1}, c={c}) - Variance: {variance:.4f}, Max/Avg Ratio: {ratio:.4f}")
                
                results.append({
                    "c_value": c,
                    "tree_index": tree_count,
                    "variance": variance,
                    "max_avg_ratio": ratio,
                    "tree_path": str(tree_save_path)
                })
                
                visualize_tree(generated_tree, sidecar, str(tree_save_path.parent / f"{tree_save_path.stem}"), view=False)
                tree_count += 1
                
            print_timer_tree()
                
    # Save and print results
    df = pd.DataFrame(results)
    csv_path = out_dir / "mcts_eval_results.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved detailed results to {csv_path}")
    
    # Summary
    summary = df.groupby("c_value")[["variance", "max_avg_ratio"]].mean().reset_index()
    logger.info("\nSummary:")
    logger.info(summary.to_string(index=False))

@hydra.main(config_path="../../config", config_name="evaluate_mcts_c", version_base=None)
def main(raw_cfg: DictConfig):
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    cfg = parse_mcts_eval_config(raw_cfg)
    logger.info(f"Running MCTS Evaluation with config:\n{cfg.model_dump_json(indent=2)}")
    
    asyncio.run(run_evaluation(cfg, raw_cfg))

if __name__ == "__main__":
    main()
