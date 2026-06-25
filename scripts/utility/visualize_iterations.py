import os
import argparse
import pandas as pd
from logging import getLogger
from pathlib import Path

from clarification_trees_v3.definitions import GENERATED_TREES_PATH
from clarification_trees_v3.training.eval_utils import gather_statistics, plot_stacked_metrics

logger = getLogger(__name__)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from clarification_trees_v3.dataset.dialog_tree import DialogTree

def gather_root_rewards(base_dir: Path):
    """
    Computes the final reward at depth 0 (the root) for all trees in the directory.
    """
    root_records = []
    tree_paths = list(base_dir.rglob("tree.json"))
    
    if not tree_paths:
        return None

    for tree_path in tqdm(tree_paths, desc="Processing root rewards"):
        if not tree_path.exists():
            continue
            
        try:
            tree = DialogTree.load(tree_path, load_images=False)
            tree.compute_rewards()
            
            root_node = tree.get_node(DialogTree.ROOT)
            root_reward = root_node.reward
            if root_reward is not None and root_reward != float('-inf'):
                root_records.append({
                    "Tree_ID": tree_path.parent.name,
                    "Score": root_reward
                })
        except Exception as e:
            logger.warning(f"Failed to process root reward for {tree_path}: {e}")
            
    df_root = pd.DataFrame(root_records) if root_records else pd.DataFrame()
    return df_root

def plot_root_reward_over_iterations(combined_root: pd.DataFrame, output_dir: Path, title_prefix: str = ""):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    records = []
    if combined_root is not None and not combined_root.empty:
        for iteration, group in combined_root.groupby('Iteration'):
            records.append({
                "Iteration": iteration,
                "Metric": "Reward @ Depth 0",
                "Score": group['Score'].mean()
            })
            
    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        logger.info("No data for root reward plot.")
        return
        
    sns.lineplot(data=df_plot, x="Iteration", y="Score", hue="Metric", marker="o", palette="tab10")
    plt.title(f"{title_prefix} Reward at Depth 0 over Iterations", fontsize=14, pad=15)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Score", fontsize=12)
    
    # Place legend outside the plot
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = output_dir / f"{title_prefix.lower().replace(' ', '_')}root_reward_over_iterations.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved root reward plot to: {out_file}")

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description="Visualize Iterations")
    parser.add_argument("--parent_dir", type=str, default=str(GENERATED_TREES_PATH),
                        help="Parent directory containing the generated trees")
    parser.add_argument("--template", type=str, required=True,
                        help="Template prefix for subdirectories, e.g., 'v3_rl_sft_'")
    parser.add_argument("--start_iter", type=int, default=0,
                        help="Starting iteration number")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Maximum number of iterations")
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir)
    print(f"Running Visualize Iterations with template: {args.template} in {parent_dir}")
    
    train_dfs_inf = []
    train_dfs_qp = []
    train_dfs_ent = []
    train_dfs_root = []
    
    val_dfs_inf = []
    val_dfs_qp = []
    val_dfs_ent = []
    val_dfs_root = []
    
    for iter_number in range(args.start_iter, args.max_iters):
        logger.info(f"Gathering statistics for Iteration {iter_number}...")
        
        # Train trees
        iter_trees_subpath = f"{args.template}iter_{iter_number}"
        out_dir = parent_dir / iter_trees_subpath
        
        if out_dir.exists():
            df_inf, df_qp, df_ent = gather_statistics(out_dir)
            df_root = gather_root_rewards(out_dir)
            if df_inf is not None and not df_inf.empty:
                df_inf['Iteration'] = iter_number
                train_dfs_inf.append(df_inf)
            if df_qp is not None and not df_qp.empty:
                df_qp['Iteration'] = iter_number
                train_dfs_qp.append(df_qp)
            if df_ent is not None and not df_ent.empty:
                df_ent['Iteration'] = iter_number
                train_dfs_ent.append(df_ent)
            if df_root is not None and not df_root.empty:
                df_root['Iteration'] = iter_number
                train_dfs_root.append(df_root)
        else:
            logger.info(f"Train trees directory not found for iteration {iter_number}: {out_dir}")
            
        # Val trees
        eval_trees_subpath = f"{args.template}eval_iter_{iter_number}"
        eval_out_dir = parent_dir / eval_trees_subpath
        
        if eval_out_dir.exists():
            df_inf_val, df_qp_val, df_ent_val = gather_statistics(eval_out_dir)
            df_root_val = gather_root_rewards(eval_out_dir)
            if df_inf_val is not None and not df_inf_val.empty:
                df_inf_val['Iteration'] = iter_number
                val_dfs_inf.append(df_inf_val)
            if df_qp_val is not None and not df_qp_val.empty:
                df_qp_val['Iteration'] = iter_number
                val_dfs_qp.append(df_qp_val)
            if df_ent_val is not None and not df_ent_val.empty:
                df_ent_val['Iteration'] = iter_number
                val_dfs_ent.append(df_ent_val)
            if df_root_val is not None and not df_root_val.empty:
                df_root_val['Iteration'] = iter_number
                val_dfs_root.append(df_root_val)
        else:
            logger.info(f"Val trees directory not found for iteration {iter_number}: {eval_out_dir}")
            
    logger.info("Plotting stacked train metrics...")
    if train_dfs_inf:
        combined_train_inf = pd.concat(train_dfs_inf, ignore_index=True)
        combined_train_qp = pd.concat(train_dfs_qp, ignore_index=True) if train_dfs_qp else pd.DataFrame()
        combined_train_ent = pd.concat(train_dfs_ent, ignore_index=True) if train_dfs_ent else pd.DataFrame()
        combined_train_root = pd.concat(train_dfs_root, ignore_index=True) if train_dfs_root else pd.DataFrame()
        
        train_output_dir = parent_dir / f"{args.template}stacked_train_visualizations"
        plot_stacked_metrics(combined_train_inf, combined_train_qp, combined_train_ent, train_output_dir)
        plot_root_reward_over_iterations(combined_train_root, train_output_dir, "Train ")
    else:
        logger.info("No train metrics gathered.")
        
    logger.info("Plotting stacked val metrics...")
    if val_dfs_inf:
        combined_val_inf = pd.concat(val_dfs_inf, ignore_index=True)
        combined_val_qp = pd.concat(val_dfs_qp, ignore_index=True) if val_dfs_qp else pd.DataFrame()
        combined_val_ent = pd.concat(val_dfs_ent, ignore_index=True) if val_dfs_ent else pd.DataFrame()
        combined_val_root = pd.concat(val_dfs_root, ignore_index=True) if val_dfs_root else pd.DataFrame()
        
        val_output_dir = parent_dir / f"{args.template}stacked_val_visualizations"
        plot_stacked_metrics(combined_val_inf, combined_val_qp, combined_val_ent, val_output_dir)
        plot_root_reward_over_iterations(combined_val_root, val_output_dir, "Val ")
    else:
        logger.info("No val metrics gathered.")

if __name__ == "__main__":
    main()
