import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from logging import getLogger

from clarification_trees_v3.config.iterative_rl_sft_schema import IterativeRLSFTConfig, parse_iterative_rl_sft_config
from clarification_trees_v3.definitions import GENERATED_TREES_PATH
from clarification_trees_v3.training.eval_utils import gather_statistics, plot_stacked_metrics

logger = getLogger(__name__)

@hydra.main(config_path="../../config", config_name="iterative_rl_sft", version_base=None)
def main(raw_cfg: DictConfig):
    cfg: IterativeRLSFTConfig = parse_iterative_rl_sft_config(raw_cfg)
    print(f"Running Visualize Iterations with config:\n{cfg.model_dump_json(indent=2)}")
    
    assert GENERATED_TREES_PATH is not None, "GENERATED_TREES_PATH is not defined"
    
    train_dfs_inf = []
    train_dfs_qp = []
    train_dfs_ent = []
    
    val_dfs_inf = []
    val_dfs_qp = []
    val_dfs_ent = []
    
    for iter_number in range(cfg.start_iter, cfg.max_iters):
        logger.info(f"Gathering statistics for Iteration {iter_number}...")
        
        # Train trees
        iter_trees_subpath = f"{cfg.paths.data.trees_subpath}_iter_{iter_number}"
        out_dir = GENERATED_TREES_PATH / iter_trees_subpath
        
        if out_dir.exists():
            df_inf, df_qp, df_ent = gather_statistics(out_dir)
            if df_inf is not None and not df_inf.empty:
                df_inf['Iteration'] = iter_number
                train_dfs_inf.append(df_inf)
            if df_qp is not None and not df_qp.empty:
                df_qp['Iteration'] = iter_number
                train_dfs_qp.append(df_qp)
            if df_ent is not None and not df_ent.empty:
                df_ent['Iteration'] = iter_number
                train_dfs_ent.append(df_ent)
        else:
            logger.info(f"Train trees directory not found for iteration {iter_number}: {out_dir}")
            
        # Val trees
        eval_trees_subpath = f"{cfg.paths.data.trees_subpath}_eval_iter_{iter_number}"
        eval_out_dir = GENERATED_TREES_PATH / eval_trees_subpath
        
        if eval_out_dir.exists():
            df_inf_val, df_qp_val, df_ent_val = gather_statistics(eval_out_dir)
            if df_inf_val is not None and not df_inf_val.empty:
                df_inf_val['Iteration'] = iter_number
                val_dfs_inf.append(df_inf_val)
            if df_qp_val is not None and not df_qp_val.empty:
                df_qp_val['Iteration'] = iter_number
                val_dfs_qp.append(df_qp_val)
            if df_ent_val is not None and not df_ent_val.empty:
                df_ent_val['Iteration'] = iter_number
                val_dfs_ent.append(df_ent_val)
        else:
            logger.info(f"Val trees directory not found for iteration {iter_number}: {eval_out_dir}")
            
    logger.info("Plotting stacked train metrics...")
    if train_dfs_inf:
        combined_train_inf = pd.concat(train_dfs_inf, ignore_index=True)
        combined_train_qp = pd.concat(train_dfs_qp, ignore_index=True) if train_dfs_qp else pd.DataFrame()
        combined_train_ent = pd.concat(train_dfs_ent, ignore_index=True) if train_dfs_ent else pd.DataFrame()
        
        train_output_dir = GENERATED_TREES_PATH / "stacked_train_visualizations"
        plot_stacked_metrics(combined_train_inf, combined_train_qp, combined_train_ent, train_output_dir)
    else:
        logger.info("No train metrics gathered.")
        
    logger.info("Plotting stacked val metrics...")
    if val_dfs_inf:
        combined_val_inf = pd.concat(val_dfs_inf, ignore_index=True)
        combined_val_qp = pd.concat(val_dfs_qp, ignore_index=True) if val_dfs_qp else pd.DataFrame()
        combined_val_ent = pd.concat(val_dfs_ent, ignore_index=True) if val_dfs_ent else pd.DataFrame()
        
        val_output_dir = GENERATED_TREES_PATH / "stacked_val_visualizations"
        plot_stacked_metrics(combined_val_inf, combined_val_qp, combined_val_ent, val_output_dir)
    else:
        logger.info("No val metrics gathered.")

if __name__ == "__main__":
    main()
