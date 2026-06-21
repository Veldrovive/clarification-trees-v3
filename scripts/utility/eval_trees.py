from pathlib import Path
import hydra
from omegaconf import DictConfig

from clarification_trees_v3.definitions import GENERATED_TREES_PATH
from clarification_trees_v3.training.eval_utils import gather_statistics, plot_metrics

@hydra.main(config_path="../../config", config_name="generate_trees", version_base=None)
def main(cfg: DictConfig):
    # Point this to where your generator script saves the trees
    data_directory = GENERATED_TREES_PATH / cfg.paths.data.trees_subpath
    output_directory = GENERATED_TREES_PATH / f"{cfg.paths.data.trees_subpath}_eval_visualizations"
    
    print(f"Gathering statistics from {data_directory}...")
    df_inf, df_qp, df_ent = gather_statistics(data_directory)
    
    print("\nSummary Statistics:")
    if df_inf is not None and not df_inf.empty:
        print(f"Total Inference Nodes Evaluated: {len(df_inf)}")
    if df_qp is not None and not df_qp.empty:
        print(f"Total Clarification Questions Evaluated: {len(df_qp)}")
        
    print("\nGenerating plots...")
    if df_inf is not None:
        plot_metrics(df_inf, df_qp, df_ent, output_directory)
    print("Done!")

if __name__ == "__main__":
    main()
