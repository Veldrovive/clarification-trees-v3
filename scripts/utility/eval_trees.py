import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig

from clarification_trees_v3.dataset.dialog_tree import DialogTree, TreeSidecar, NodeType
from clarification_trees_v3.definitions import GENERATED_TREES_PATH

def calculate_node_depths(tree: DialogTree) -> dict[int, int]:
    """
    Crawls the tree and calculates the depth of each node.
    Depth starts at 0 and increments by 1 every time a CLARIFICATION_QUESTION is asked.
    """
    depths = {}
    
    def _dfs(node_idx: int, current_depth: int):
        node = tree.get_node(node_idx)
        
        # Increment depth if we hit a clarification question
        if node.node_type == NodeType.CLARIFICATION_QUESTION:
            current_depth += 1
            
        depths[node_idx] = current_depth
        
        for child_idx in tree.get_children_idxs(node_idx):
            _dfs(child_idx, current_depth)

    # Start DFS from the root
    _dfs(DialogTree.ROOT, 0)
    return depths

def gather_statistics(base_dir: Path):
    """
    Finds all tree sidecars, loads the trees, calculates depths, 
    and aggregates scores into pandas DataFrames.
    """
    inference_records = []
    qp_records = []
    entailment_records = []
    
    # Recursively find all tree_sidecar.json files
    sidecar_paths = list(base_dir.rglob("tree_sidecar.json"))
    
    if not sidecar_paths:
        print(f"No sidecar files found in {base_dir}")
        return None, None, None

    for sidecar_path in tqdm(sidecar_paths, desc="Processing trees"):
        # The sidecar stores the path to its corresponding tree
        tree_path = sidecar_path.parent / "tree.json"
        if not tree_path.exists():
            print(f"Warning: Could not find tree for {sidecar_path}")
            continue
                
        # Load tree and sidecar and compute depths
        tree = DialogTree.load(tree_path, load_images=False)
        sidecar = TreeSidecar.load(sidecar_path)
        depths = calculate_node_depths(tree)
        tree_id = tree_path.parent.name # Useful for tracking which tree data came from
        
        # Extract scores
        inf_scores = sidecar.inference_scores
        qp_costs = sidecar.question_presence_costs
        ent_costs = sidecar.entailment_costs
        
        for node_idx, depth in depths.items():
            node = tree.get_node(node_idx)
            
            if node.node_type == NodeType.INFERENCE:
                if node_idx in inf_scores:
                    inference_records.append({
                        "Tree_ID": tree_id,
                        "Node_ID": node_idx,
                        "Depth": depth,
                        "Score": inf_scores[node_idx]
                    })
                    
            elif node.node_type == NodeType.CLARIFICATION_QUESTION:
                if node_idx in qp_costs:
                    qp_records.append({
                        "Tree_ID": tree_id,
                        "Node_ID": node_idx,
                        "Depth": depth,
                        "Score": qp_costs[node_idx]
                    })
                if node_idx in ent_costs:
                    entailment_records.append({
                        "Tree_ID": tree_id,
                        "Node_ID": node_idx,
                        "Depth": depth,
                        "Score": ent_costs[node_idx]
                    })

    # Convert to DataFrames
    df_inf = pd.DataFrame(inference_records) if inference_records else pd.DataFrame()
    df_qp = pd.DataFrame(qp_records) if qp_records else pd.DataFrame()
    df_ent = pd.DataFrame(entailment_records) if entailment_records else pd.DataFrame()
    
    return df_inf, df_qp, df_ent

def plot_metrics(df_inf: pd.DataFrame, df_qp: pd.DataFrame, df_ent: pd.DataFrame, output_dir: Path):
    """
    Generates and saves boxplots for Depth vs. Scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    datasets = [
        (df_inf, "Inference Score", "depth_vs_inference.png", "Depth vs. Inference Score (Final Answer Quality)", (0, 1)),
        (df_qp, "Question Presence Cost", "depth_vs_qp.png", "Depth vs. Question Presence Cost", (None, 0)),
        (df_ent, "Entailment Cost", "depth_vs_entailment.png", "Depth vs. Entailment Cost (Redundancy Penalty)", (None, 0))
    ]
    
    for df, ylabel, filename, title, y_lim in datasets:
        if df.empty:
            print(f"Skipping {title} - no data found.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Using a boxplot overlaid with a pointplot to show both distribution and mean trend
        # sns.boxplot(data=df, x="Depth", y="Score", color="lightblue", showfliers=False)
        sns.pointplot(data=df, x="Depth", y="Score", color="darkblue", errorbar=None, markers="D", linestyles="--")
        
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Dialogue Depth (Number of Clarification Questions)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        if y_lim[0] is not None:
            plt.ylim(bottom=y_lim[0])
        if y_lim[1] is not None:
            plt.ylim(top=y_lim[1])
        plt.tight_layout()
        
        out_file = output_dir / filename
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved plot to: {out_file}")

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
