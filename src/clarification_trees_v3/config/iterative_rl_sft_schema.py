from pydantic import BaseModel
from typing import cast
from omegaconf import DictConfig, OmegaConf

from clarification_trees_v3.config.schema import *
from clarification_trees_v3.config.sft_tree_schema import SFTTreeDatasetConfig

class IterativeRLSFTConfig(Config):
    max_iters: int = 10
    trees_per_iteration: int = 500
    start_iter: int = 0
    sft_dataset: SFTTreeDatasetConfig
    eval_trees_per_iteration: int = 100
    concurrent_eval_trees: bool = True
    stop_vllm_during_sft: bool = False
    vllm_restart_delay: int = 30

def parse_iterative_rl_sft_config(cfg: DictConfig) -> IterativeRLSFTConfig:
    """Parse the Hydra config into a Pydantic IterativeRLSFTConfig object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = IterativeRLSFTConfig(**raw_config_dict)
    
    return config
