from pydantic import BaseModel
from typing import cast, Literal
from omegaconf import DictConfig, OmegaConf

from clarification_trees_v3.config.schema import *

class DPOTreeDatasetConfig(BaseModel):
    positive_reward_threshold: float = 0.0
    pair_selection_method: Literal["top-bottom", "top-random", "top-all", "all-all"] = "top-random"
    batches_per_epoch: int | None = None
    eval_batches_per_epoch: int | None = None
    val_split: float = 0.1

class IterativeRLDPOConfig(Config):
    max_iters: int = 10
    trees_per_iteration: int = 500
    start_iter: int = 0
    dpo_dataset: DPOTreeDatasetConfig
    eval_trees_per_iteration: int = 50
    beta: float = 0.1

def parse_iterative_rl_dpo_config(cfg: DictConfig) -> IterativeRLDPOConfig:
    """Parse the Hydra config into a Pydantic IterativeRLDPOConfig object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = IterativeRLDPOConfig(**raw_config_dict)
    
    return config
