from pydantic import BaseModel
from typing import cast
from omegaconf import DictConfig, OmegaConf

from clarification_trees_v3.config.schema import Config

class MCTSEvalConfig(Config):
    c_values: list[float] = [0.1, 0.5, 1.0, 1.414, 2.0, 5.0]
    num_trees_per_c: int = 5
    out_dir: str = "mcts_eval_results"

def parse_mcts_eval_config(cfg: DictConfig) -> MCTSEvalConfig:
    """Parse the Hydra config into a Pydantic MCTSEvalConfig object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = MCTSEvalConfig(**raw_config_dict)
    
    return config
