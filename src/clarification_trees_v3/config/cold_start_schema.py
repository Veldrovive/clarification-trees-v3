from pydantic import BaseModel
from .schema import *

class ColdStartConfig(BaseModel):
    seed: int
    clarification_model: ClarificationModelType
    paths: PathsConfig
    runtime_meta: RuntimeMetaConfig
    wandb: WandbConfig
def parse_cold_start_config(cfg: DictConfig) -> ColdStartConfig:
    """Parse the Hydra config into a Pydantic ColdStartConfig object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = ColdStartConfig(**raw_config_dict)
    
    return config