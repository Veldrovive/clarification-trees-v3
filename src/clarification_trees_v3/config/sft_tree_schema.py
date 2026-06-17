from pydantic import BaseModel
from .schema import *
import typing

class SFTTreeDatasetConfig(BaseModel):
    load_images: bool = True
    advantage_threshold: float | None = None
    top_n: int | None = None

class SFTTreeConfig(BaseModel):
    seed: int
    paths: PathsConfig
    runtime_meta: RuntimeMetaConfig
    wandb: WandbConfig
    clarification_model: ClarificationModelType
    sft_dataset: SFTTreeDatasetConfig

def parse_sft_tree_config(cfg: DictConfig) -> SFTTreeConfig:
    """Parse the Hydra config into a Pydantic SFTTreeConfig object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = typing.cast(dict, OmegaConf.to_container(cfg, resolve=True))
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = SFTTreeConfig(**raw_config_dict)
    
    return config

