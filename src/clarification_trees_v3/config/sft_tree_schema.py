from pydantic import BaseModel
from .schema import *

class SFTTreeDatasetConfig(BaseModel):
    load_images: bool = True
    advantage_threshold: float | None = None
    top_n: int | None = None

class SFTTreeConfig(BaseModel):
    seed: int
    dialog_tree: DialogTreeConfig
    devices: DevicesConfig
    remote_vllm: RemoteVLLMConfigs
    paths: PathsConfig
    runtime_meta: RuntimeMetaConfig
    clarification_model: ClarificationModelType
    answer_model: AnswerModelType
    semantic_cluster_model: SemanticClusterModelType
    sft_dataset: SFTTreeDatasetConfig

def parse_sft_tree_config(cfg: DictConfig) -> SFTTreeConfig:
    """Parse the Hydra config into a Pydantic SFTTreeConfig object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = SFTTreeConfig(**raw_config_dict)
    
    return config

def get_base_config(sft_tree_config: SFTTreeConfig) -> Config:
    return Config(
        seed=sft_tree_config.seed,
        dialog_tree=sft_tree_config.dialog_tree,
        devices=sft_tree_config.devices,
        remote_vllm=sft_tree_config.remote_vllm,
        paths=sft_tree_config.paths,
        runtime_meta=sft_tree_config.runtime_meta,
        clarification_model=sft_tree_config.clarification_model,
        answer_model=sft_tree_config.answer_model,
        semantic_cluster_model=sft_tree_config.semantic_cluster_model
    )
