from typing import Union, Literal, Annotated
from pydantic import BaseModel, Field, model_validator
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

class DialogTreeConfig(BaseModel):
    max_depth: int = 3
    question_expansion_factor: int = 3
    answer_expansion_factor: int = 1
    question_diverse_sample_count: int = 5
    answer_diverse_sample_count: int = 5
    inference_diverse_sample_count: int = 5

class DevicesConfig(BaseModel):
    clarification: list[int]
    answer: list[int]
    semantic_cluster: list[int]
    sft: list[int] | None = None

class RemoteVLLMConfig(BaseModel):
    port: int
    gpu_memory_utilization: float = 0.90
    max_lora_rank: int = 16
    max_model_len: int = 4096
    log_file: str

class RemoteVLLMConfigs(BaseModel):
    clarification: RemoteVLLMConfig
    answer: RemoteVLLMConfig

class DataPathsConfig(BaseModel):
    trees_subpath: str

class CheckpointsPathsConfig(BaseModel):
    loras_subpath: str
    merged_models_subpath: str

class PathsConfig(BaseModel):
    data: DataPathsConfig
    checkpoints: CheckpointsPathsConfig

class LoraTrainingConfig(BaseModel):
    epochs: int = 10
    evaluate_first: bool = True
    fallback_on_no_improvement: Literal["first_epoch", "previous_lora", "last_epoch"] = "previous_lora"
    seed: int = 42
    device: str = "cuda:7"
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 25
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    patience: int = 2

class PeftConfig(BaseModel):
    r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: list[str]

class LoraConfig(BaseModel):
    use_lora: bool = True
    lora_id: str | None = None
    lora_id_postfix: str = ""
    adapter_subpath: str = "best_adapter"
    training_config: LoraTrainingConfig | None = None
    peft_config: PeftConfig | None = None

    # If use lora is true then we need the rest. If it is false then we don't need the rest.
    @model_validator(mode="after")
    def check_lora_config(self) -> "LoraConfig":
        if self.use_lora:
            if self.lora_id is None:
                raise ValueError("lora_id is required when use_lora is true")
            if self.training_config is None:
                raise ValueError("training_config is required when use_lora is true")
            if self.peft_config is None:
                raise ValueError("peft_config is required when use_lora is true")
        return self

class BnBConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float32"

class ImageResizeConfig(BaseModel):
    width: int = 512
    height: int = 512
    pad_color: list[int] = Field(default_factory=lambda: [0, 0, 0])

class RLTrainingConfig(BaseModel):
    epochs: int = 2
    evaluate_first: bool = True
    seed: int = 42
    device: str = "cuda:0"
    batch_size: int = 4
    lr: float = 1e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    train_split_ratio: float = 0.8
    clip_eps: float = 0.2
    beta: float = 0.01

    n_generation_samples: int = 10
    max_train_steps: int | None = None
    max_val_steps: int | None = None

class RLConfig(BaseModel):
    training_config: RLTrainingConfig

class BaseModelSourceConfig(BaseModel):
    source_type: Literal["huggingface", "local_merged"] = "huggingface"
    huggingface_key: str | None = None
    merged_lora_id: str | None = None

    @model_validator(mode="after")
    def check_source_config(self) -> "BaseModelSourceConfig":
        if self.source_type == "huggingface" and self.huggingface_key is None:
            raise ValueError("huggingface_key is required when source_type is 'huggingface'")
        if self.source_type == "local_merged" and self.merged_lora_id is None:
            raise ValueError("merged_lora_id is required when source_type is 'local_merged'")
        return self

# --- Clarification Models ---

class BaseClarificationModelConfig(BaseModel):
    model_name: str
    base_prompt: str

    lora_config: LoraConfig | None = None
    rl_config: RLConfig | None = None
    bnb_config: BnBConfig | None = None
    torch_dtype: str | None = None
    image_resize_config: ImageResizeConfig

class HuggingfaceClarificationModelConfig(BaseClarificationModelConfig):
    model_type: Literal["huggingface"] = "huggingface"
    base_model_source: BaseModelSourceConfig
    use_flash_attention: bool = False
    max_new_tokens: int = 128

# --- Answer Models ---

class JudgePromptsConfig(BaseModel):
    n_judgements: int = 5
    base_prompt: str
    instruction_prompt: str

class BaseAnswerModelConfig(BaseModel):
    model_name: str
    answer_base_prompt: str
    answer_instruction_prompt: str
    inference_base_prompt: str
    inference_instruction_prompt: str
    judge_prompts: JudgePromptsConfig
    lora_config: LoraConfig | None = None
    bnb_config: BnBConfig | None = None
    torch_dtype: str | None = None
    image_resize_config: ImageResizeConfig

class HuggingfaceAnswerModelConfig(BaseAnswerModelConfig):
    model_type: Literal["huggingface"] = "huggingface"
    base_model_source: BaseModelSourceConfig
    use_flash_attention: bool = False
    max_new_tokens: int = 128

# --- Semantic Cluster Models ---

class BidirectionalEntailmentClustererConfig(BaseModel):
    model_type: Literal["bidirectional_entailment_clusterer"] = "bidirectional_entailment_clusterer"
    cross_encoder_key: str = "cross-encoder/nli-deberta-v3-base"
    exemplar_selection_method: str = "random"
    entailment_threshold: float = 0.5

class HybridClustererConfig(BaseModel):
    model_type: Literal["hybrid_clusterer"] = "hybrid_clusterer"
    sentence_transformers_key: str = "all-MiniLM-L6-v2"
    cross_encoder_key: str = "cross-encoder/nli-deberta-v3-base"
    similarity_threshold: float = 0.10
    entailment_threshold: float = 0.3
    exemplar_selection_method: str = "shortest"

class SentenceTransformersClustererConfig(BaseModel):
    model_type: Literal["sentence_transformers_clusterer"] = "sentence_transformers_clusterer"
    model_name: str = "all-MiniLM-L6-v2"
    sentence_transformers_key: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.10
    clustering_method: str = "agglomerative"
    exemplar_selection_method: str = "random"

# --- Runtime & Root Config ---

class RuntimeMetaConfig(BaseModel):
    git_commit: str | None = None
    git_strict: bool = True

class WandbConfig(BaseModel):
    project: str
    name: str | None = None


# Define the Discriminated Unions for Polymorphism
ClarificationModelType = Annotated[
    Union[HuggingfaceClarificationModelConfig], # Add future models to this Union
    Field(discriminator="model_type")
]

AnswerModelType = Annotated[
    Union[HuggingfaceAnswerModelConfig], # Add future models to this Union
    Field(discriminator="model_type")
]

SemanticClusterModelType = Annotated[
    Union[
        BidirectionalEntailmentClustererConfig,
        HybridClustererConfig,
        SentenceTransformersClustererConfig
    ],
    Field(discriminator="model_type")
]

class Config(BaseModel):
    seed: int
    dialog_tree: DialogTreeConfig
    devices: DevicesConfig
    remote_vllm: RemoteVLLMConfigs
    paths: PathsConfig
    runtime_meta: RuntimeMetaConfig
    wandb: WandbConfig

    clarification_model: ClarificationModelType
    answer_model: AnswerModelType
    semantic_cluster_model: SemanticClusterModelType

def parse_config(cfg: DictConfig) -> Config:
    """Parse the Hydra config into a Pydantic Config object."""
    # 1. Resolve Hydra interpolations and convert to standard dict
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 2. Pass dict to Pydantic for strict polymorphic validation
    config = Config(**raw_config_dict)
    
    return config

def resolve_base_model_path(source_config: BaseModelSourceConfig, paths_config: PathsConfig) -> str:
    if source_config.source_type == "huggingface":
        assert source_config.huggingface_key is not None
        return source_config.huggingface_key
    elif source_config.source_type == "local_merged":
        assert source_config.merged_lora_id is not None
        assert BASE_WEIGHTS_PATH is not None, "BASE_WEIGHTS_PATH environment variable is required for local merged weights."
        
        path = BASE_WEIGHTS_PATH / paths_config.checkpoints.merged_models_subpath / source_config.merged_lora_id
        if not path.exists():
            print(f"Warning: Merged local model not found at {path}")
        return str(path)
    else:
        raise ValueError(f"Unknown source type: {source_config.source_type}")

if __name__ == "__main__":
    # How to integrate this with Hydra
    @hydra.main(version_base=None, config_path="./", config_name="config")
    def main(cfg: DictConfig):
        config = parse_config(cfg)
        
        print("Pydantic Config Loaded Successfully!")
        print(f"Seed: {config.seed}")
        print(f"Answer Model Keys: {config.answer_model.model_hf_transformers_key}")

    main()
