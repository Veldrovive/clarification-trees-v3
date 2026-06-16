from omegaconf import DictConfig
from pathlib import Path

from .transformers_model_v2 import TransformersModelV2
from .semantic_clustering import SemanticClusterer, BidirectionalEntailmentClusterer, HybridClusterer, Clusterer
import clarification_trees_v3.config.schema as schema

def construct_model(model_config: schema.ClarificationModelType | schema.AnswerModelType, device: str | int, load_lora: bool = True, loras_path: Path | None = None) -> TransformersModelV2:
    """
    Construct a TransformersModelV2 from the given config.
    """
    model = TransformersModelV2(model_config, device)
    if load_lora:
        assert loras_path is not None, "LORA/Adapter path must be specified if load_lora is True"
        
        lora_config = getattr(model_config, "lora_config", None)

        if lora_config and getattr(lora_config, "use_lora", False):
            adapter_id = lora_config.lora_id
            adapter_path = loras_path / adapter_id / "best_adapter"
            assert adapter_path.exists(), f"LoRA path {adapter_path} does not exist"
            model.load_adapter(adapter_path, adapter_name=adapter_id)
            model.set_active_lora(adapter_id)
            
        else:
            print(f"WARNING: No Adapter (LoRA) enabled for model {model_config.model_name}")
    return model

def construct_semantic_clusterer(semantic_cluster_config: DictConfig, device: str) -> Clusterer:
    """
    Construct a Clusterer from the given config.
    """
    if semantic_cluster_config.model_type == "sentence_transformers":
        return SemanticClusterer(semantic_cluster_config, device)
    elif semantic_cluster_config.model_type == "bidirectional_entailment":
        return BidirectionalEntailmentClusterer(semantic_cluster_config, device)
    elif semantic_cluster_config.model_type == "hybrid":
        return HybridClusterer(semantic_cluster_config, device)
    else:
        raise NotImplementedError(f"Model type {semantic_cluster_config.model_type} is not implemented for semantic clustering")
