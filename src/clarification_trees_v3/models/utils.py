import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from clarification_trees_v3.models import BidirectionalEntailmentClusterer, construct_semantic_clusterer
from clarification_trees_v3.models.remote_vllm_model import RemoteVLLMModel
from clarification_trees_v3.config.schema import Config
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

@asynccontextmanager
async def use_models(cfg: Config):
    if BASE_WEIGHTS_PATH is None:
        raise ValueError("BASE_WEIGHTS_PATH environment variable is required")
    
    lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
    
    clarification_model_cfg = cfg.clarification_model
    answer_model_cfg = cfg.answer_model
    clarification_model_gpus = cfg.devices.clarification
    answer_model_gpus = cfg.devices.answer
    async def _start_vllm_server(model_cfg, gpus: list[int], vllm_cfg):
        model = RemoteVLLMModel(
            model_cfg,
            cfg.paths,
            lora_checkpoint_path,
            gpus=gpus,
            port=vllm_cfg.port,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            max_model_len=vllm_cfg.max_model_len,
            max_lora_rank=vllm_cfg.max_lora_rank,
            max_num_seqs=vllm_cfg.max_num_seqs,
            max_num_batched_tokens=vllm_cfg.max_num_batched_tokens,
            log_file=Path(vllm_cfg.log_file)
        )
        await model.initialize_server()
        return model

    clarification_model, answer_model = await asyncio.gather(
        _start_vllm_server(
            clarification_model_cfg,
            clarification_model_gpus,
            cfg.remote_vllm.clarification
        ),
        _start_vllm_server(
            answer_model_cfg,
            answer_model_gpus,
            cfg.remote_vllm.answer
        )
    )

    try:
        yield clarification_model, answer_model
    finally:
        clarification_model.stop_server()
        answer_model.stop_server()
