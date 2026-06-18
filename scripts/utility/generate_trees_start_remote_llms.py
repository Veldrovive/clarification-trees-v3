import asyncio
from pathlib import Path
import hydra
from omegaconf import DictConfig

from clarification_trees_v3.models.remote_vllm_model import RemoteVLLMModel
from clarification_trees_v3.config.schema import parse_config, Config
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

async def start_servers(cfg: Config):
    if BASE_WEIGHTS_PATH is None:
        raise ValueError("BASE_WEIGHTS_PATH environment variable is required")
    
    lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(cfg.paths.checkpoints.loras_subpath)
    merged_models_path = BASE_WEIGHTS_PATH / cfg.paths.checkpoints.merged_models_subpath
    
    clarification_model_cfg = cfg.clarification_model
    answer_model_cfg = cfg.answer_model
    clarification_model_gpus = cfg.devices.clarification
    answer_model_gpus = cfg.devices.answer
    clarification_model_port = cfg.remote_vllm.clarification.port
    answer_model_port = cfg.remote_vllm.answer.port
    clarification_model_log_file = Path(cfg.remote_vllm.clarification.log_file)
    answer_model_log_file = Path(cfg.remote_vllm.answer.log_file)

    async def _start_vllm_server(model_cfg, gpus: list[int], port: int, log_file: Path):
        model = RemoteVLLMModel(
            model_cfg,
            cfg.paths,
            lora_checkpoint_path,
            gpus=gpus,
            port=port,
            log_file=log_file
        )
        print(f"Initializing server on port {port} with GPUs {gpus}...")
        await model.initialize_server()
        print(f"Server initialized on port {port}.")
        return model

    clarification_model, answer_model = await asyncio.gather(
        _start_vllm_server(
            clarification_model_cfg,
            clarification_model_gpus,
            clarification_model_port,
            clarification_model_log_file
        ),
        _start_vllm_server(
            answer_model_cfg,
            answer_model_gpus,
            answer_model_port,
            answer_model_log_file
        )
    )

    print("Servers started successfully. Press Ctrl+C to exit.", flush=True)
    try:
        # Keep the event loop running to keep servers alive
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        print("Stopping servers...")
        clarification_model.stop_server()
        answer_model.stop_server()

@hydra.main(config_path="../../config", config_name="generate_trees", version_base=None)
def main(raw_cfg: DictConfig):
    cfg = parse_config(raw_cfg)
    try:
        asyncio.run(start_servers(cfg))
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
