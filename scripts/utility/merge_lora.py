import dotenv
dotenv.load_dotenv()

import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import peft

from clarification_trees_v3.config.schema import Config, parse_config
from clarification_trees_v3.models.transformers_model_v2 import TransformersModelV2
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

from logging import getLogger
logger = getLogger(Path(__file__).name)

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Parse existing config
    parsed_cfg: Config = parse_config(cfg)
    model_config = parsed_cfg.clarification_model
    
    # Get custom CLI arguments if provided (using OmegaConf.select for custom hydra kwargs)
    adapter_path_str = OmegaConf.select(cfg, "adapter_path", default=None)
    output_dir_str = OmegaConf.select(cfg, "output_dir", default=None)

    lora_id = model_config.lora_config.lora_id if model_config.lora_config else None
    
    # Resolve adapter path
    if adapter_path_str is not None:
        adapter_path = Path(adapter_path_str)
    else:
        assert lora_id is not None, "lora_id must be specified in the config, or +adapter_path must be provided."
        lora_checkpoint_path = BASE_WEIGHTS_PATH / Path(parsed_cfg.paths.checkpoints.loras_subpath) / lora_id
        adapter_path = lora_checkpoint_path / "best_adapter"

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}. Please ensure the model was trained and saved.")

    # Resolve output directory
    if output_dir_str is not None:
        output_dir = Path(output_dir_str)
    else:
        assert lora_id is not None, "lora_id must be specified in the config, or +output_dir must be provided."
        output_dir = BASE_WEIGHTS_PATH / "merged_models" / lora_id

    # 1. Bypass BNB Config (disable quantization)
    if model_config.bnb_config is not None:
        logger.info("Disabling bnb_config to load base model without quantization.")
        model_config.bnb_config = None
        
    # 2. Ensure we load in half precision instead of 8-bit/4-bit int
    if model_config.torch_dtype not in ["bfloat16", "float16"]:
        logger.info(f"Changing torch_dtype from {model_config.torch_dtype} to bfloat16 for merging.")
        model_config.torch_dtype = "bfloat16"
        
    # 3. Determine device (Default to CPU for large model merging to prevent OOM)
    device = OmegaConf.select(cfg, "device", default="cpu")

    logger.info(f"Loading base model ({model_config.model_name}) in {model_config.torch_dtype} on {device}...")
    model = TransformersModelV2(model_config, device)
    
    # 4. Load PEFT Adapter directly bypassing prepare_model_for_kbit_training
    logger.info(f"Loading adapter from {adapter_path}...")
    peft_model = peft.PeftModel.from_pretrained(
        model.base_model,
        adapter_path.absolute().as_posix()
    )
    
    # 5. Merge and Unload
    logger.info("Merging adapter into base model (this may take a while)...")
    merged_model = peft_model.merge_and_unload()
    
    # 6. Save Merged Model
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir.absolute().as_posix())
    model.processor.save_pretrained(output_dir.absolute().as_posix())
    
    logger.info(f"Merge complete! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
