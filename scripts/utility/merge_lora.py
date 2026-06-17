import dotenv
dotenv.load_dotenv()

import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import peft

from clarification_trees_v3.config.cold_start_schema import ColdStartConfig, parse_cold_start_config
from clarification_trees_v3.models.transformers_model_v2 import TransformersModelV2
from clarification_trees_v3.definitions import BASE_WEIGHTS_PATH

from logging import getLogger
logger = getLogger(Path(__file__).name)

@hydra.main(config_path="../../config", config_name="cold_start_config", version_base=None)
def main(cfg: DictConfig):
    # Parse existing config
    # hi <3
    parsed_cfg: ColdStartConfig = parse_cold_start_config(cfg)
    model_config = parsed_cfg.clarification_model

    lora_id = model_config.lora_config.lora_id
    assert lora_id is not None
    assert BASE_WEIGHTS_PATH
    loras_base_path = BASE_WEIGHTS_PATH / parsed_cfg.paths.checkpoints.loras_subpath
    merged_models_base_path = BASE_WEIGHTS_PATH / parsed_cfg.paths.checkpoints.merged_models_subpath

    adapter_path = loras_base_path / lora_id / "best_adapter"
    assert adapter_path.exists(), f"Cannot find adapter to merge at {adapter_path}"
    merged_model_path = merged_models_base_path / lora_id
    assert not merged_model_path.exists(), f"Merged model already exists at {merged_model_path}"
    output_dir = merged_model_path

    # 1. Bypass BNB Config (disable quantization)
    if model_config.bnb_config is not None:
        logger.info("Disabling bnb_config to load base model without quantization.")
        model_config.bnb_config = None
        
    # 2. Ensure we load in half precision instead of 8-bit/4-bit int
    if model_config.torch_dtype not in ["bfloat16", "float16"]:
        logger.info(f"Changing torch_dtype from {model_config.torch_dtype} to bfloat16 for merging.")
        model_config.torch_dtype = "bfloat16"
        
    # 3. Determine device (Default to CPU for large model merging to prevent OOM)
    device = "cpu"

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
