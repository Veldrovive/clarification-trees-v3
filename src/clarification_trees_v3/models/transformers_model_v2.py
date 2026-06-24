from transformers import BatchEncoding
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoProcessor
from pathlib import Path
from typing import Optional
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig as PeftLoraConfig
import peft
import transformers
from PIL import Image
from typing import Literal
import logging

logger = logging.getLogger(__name__)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from clarification_trees_v3.dataset.dialog_tree import DialogTrajectory, DialogTree, TreeSidecar
from clarification_trees_v3.dataset.dialog_tree import NodeType
from clarification_trees_v3 import utils
import clarification_trees_v3.config.schema as schema

class TransformersModelV2:
    model_config: schema.ClarificationModelType | schema.AnswerModelType
    model_name: str
    device: str
    max_new_tokens: int
    image_resize_config: schema.ImageResizeConfig | None

    bnb_config: BitsAndBytesConfig | None
    base_model: transformers.PreTrainedModel
    peft_model: peft.PeftModel | None = None

    def __init__(
        self,
        model_config: schema.ClarificationModelType | schema.AnswerModelType,
        paths_config: schema.PathsConfig,
        device: str | int
    ):
        self.model_config = model_config
        self.model_name = self.model_config.model_name
        self.device = device if isinstance(device, str) and not device.isnumeric() else f"cuda:{device}"
        self.max_new_tokens = self.model_config.max_new_tokens

        if self.model_config.bnb_config:
            self.bnb_config = BitsAndBytesConfig(**self.model_config.bnb_config.model_dump())
        else:
            self.bnb_config = None
        self.image_resize_config = self.model_config.image_resize_config

        if self.image_resize_config:
            logger.info(f"Image resizing enabled: {self.image_resize_config}")

        self.base_model, self.processor = self._load_base_model(self.model_config, paths_config, self.bnb_config)

    ### BASE MODEL MANAGEMENT ###
    def _load_base_model(self, model_config: schema.ClarificationModelType | schema.AnswerModelType, paths_config: schema.PathsConfig, bnb_config: Optional[BitsAndBytesConfig] = None) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        if model_config.model_name == "qwen-3-vl-2b":
            base_model, processor = self._load_qwen_vl_model(model_config, paths_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-4b":
            base_model, processor = self._load_qwen_vl_model(model_config, paths_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-8b":
            base_model, processor = self._load_qwen_vl_model(model_config, paths_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-32b":
            base_model, processor = self._load_qwen_vl_model(model_config, paths_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-235b":
            base_model, processor = self._load_qwen_vl_model(model_config, paths_config, bnb_config)
        else:
            raise NotImplementedError(f"Model {model_config.model_name} is not implemented")

        return base_model, processor

    def _load_qwen_vl_model(self, model_config: schema.ClarificationModelType | schema.AnswerModelType, paths_config: schema.PathsConfig, bnb_config: Optional[BitsAndBytesConfig] = None) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError("Qwen3VLForConditionalGeneration is not available. Please install transformers.")

        if bnb_config is not None:
            logger.info(f"Loading model with BNB config: {bnb_config}")
        else:
            logger.info("Loading model without BNB config")
        
        desired_dtype = model_config.torch_dtype if model_config.torch_dtype is not None else "auto"
        base_model_path = schema.resolve_base_model_path(model_config.base_model_source, paths_config)
        if model_config.use_flash_attention:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                dtype=desired_dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device,
                quantization_config=bnb_config
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                dtype=desired_dtype,
                device_map=self.device,
                quantization_config=bnb_config
            )
        processor = AutoProcessor.from_pretrained(base_model_path)

        # assert isinstance(processor, transformers.PreTrainedTokenizer), f"Processor is not a PreTrainedTokenizer: {type(processor)}"

        return model, processor

    ### LORA ADAPTER MANAGEMENT ###
    def set_active_lora(self, adapter_name: str):
        """
        Sets the active LoRA adapter.
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model is loaded. Please construct a LoRA adapter first.")
        
        self.peft_model.set_adapter(adapter_name)

    def construct_lora_adapter(self, lora_config: schema.PeftConfig, adapter_name: str) -> None:
        """
        Constructs a lora adapter that can be activated using set_active_lora(adapter_name)
        adapter_name != lora_id so that we can train multiple iterations of the same adapter at once
        """
        
        model = prepare_model_for_kbit_training(self.base_model)
        
        lora_config_obj = PeftLoraConfig(**lora_config.model_dump())
        logger.info(f"Adding LoRA to model with config: {lora_config_obj}")

        if self.peft_model is None:
            # Then this is the first adapter on the model
            adapted_model = get_peft_model(model, lora_config_obj, adapter_name=adapter_name)
            assert isinstance(adapted_model, peft.PeftModel), f"Adapted model is not a PeftModel: {type(adapted_model)}"
            self.peft_model = adapted_model
        else:
            # Then this is an additional adapter on the model
            self.peft_model.add_adapter(peft_config=lora_config_obj, adapter_name=adapter_name)

    def load_adapter(self, adapter_load_dir: Path, adapter_name: str, is_trainable: bool = False) -> None:
        """
        Loads a LoRA adapter from a path and applies it to the base model.
        """
        logger.info(f"Loading adapter from {adapter_load_dir}")
        assert adapter_load_dir.exists(), f"Adapter not found at {adapter_load_dir}"
        if self.peft_model is None:
            # Then this is the first adapter on the model
            model = prepare_model_for_kbit_training(self.base_model)
            adapted_model = peft.PeftModel.from_pretrained(
                model,
                adapter_load_dir.absolute().as_posix(),
                is_trainable=is_trainable,
                adapter_name=adapter_name
            )
            assert isinstance(adapted_model, peft.PeftModel), f"Adapted model is not a PeftModel: {type(adapted_model)}"
            self.peft_model = adapted_model
        else:
            # Then this is an additional adapter on the model
            self.peft_model.load_adapter(adapter_load_dir.absolute().as_posix(), adapter_name=adapter_name, is_trainable=is_trainable)

    def save_adapter(self, adapter_save_dir: Path, adapter_name: str) -> None:
        """
        Saves the LoRA adapter to a path.
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model is loaded. Please construct a LoRA adapter first.")
        
        logger.info(f"Saving adapter to {adapter_save_dir}")
        self.peft_model.save_pretrained(adapter_save_dir.absolute().as_posix(), adapter_name=adapter_name)

    ### GENERATION & DIALOG TREE ###
    def _pad_and_resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resizes an image to fit within target dimensions while maintaining aspect ratio,
        then pads the remaining space to ensure exact output dimensions.
        """
        if not self.image_resize_config:
            return image

        target_w = self.image_resize_config.width
        target_h = self.image_resize_config.height

        pad_color = tuple(self.image_resize_config.pad_color)

        original_w, original_h = image.size
        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        # Resize with high-quality downsampling
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create new background image
        new_image = Image.new("RGB", (target_w, target_h), pad_color)
        
        # Paste resized image in the center
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image

    def _process_images_in_messages(self, messages: list[dict]):
        """
        Iterates over the message structure (list of dicts) used by `apply_chat_template`.
        Finds PIL Images and replaces them with padded/resized versions.
        """
        if not self.image_resize_config:
            return messages

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        # Some templates use "image" key with PIL object, some use "image_url"
                        # Adjust based on how DialogTrajectory stores it. 
                        # Assuming 'image' key holds the PIL object based on standard VLM usage.
                        if "image" in item and isinstance(item["image"], Image.Image):
                            item["image"] = self._pad_and_resize_image(item["image"])
        return messages

    def preprocess_generation_inputs(self, trajectory: "DialogTrajectory", role: Literal["user", "assistant"]) -> BatchEncoding:
        messages = trajectory.to_messages(model_name=self.model_name, reverse_roles=False)

        if role == "user":
            assert isinstance(self.model_config, schema.BaseClarificationModelConfig), f"Adding clarification quesiton not supported for model type {type(self.model_config)}"
            utils.add_cq_messages(messages, model_cfg=self.model_config)
        elif role == "assistant":
            raise NotImplementedError("Assistant role not implemented. I don't think we need it.")
        else:
            raise ValueError(f"Invalid role: {role}")
        
        messages = self._process_images_in_messages(messages)
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, inputs: BatchEncoding) -> list[str]:
        assert self.peft_model is not None, "No PEFT model is loaded. Please construct a LoRA adapter first."
        
        generated_ids = self.peft_model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return generated_text

    def preprocess_sft_training_inputs(self, trajectory: "DialogTrajectory", role: Literal["user", "assistant"]):
        """
        Trajectory is a list of DialogNode objects that should end with the response we want to train on.
        """
        messages = trajectory.to_messages(model_name=self.model_name, reverse_roles=False)

        if role == "user":
            assert isinstance(self.model_config, schema.BaseClarificationModelConfig), f"Adding clarification quesiton not supported for model type {type(self.model_config)}"
            utils.add_cq_messages(messages, model_cfg=self.model_config)
        elif role == "assistant":
            raise NotImplementedError("Assistant role not implemented. I don't think we need it.")
        else:
            raise ValueError(f"Invalid role: {role}")
        
        messages = self._process_images_in_messages(messages)
        
        # To get which tokens are context and which are learnable, we tokenize twice, once with the generation prompt and only the previous context
        # and once with the full context without a new generation prompt.
        # By then masking out the length of the one with the generation prompt, we can get a mask on the labels that only has the learnable tokens unmasked.
        full_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False, # We want the full text, not a prompt for generation
            return_dict=True,
            return_tensors="pt"
        )

        prompt_messages = messages[:-1] # Everything except the final assistant response
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True, # Add the token that triggers assistant generation
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = full_inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask out the prompt part in the labels
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100  # -100 is the ignore_index for CrossEntropyLoss

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full_inputs["attention_mask"][0],
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"]
        }

        if "mm_token_type_ids" in full_inputs:
            result["mm_token_type_ids"] = full_inputs["mm_token_type_ids"][0]

        return result

    @dataclass
    class RLTrainingDatapoint:
        advantage: float
        log_probs: torch.Tensor  # (prompt_tokens + res_tokens). First prompt_tokens are 0
        tokens: torch.Tensor  # (prompt_tokens + res_tokens). Concatenation of prompt tokens and response tokens
        attention_mask: torch.Tensor  # (prompt_tokens + res_tokens). I think it's just all 1s.
        action_mask: torch.Tensor  # (prompt_tokens + res_tokens). First prompt_tokens are 0, rest are 1
        prompt_pixel_values: torch.Tensor  # Handled by the tokenizer. We don't look at it.
        prompt_image_grid: torch.Tensor  # Handled by the tokenizer. We don't look at it.

    def preprocess_rl_training_inputs(self, parent_node_idx: int, tree: "DialogTree", sidecar: "TreeSidecar", role: Literal["user", "assistant"]) -> tuple[List["TransformersModelV2.RLTrainingDatapoint"], float, float]:
        """
        Extracts the different possible children of the parent node and extracts the values
        relevant to the GRPO training step.

        Needed values:
        1. The advantage for each child
        2. The old log probability for each child
        3. The tokens for each child (which come with the log probs in practice)
        4. The prompt tokens
        5. Action mask (1s where decision are made by the model and 0 otherwise)
        6. Prompt pixel values
        7. Prompt image grid
        We return a list of dicts, one for each child.

        When computing the GRPO loss, we do a forward pass through the reference SFT model
        to get the reference logprobs for the KL divergence term (with nograd). These are the
        logprobs of the tokens for each child.
        Then we do a forward pass through the current model to get the current logprobs of the
        tokens for each child with gradients intact. These are then used to compute the clipped
        policy ratio and the KL divergence term.
        The prompt length is used to mask out the prompt tokens to create the action mask.
        """
        assert role == "user", "Only user role implemented for now"

        child_cq_idxs = tree.get_children_idxs(parent_node_idx, NodeType.CLARIFICATION_QUESTION)
        
        # We use the trajectory up to the parent node to create the prompt
        to_parent_trajectory = tree.get_trajectory(parent_node_idx)
        prompt_messages = to_parent_trajectory.to_messages(model_name=self.model_name, reverse_roles=False)
        assert isinstance(self.model_config, schema.BaseClarificationModelConfig), f"Adding clarification quesiton not supported for model type {type(self.model_config)}"
        utils.add_cq_messages(prompt_messages, model_cfg=self.model_config)
        prompt_messages = self._process_images_in_messages(prompt_messages)
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True, # Add the token that triggers assistant generation
            return_dict=True,
            return_tensors="pt"
        )
        prompt_tokens = prompt_inputs["input_ids"][0]
        prompt_attention_mask = prompt_inputs["attention_mask"][0]
        prompt_pixel_values = prompt_inputs["pixel_values"]
        prompt_image_grid = prompt_inputs["image_grid_thw"]

        child_datapoints: List["TransformersModelV2.RLTrainingDatapoint"] = []
        for child_node_idx in child_cq_idxs:
            advantage = sidecar.get_node_advantage(child_node_idx)

            old_tokens_and_logprobs = sidecar.get_node_logprobs(child_node_idx)
            old_tokens = torch.tensor([token for token, logprob in old_tokens_and_logprobs], dtype=torch.long)
            old_log_probs = torch.tensor([logprob for token, logprob in old_tokens_and_logprobs], dtype=torch.float)

            # The full set of input tokens is made up of the concatenation of the prompt tokens
            # and the old tokens.
            tokens = torch.cat([prompt_tokens, old_tokens])
            prompt_logprobs_pad = torch.zeros(prompt_tokens.shape, dtype=torch.float)
            all_logprobs = torch.cat([prompt_logprobs_pad, old_log_probs])
            attention_mask = torch.cat([prompt_attention_mask, torch.ones_like(old_tokens)])
            action_mask = torch.cat([torch.zeros_like(prompt_tokens), torch.ones_like(old_tokens)])
            # The last token is the EOS token for the assistant. We don't want to train on that.
            # action_mask[-1] = 0
            if old_tokens[-1] == self.processor.tokenizer.eos_token_id:
                action_mask[-1] = 0

            child_datapoints.append(
                TransformersModelV2.RLTrainingDatapoint(
                    advantage=advantage,
                    log_probs=all_logprobs,
                    tokens=tokens,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    prompt_pixel_values=prompt_pixel_values,
                    prompt_image_grid=prompt_image_grid
                )
            )

        max_advantage = max([dp.advantage for dp in child_datapoints])
        min_advantage = min([dp.advantage for dp in child_datapoints])

        return child_datapoints, max_advantage, min_advantage
        
