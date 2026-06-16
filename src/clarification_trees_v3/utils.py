from typing import Iterable, TYPE_CHECKING
from omegaconf import DictConfig
import os
import sys
import random
import logging
import subprocess
import numpy as np
import torch
import re
from typing import Any, cast
import spacy
from dataclasses import dataclass

if TYPE_CHECKING:
    import transformers

import clarification_trees_v3.config.schema as schema

class SentenceAnalyzer:
    @dataclass
    class SentenceMetadata:
        text: str
        length: int
        is_question: bool

    def __init__(self, model_name: str = "en_core_web_sm"):
        # Download model if not already installed
        # This can be problematic if the python environment is odd. If python -m pip will fail, this will not work
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            compatibility = spacy.cli.download_module.get_compatibility()
            version = spacy.cli.download_module.get_version(model_name, compatibility)
            filename = spacy.cli.download_module.get_model_filename(model_name, version, False)
            url = spacy.about.__download_url__ + "/" + filename
            print(f"Please download spacy model {model_name} by running: `pip install {url}`")
            sys.exit(1)
        

    def analyze_sentences(self, text: str) -> list["SentenceAnalyzer.SentenceMetadata"]:
        """
        Parses a string into sentences and returns metadata for each.
    
        Args:
            text (str): The input text to analyze.
            
        Returns:
            list[SentenceAnalyzer.SentenceMetadata]: A list of SentenceMetadata objects, where each object contains:
                - 'text': The string content of the sentence.
                - 'length': The character count of the sentence.
                - 'is_question': Boolean indicating if it ends with a question mark.
        """
        if not text:
            return []

        doc = self.nlp(text)
        
        results = []
        for sent in doc.sents:
            # Strip whitespace for accurate length and cleaner text
            sent_text = sent.text.strip()
            
            if not sent_text:
                continue
            
            results.append(SentenceAnalyzer.SentenceMetadata(
                text=sent_text,
                length=len(sent_text),
                is_question=sent_text.endswith("?")
            ))
        
        return results

def get_git_commit(short=True):
    """
    Retrieves the current git commit hash.
    
    Args:
        short (bool): If True, returns the short hash (7-8 chars).
        
    Returns:
        str: The git commit hash, or "unknown" if not a git repo.
    """
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        if short:
            cmd.insert(2, '--short')
            
        commit = subprocess.check_output(cmd).decode('ascii').strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Handles cases where git is not installed or not a git repo
        return "unknown"

def check_git_commit(target_commit):
    """
    Verifies if the current environment matches the expected git commit.
    
    Args:
        target_commit (str): The expected commit hash.
        
    Returns:
        bool: False if current commit differs from target, True otherwise.
    """
    current_commit = get_git_commit()
    
    if current_commit != target_commit:
        print(f"[Warning] Git commit mismatch! Expected: {target_commit}, Got: {current_commit}")
        return False
        
    return True

def setup_logger(name, save_dir, filename="train.log", level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.
    
    Args:
        name (str): Name of the logger.
        save_dir (str): Directory to save the log file.
        filename (str): Name of the log file.
        level (int): Logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def set_seed(seed=42):
    """
    Sets seeds for reproducibility across Python, NumPy, and PyTorch 
    (including CUDA and MPS).
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        # Deterministic algorithms (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # PyTorch MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"Random seed set to {seed} (CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()})")

def add_cq_messages(messages: list[dict], cfg: schema.Config | None = None, model_cfg: schema.ClarificationModelType | None = None) -> list[dict]:
    assert cfg is None or model_cfg is None, "Only one of cfg or model_cfg can be provided"

    if cfg is not None:
        system_prompt = cfg.clarification_model.base_prompt
    elif model_cfg is not None:
        system_prompt = model_cfg.base_prompt
    else:
        raise ValueError("Either cfg or model_cfg must be provided")

    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    return messages

def add_answer_messages(messages: list[dict], unambiguous_question: str, answers: list[str], cfg: DictConfig | None = None, model_cfg: DictConfig | None = None) -> list[dict]:
    assert cfg is None or model_cfg is None, "Only one of cfg or model_cfg can be provided"

    if cfg is not None:
        system_prompt = cfg.answer_model.answer_base_prompt
        instruction_prompt = cfg.answer_model.answer_instruction_prompt
    elif model_cfg is not None:
        system_prompt = model_cfg.answer_base_prompt
        instruction_prompt = model_cfg.answer_instruction_prompt
    else:
        raise ValueError("Either cfg or model_cfg must be provided")

    formatted_system_prompt = system_prompt.format(unambiguous_question=unambiguous_question, answers=answers)

    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": formatted_system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": instruction_prompt}]})
    return messages

def add_inference_messages(messages: list[dict], cfg: DictConfig | None = None, model_cfg: DictConfig | None = None) -> list[dict]:
    assert cfg is None or model_cfg is None, "Only one of cfg or model_cfg can be provided"

    if cfg is not None:
        system_prompt = cfg.answer_model.inference_base_prompt
        instruction_prompt = cfg.answer_model.inference_instruction_prompt
    elif model_cfg is not None:
        system_prompt = model_cfg.inference_base_prompt
        instruction_prompt = model_cfg.inference_instruction_prompt
    else:
        raise ValueError("Either cfg or model_cfg must be provided")

    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": instruction_prompt}]})
    return messages
        
def get_judge_messages(
    unambiguous_question: str,
    gold_answer: str,
    answers: list[str],
    caption: str,
    inference_response: str,
    cfg: schema.Config | None = None, model_cfg: schema.BaseAnswerModelConfig | None = None
) -> list[dict]:
    assert cfg is None or model_cfg is None, "Only one of cfg or model_cfg can be provided"

    prompt_fills = {
        "unambiguous_question": unambiguous_question,
        "gold_answer": gold_answer,
        "answers": answers,
        "caption": caption,
        "inference_response": inference_response,
    }

    if cfg is not None:
        system_prompt = cfg.answer_model.judge_prompts.base_prompt.format(**prompt_fills)
        instruction_prompt = cfg.answer_model.judge_prompts.instruction_prompt.format(**prompt_fills)
    elif model_cfg is not None:
        system_prompt = model_cfg.judge_prompts.base_prompt.format(**prompt_fills)
        instruction_prompt = model_cfg.judge_prompts.instruction_prompt.format(**prompt_fills)
    else:
        raise ValueError("Either cfg or model_cfg must be provided")

    formatted_system_prompt = system_prompt.format(unambiguous_question=unambiguous_question, gold_answer=gold_answer, answers=answers, caption=caption, inference_response=inference_response)

    messages = []
    messages.append({"role": "system", "content": [{"type": "text", "text": formatted_system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": instruction_prompt}]})
    return messages

def processes_judge_response(response: str) -> tuple[str, int]:
    # Extract reasoning
    reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=\nScore:|$)", response, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract score
    score_match = re.search(r"Score:\s*(\d+)", response, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else -1

    return reasoning, score

def tokens_to_str(tokens: list[int], tokenizer: "transformers.PreTrainedTokenizer") -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True)

def tokens_to_str_list(tokens: list[int], tokenizer: "transformers.PreTrainedTokenizer") -> list[str]:
    id_to_token_map = {v: k for k, v in tokenizer.get_vocab().items()}
    id_to_token_map[-100] = "<mask>"
    return [id_to_token_map.get(int(id), "<unk>") for id in tokens]
