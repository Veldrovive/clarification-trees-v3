import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)

def safe_load_environ_path(key: str) -> Path | None:
    val = os.environ.get(key, None)
    if val is None:
        print(f"Warning: Key {key} not found in environment")
        return None
    path = Path(val)
    if not path.exists():
        print(f"Warning: Path {path} does not exist")
        return None
    return path

def safe_load_environ_str(key: str) -> str | None:
    val = os.environ.get(key, None)
    if val is None:
        return None
    return val



CLEAR_VQA_BASE_PATH = safe_load_environ_path("CLEAR_VQA_BASE_PATH")

GENERATED_TREES_PATH = safe_load_environ_path("GENERATED_TREES_PATH")

BASE_WEIGHTS_PATH = safe_load_environ_path("BASE_WEIGHTS_PATH")