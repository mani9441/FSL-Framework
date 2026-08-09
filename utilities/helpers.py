"""
Common Helper Functions for FSL Research Framework.
Includes seed setting, file I/O operations, timing utilities, and hashing.
"""

import os
import json
import random
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from functools import wraps

import numpy as np
from utilities.logger import get_logger

logger = get_logger("helpers")


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed across standard library random, NumPy, and PyTorch (if installed).
    Ensures experimental reproducibility across repeated runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch random seed set to {seed}")
    except ImportError:
        pass

    logger.info(f"Global random seed set to {seed}")


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Safely writes data to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Saved JSON data to {path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """Safely loads data from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data_list: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Writes a list of dictionaries to a JSON Lines file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.debug(f"Saved {len(data_list)} records to JSONL at {path}")


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads records from a JSON Lines file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class Timer:
    """
    Context manager and decorator for measuring code execution latency.
    """

    def __init__(self, name: str = "Execution"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_seconds: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_seconds = self.end_time - self.start_time
        logger.debug(f"[{self.name}] Elapsed time: {self.elapsed_seconds:.4f} seconds")


def measure_execution_time(func):
    """Decorator to measure function execution latency."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"Function '{func.__name__}' executed in {elapsed:.4f}s")
        return result, elapsed

    return wrapper


def generate_experiment_hash(config_dict: Dict[str, Any]) -> str:
    """
    Generates a deterministic MD5 hash string based on configuration dictionary.
    Used for unique identification of experimental runs.
    """
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]
