"""
Dataset Loader for FSL Research Framework.
Loads authentic benchmark datasets (AG News, SQuAD v2, CNN/DailyMail) from Hugging Face or local dataset paths.
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from utilities.helpers import load_json, load_jsonl
from utilities.logger import get_logger

logger = get_logger("dataset_loader")

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
METADATA_DIR = DATASETS_DIR / "metadata"


def download_hf_dataset_isolated(target_hf_name: str, split: str, output_file: Path) -> None:
    """
    Downloads a Hugging Face dataset via an isolated Python subprocess executed from /tmp.
    Bypasses local project directory namespace shadowing and Python 3.13 RLock issues.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Python script executed in an isolated environment outside the project root
    download_script = f"""
import sys
from datasets import load_dataset

target_hf_name = {json.dumps(target_hf_name)}
split = {json.dumps(split)}
output_path = {json.dumps(str(output_file))}

if "cnn_dailymail" in target_hf_name.lower():
    ds = load_dataset(target_hf_name, "3.0.0", split=split)
else:
    ds = load_dataset(target_hf_name, split=split)

ds.to_json(output_path, force_ascii=False)
print(f"Successfully downloaded {{len(ds)}} records to {{output_path}}")
"""

    temp_dir = tempfile.gettempdir()
    logger.info(f"Executing isolated download subprocess for '{target_hf_name}' (split={split})...")
    
    result = subprocess.run(
        [sys.executable, "-c", download_script],
        cwd=temp_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Subprocess download failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError(f"Isolated HF download failed for dataset '{target_hf_name}': {result.stderr}")

    logger.info(result.stdout.strip())


class DatasetLoader:
    """Loader for acquiring authentic benchmark dataset records."""

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Loads dataset records from a local JSON, JSONL, or CSV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            data = load_json(path)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            return [data]
        elif suffix == ".jsonl":
            return load_jsonl(path)
        elif suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported dataset file extension: '{suffix}'")

    @classmethod
    def load_benchmark(
        cls, dataset_name: str, task_type: str, split: str = "test"
    ) -> List[Dict[str, Any]]:
        """
        Loads authentic benchmark dataset by name using Hugging Face `datasets`.
        Saves raw records locally to `datasets/raw/` for offline reproducibility.
        """
        logger.info(f"Loading authentic benchmark dataset '{dataset_name}' for task '{task_type}' (split={split})")

        # Normalize HF dataset names
        hf_name_map = {
            "ag_news": "fancyzhx/ag_news",
            "AGNews": "fancyzhx/ag_news",
            "sample_ag_news": "fancyzhx/ag_news",
            "squad_v2": "rajpurkar/squad_v2",
            "SQuAD_v2": "rajpurkar/squad_v2",
            "cnn_dailymail": "abisee/cnn_dailymail",
            "CNN_DailyMail": "abisee/cnn_dailymail",
        }

        target_hf_name = hf_name_map.get(dataset_name, dataset_name)
        
        # SQuAD v2 uses 'validation' as its evaluation split instead of 'test'
        hf_split = split
        if "squad_v2" in target_hf_name.lower() and split == "test":
            hf_split = "validation"

        raw_save_dir = RAW_DIR / dataset_name
        raw_save_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_save_dir / f"{split}.jsonl"

        # 1. Check if already cached locally in datasets/raw/
        if raw_file.exists():
            cached_records = load_jsonl(raw_file)
            if len(cached_records) >= 50:
                logger.info(f"Loading {len(cached_records)} cached raw records from '{raw_file}'...")
                return cached_records
            else:
                logger.warning(f"Cached raw file '{raw_file}' contains only {len(cached_records)} records. Re-downloading authentic dataset...")

        # 2. Check fallback alias cache dir
        fallback_raw_file = RAW_DIR / "ag_news" / f"{split}.jsonl"
        if fallback_raw_file.exists() and dataset_name in ["sample_ag_news", "AGNews", "ag_news"]:
            fallback_records = load_jsonl(fallback_raw_file)
            if len(fallback_records) >= 50:
                logger.info(f"Loading {len(fallback_records)} raw records from fallback alias cache '{fallback_raw_file}'...")
                return fallback_records

        # 3. Download dataset via isolated subprocess from /tmp
        try:
            download_hf_dataset_isolated(target_hf_name, hf_split, raw_file)
            records = load_jsonl(raw_file)
            logger.info(f"Successfully loaded {len(records)} downloaded records for '{dataset_name}'.")
            return records

        except Exception as err:
            logger.error(f"Could not download dataset '{dataset_name}' via isolated Hugging Face API: {err}")
            raise RuntimeError(
                f"Failed to acquire authentic benchmark dataset '{dataset_name}' for task '{task_type}'. "
                f"Ensure internet connectivity or pre-populate local raw file at '{raw_file}'. Error: {err}"
            ) from err