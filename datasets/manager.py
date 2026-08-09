"""
Dataset Manager for FSL Research Framework.
Unified manager executing loading, processing, formatting, splitting, and metadata exporting.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

from datasets.schema import StandardRecord, DatasetMetadata
from datasets.loader import DatasetLoader
from datasets.processor import DatasetProcessor
from datasets.formatter import DatasetFormatter
from datasets.sampler import DatasetSampler
from utilities.constants import PROCESSED_DATASETS_DIR, METADATA_DATASETS_DIR
from utilities.helpers import save_json, save_jsonl, load_jsonl
from utilities.logger import get_logger

logger = get_logger("dataset_manager")


class DatasetManager:
    """Centralized dataset pipeline manager."""

    def __init__(
        self,
        dataset_name: str,
        task_type: str,
        processed_dir: Path = PROCESSED_DATASETS_DIR,
        metadata_dir: Path = METADATA_DATASETS_DIR,
    ):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.train_records: List[StandardRecord] = []
        self.test_records: List[StandardRecord] = []
        self.metadata: Optional[DatasetMetadata] = None

    def prepare_dataset(
        self,
        source_file: Optional[Union[str, Path]] = None,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[StandardRecord], List[StandardRecord], DatasetMetadata]:
        """
        Executes end-to-end dataset preparation pipeline:
        1. Load raw records
        2. Clean and validate records
        3. Format into StandardRecord instances
        4. Split into train and test sets
        5. Generate dataset metadata
        6. Persist processed records and metadata
        """
        # Step 1: Load raw records
        if source_file is not None:
            raw_records = DatasetLoader.load_from_file(source_file)
        else:
            raw_records = DatasetLoader.load_benchmark(self.dataset_name, self.task_type)

        # Step 2: Clean and validate records
        cleaned_records, dropped_count = DatasetProcessor.process_records(raw_records, self.task_type)

        # Step 3: Format into StandardRecord
        formatted_records = DatasetFormatter.format_dataset(cleaned_records, self.task_type, self.dataset_name)

        # Step 4: Split into Train and Test
        self.train_records, self.test_records = DatasetSampler.split_train_test(
            formatted_records, test_size=test_size, seed=seed
        )

        # Step 5: Compute metadata statistics
        label_dist: Dict[str, int] = {}
        for rec in formatted_records:
            if rec.label:
                label_dist[rec.label] = label_dist.get(rec.label, 0) + 1

        self.metadata = DatasetMetadata(
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            total_records=len(formatted_records),
            splits={
                "train": len(self.train_records),
                "test": len(self.test_records),
            },
            label_distribution=label_dist,
            additional_info={"dropped_records": dropped_count, "split_seed": seed},
        )

        # Step 6: Persist output files
        self.save_dataset()
        logger.info(f"Dataset '{self.dataset_name}' preparation complete.")

        return self.train_records, self.test_records, self.metadata

    def save_dataset(self) -> None:
        """Saves train/test jsonl records and dataset metadata json."""
        train_file = self.processed_dir / f"{self.dataset_name}_{self.task_type}_train.jsonl"
        test_file = self.processed_dir / f"{self.dataset_name}_{self.task_type}_test.jsonl"
        metadata_file = self.metadata_dir / f"{self.dataset_name}_{self.task_type}_metadata.json"

        save_jsonl([rec.to_dict() for rec in self.train_records], train_file)
        save_jsonl([rec.to_dict() for rec in self.test_records], test_file)

        if self.metadata:
            save_json(self.metadata.to_dict(), metadata_file)

        logger.info(f"Persisted standardized datasets to {train_file} and {test_file}")

    def load_processed_split(self, split: str = "test") -> List[StandardRecord]:
        """Loads previously saved standardized dataset split."""
        split_file = self.processed_dir / f"{self.dataset_name}_{self.task_type}_{split}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Processed split file not found: {split_file}")

        jsonl_data = load_jsonl(split_file)
        records = [StandardRecord.from_dict(item) for item in jsonl_data]
        logger.info(f"Loaded {len(records)} records from processed split '{split}'")
        return records
