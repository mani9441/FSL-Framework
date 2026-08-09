"""
Dataset Sampler for FSL Research Framework.
Handles train/test splitting, stratified demonstration sampling, and evaluation sub-sampling.
"""

import random
from typing import List, Tuple, Dict, Optional
from datasets.schema import StandardRecord
from utilities.logger import get_logger

logger = get_logger("dataset_sampler")


class DatasetSampler:
    """Sampler for creating train/test splits and selecting few-shot candidate pools."""

    @staticmethod
    def split_train_test(
        records: List[StandardRecord], test_size: float = 0.2, seed: int = 42
    ) -> Tuple[List[StandardRecord], List[StandardRecord]]:
        """
        Splits records into train and test lists deterministically.
        """
        if not records:
            return [], []

        shuffled = list(records)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        num_test = max(1, int(len(shuffled) * test_size)) if len(shuffled) >= 2 else int(len(shuffled) * test_size)
        test_records = shuffled[:num_test]
        train_records = shuffled[num_test:]

        if not test_records:
            raise ValueError(f"Dataset split resulted in 0 test records (total records: {len(records)}).")

        logger.info(
            f"Split dataset of size {len(records)} into Train ({len(train_records)}) and Test ({len(test_records)}) [seed={seed}]"
        )
        return train_records, test_records

    @classmethod
    def sample_demonstration_pool(
        cls, train_records: List[StandardRecord], sample_size: int = 50, seed: int = 42
    ) -> List[StandardRecord]:
        """
        Samples demonstration candidate pool from train set.
        For classification tasks, performs class-balanced stratified sampling.
        """
        if not train_records:
            return []

        if len(train_records) <= sample_size:
            return list(train_records)

        rng = random.Random(seed)
        first_record = train_records[0]

        # Stratified sampling for classification
        if first_record.task_type == "text_classification":
            label_buckets: Dict[str, List[StandardRecord]] = {}
            for rec in train_records:
                label_buckets.setdefault(rec.label, []).append(rec)

            per_class_limit = max(1, sample_size // len(label_buckets))
            sampled: List[StandardRecord] = []
            for lbl, items in label_buckets.items():
                items_copy = list(items)
                rng.shuffle(items_copy)
                sampled.extend(items_copy[:per_class_limit])

            rng.shuffle(sampled)
            logger.info(f"Class-balanced demonstration pool sampled ({len(sampled)} records).")
            return sampled[:sample_size]

        # Uniform random sampling for QA / Generation
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        sampled = shuffled[:sample_size]
        logger.info(f"Demonstration pool sampled ({len(sampled)} records).")
        return sampled

    @staticmethod
    def sample_evaluation_set(
        records: List[StandardRecord], sample_size: Optional[int] = None, seed: int = 42
    ) -> List[StandardRecord]:
        """
        Sub-samples dataset records for evaluation runs.
        """
        if not records or sample_size is None or sample_size >= len(records):
            return list(records)

        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        sampled = shuffled[:sample_size]
        logger.info(f"Evaluation dataset sub-sampled from {len(records)} to {len(sampled)} records.")
        return sampled
