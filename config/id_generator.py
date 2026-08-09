"""
Experiment ID Generator for FSL Research Framework.
Generates unique, human-readable, and deterministic experiment IDs and hashes.
"""

import hashlib
import json
import re
from typing import Dict, Any, Tuple
from utilities.logger import get_logger

logger = get_logger("id_generator")


class ExperimentIDGenerator:
    """Generates reproducible IDs and hashes for experimental runs."""

    @staticmethod
    def compute_hash(config: Dict[str, Any]) -> str:
        """Computes a deterministic 12-character MD5 hash from configuration parameters."""
        # Create sanitized copy removing transient paths or metadata
        config_copy = json.loads(json.dumps(config))
        if "experiment" in config_copy:
            config_copy["experiment"].pop("id", None)

        serialized = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def generate_id(cls, config: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generates or resolves experiment ID and hash digest.
        Returns tuple: (experiment_id, hash_digest)
        """
        hash_digest = cls.compute_hash(config)

        # Use user-specified ID if present and valid
        user_id = config.get("experiment", {}).get("id")
        if user_id and isinstance(user_id, str) and user_id.strip():
            sanitized_user_id = re.sub(r"[^\w\-]", "_", user_id.strip())
            logger.info(f"Using explicitly specified Experiment ID: '{sanitized_user_id}'")
            return sanitized_user_id, hash_digest

        # Generate structured ID: EXP_{TASK}_{SHOT}SHOT_{MODEL}_{HASH}
        task = config.get("dataset", {}).get("task", "TASK").upper()
        num_examples = config.get("few_shot", {}).get("num_examples", 0)
        model_raw = config.get("model", {}).get("name", "MODEL")

        # Sanitize model name (e.g. gpt-3.5-turbo -> GPT35TURBO)
        model_clean = re.sub(r"[^\w]", "", model_raw).upper()

        generated_id = f"EXP_{task}_{num_examples}SHOT_{model_clean}_{hash_digest}"
        logger.info(f"Generated Experiment ID: '{generated_id}' (Hash: {hash_digest})")

        return generated_id, hash_digest
