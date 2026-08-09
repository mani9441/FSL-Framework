"""
Configuration Validator for FSL Research Framework.
Enforces type, range, and value domain constraints for all 10 configuration items.
"""

from typing import Dict, Any, List
from utilities.constants import (
    SUPPORTED_TASKS,
    SUPPORTED_PROMPT_STRATEGIES,
    SUPPORTED_SHOT_COUNTS,
    SUPPORTED_ORDERING_STRATEGIES,
    SUPPORTED_DIVERSITY_LEVELS,
)
from utilities.logger import get_logger

logger = get_logger("config_validator")


class ConfigurationValidationError(Exception):
    """Raised when configuration validation fails with detailed error report."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration Validation Failed with errors:\n  - " + "\n  - ".join(errors)
        super().__init__(message)


class ConfigValidator:
    """Validator class enforcing domain rules across all 10 configuration items."""

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        Validates configuration dictionary.
        Returns True if valid, raises ConfigurationValidationError with error list otherwise.
        """
        errors: List[str] = []

        # 1. Required Top-Level Sections
        required_sections = ["experiment", "dataset", "prompt", "few_shot", "model"]
        for section in required_sections:
            if section not in config or not isinstance(config[section], dict):
                errors.append(f"Missing or invalid section '{section}'. Must be a dictionary.")

        if errors:
            raise ConfigurationValidationError(errors)

        exp = config["experiment"]
        ds = config["dataset"]
        prompt = config["prompt"]
        fs = config["few_shot"]
        model = config["model"]

        # 2. Validate Dataset & Task
        ds_name = ds.get("name")
        if not ds_name or not isinstance(ds_name, str):
            errors.append("Dataset 'name' must be a non-empty string.")

        task = ds.get("task")
        if task not in SUPPORTED_TASKS:
            errors.append(
                f"Invalid dataset 'task': '{task}'. Supported tasks: {SUPPORTED_TASKS}"
            )

        sample_size = ds.get("sample_size")
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size <= 0):
            errors.append(f"Dataset 'sample_size' must be a positive integer, got: {sample_size}")

        # 3. Validate Prompt Strategy
        strategy = prompt.get("strategy")
        if strategy not in SUPPORTED_PROMPT_STRATEGIES:
            errors.append(
                f"Invalid prompt 'strategy': '{strategy}'. Supported: {SUPPORTED_PROMPT_STRATEGIES}"
            )

        # 4. Validate Few-Shot Size
        num_examples = fs.get("num_examples")
        if not isinstance(num_examples, int) or num_examples < 0:
            errors.append(f"Few-shot 'num_examples' must be a non-negative integer, got: {num_examples}")
        elif num_examples not in SUPPORTED_SHOT_COUNTS:
            logger.warning(
                f"Non-standard few-shot size '{num_examples}'. Recommended baseline sizes: {SUPPORTED_SHOT_COUNTS}"
            )

        # 5. Validate Ordering Strategy
        ordering = fs.get("ordering")
        if ordering not in SUPPORTED_ORDERING_STRATEGIES:
            errors.append(
                f"Invalid ordering 'strategy': '{ordering}'. Supported: {SUPPORTED_ORDERING_STRATEGIES}"
            )

        # 6. Validate Diversity Strategy
        diversity = fs.get("diversity")
        if diversity not in SUPPORTED_DIVERSITY_LEVELS:
            errors.append(
                f"Invalid diversity 'strategy': '{diversity}'. Supported: {SUPPORTED_DIVERSITY_LEVELS}"
            )

        # 7. Validate Model Name & Provider
        model_name = model.get("name")
        if not model_name or not isinstance(model_name, str):
            errors.append("Model 'name' must be a non-empty string.")

        # 8. Validate Temperature (0.0 to 2.0)
        temperature = model.get("temperature")
        if not isinstance(temperature, (int, float)) or not (0.0 <= float(temperature) <= 2.0):
            errors.append(
                f"Model 'temperature' must be a float between 0.0 and 2.0, got: {temperature}"
            )

        # 9. Validate Max Tokens (1 to 4096)
        max_tokens = model.get("max_tokens")
        if not isinstance(max_tokens, int) or not (1 <= max_tokens <= 4096):
            errors.append(
                f"Model 'max_tokens' must be an integer between 1 and 4096, got: {max_tokens}"
            )

        # 10. Validate Trial Count & Seed
        repeated_trials = exp.get("repeated_trials", 1)
        if not isinstance(repeated_trials, int) or repeated_trials < 1:
            errors.append(
                f"Execution 'repeated_trials' count must be an integer >= 1, got: {repeated_trials}"
            )

        seed = exp.get("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append(f"Execution 'seed' must be an integer, got: {seed}")

        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} error(s).")
            raise ConfigurationValidationError(errors)

        logger.info("Configuration validation passed all checks successfully.")
        return True
