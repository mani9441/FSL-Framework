"""
Configuration Loader & Validator for FSL Research Framework.
Loads, validates, and merges YAML experimental configuration files.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml

from utilities.constants import (
    CONFIG_DIR,
    SUPPORTED_TASKS,
    SUPPORTED_PROMPT_STRATEGIES,
    SUPPORTED_SHOT_COUNTS,
    SUPPORTED_ORDERING_STRATEGIES,
    SUPPORTED_DIVERSITY_LEVELS,
)
from utilities.logger import get_logger

logger = get_logger("config_loader")


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Reads and parses a YAML configuration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file {path}: {exc}")
        raise ConfigurationError(f"Invalid YAML syntax in {path}: {exc}") from exc


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validates experimental configuration schema and value constraints.
    Raises ConfigurationError if essential keys or invalid values are detected.
    """
    required_sections = ["experiment", "dataset", "prompt", "few_shot", "model"]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ConfigurationError(f"Missing required configuration section: '{section}'")

    # Validate dataset task
    task = config["dataset"].get("task")
    if task not in SUPPORTED_TASKS:
        raise ConfigurationError(
            f"Unsupported task '{task}'. Supported tasks: {SUPPORTED_TASKS}"
        )

    # Validate prompt strategy
    strategy = config["prompt"].get("strategy")
    if strategy not in SUPPORTED_PROMPT_STRATEGIES:
        raise ConfigurationError(
            f"Unsupported prompt strategy '{strategy}'. Supported strategies: {SUPPORTED_PROMPT_STRATEGIES}"
        )

    # Validate few-shot parameters
    num_examples = config["few_shot"].get("num_examples")
    if num_examples not in SUPPORTED_SHOT_COUNTS:
        logger.warning(
            f"Non-standard example count '{num_examples}'. Recommended counts: {SUPPORTED_SHOT_COUNTS}"
        )

    ordering = config["few_shot"].get("ordering")
    if ordering not in SUPPORTED_ORDERING_STRATEGIES:
        raise ConfigurationError(
            f"Unsupported ordering strategy '{ordering}'. Supported: {SUPPORTED_ORDERING_STRATEGIES}"
        )

    diversity = config["few_shot"].get("diversity")
    if diversity not in SUPPORTED_DIVERSITY_LEVELS:
        raise ConfigurationError(
            f"Unsupported diversity level '{diversity}'. Supported: {SUPPORTED_DIVERSITY_LEVELS}"
        )

    logger.info("Configuration validation passed successfully.")
    return True


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads experiment configuration. Merges with default_config.yaml if path provided.
    """
    default_config_path = CONFIG_DIR / "default_config.yaml"
    base_config = load_yaml(default_config_path) if default_config_path.exists() else {}

    if config_path is None:
        final_config = base_config
    else:
        user_config = load_yaml(config_path)
        final_config = merge_dicts(base_config, user_config)

    validate_config(final_config)
    return final_config
