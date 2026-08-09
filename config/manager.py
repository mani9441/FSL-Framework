"""
Configuration Manager for FSL Research Framework.
Unified manager orchestrating configuration loading, validation, ID generation, seed management, and output directory allocation.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional

from config.schema import ExperimentConfig
from config.validator import ConfigValidator
from config.id_generator import ExperimentIDGenerator
from config.seed_manager import SeedManager
from config.path_manager import OutputPathManager
from utilities.constants import CONFIG_DIR
from utilities.config_loader import load_yaml, merge_dicts
from utilities.logger import get_logger

logger = get_logger("config_manager")


class ConfigurationManager:
    """Centralized manager for loading, validating, and managing experimental parameters."""

    def __init__(self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None, base_dir: Optional[Union[str, Path]] = None):
        self.raw_config: Dict[str, Any] = {}
        self.config: Optional[ExperimentConfig] = None
        self.experiment_id: str = ""
        self.hash_digest: str = ""
        self.seed_manager: Optional[SeedManager] = None
        self.path_manager: Optional[OutputPathManager] = None

        self._load_and_initialize(config_source, base_dir=base_dir)

    def _load_and_initialize(self, config_source: Optional[Union[str, Path, Dict[str, Any]]], base_dir: Optional[Union[str, Path]] = None):
        # Load baseline default configuration
        default_config_path = CONFIG_DIR / "default_config.yaml"
        base_dict = load_yaml(default_config_path) if default_config_path.exists() else {}

        if config_source is None:
            user_dict = {}
        elif isinstance(config_source, (str, Path)):
            user_dict = load_yaml(config_source)
        elif isinstance(config_source, dict):
            user_dict = config_source
        else:
            raise ValueError(f"Invalid config_source type: {type(config_source)}")

        merged_dict = merge_dicts(base_dict, user_dict)

        # Validate configuration
        ConfigValidator.validate(merged_dict)

        # Generate Experiment ID and Hash
        self.experiment_id, self.hash_digest = ExperimentIDGenerator.generate_id(merged_dict)
        merged_dict["experiment"]["id"] = self.experiment_id

        # Instantiate dataclass model
        self.raw_config = merged_dict
        self.config = ExperimentConfig.from_dict(merged_dict)

        # Initialize SeedManager
        self.seed_manager = SeedManager(
            base_seed=self.config.experiment.seed,
            total_trials=self.config.experiment.repeated_trials,
        )

        # Initialize OutputPathManager
        target_dir = base_dir or self.config.experiment.output_dir
        self.path_manager = OutputPathManager(
            experiment_id=self.experiment_id,
            base_dir=target_dir,
        )

        logger.info(f"ConfigurationManager successfully initialized for Experiment ID '{self.experiment_id}'")

    def prepare_experiment_environment(self) -> Dict[str, Path]:
        """
        Creates output directories, sets base seed, and saves config snapshot.
        """
        # Save snapshot
        self.path_manager.save_config_snapshot(self.raw_config)
        # Apply base seed
        self.seed_manager.set_trial_seed(0)
        return self.path_manager.get_paths()

    def get_dict(self) -> Dict[str, Any]:
        """Returns raw configuration dictionary."""
        return self.raw_config
