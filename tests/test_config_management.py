"""
Unit tests for Module 1 — Configuration Management system.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from config.schema import ExperimentConfig
from config.validator import ConfigValidator, ConfigurationValidationError
from config.id_generator import ExperimentIDGenerator
from config.seed_manager import SeedManager
from config.path_manager import OutputPathManager
from config.manager import ConfigurationManager


class TestConfigurationManagement(unittest.TestCase):

    def setUp(self):
        self.valid_config_dict = {
            "experiment": {
                "name": "Test Trial",
                "description": "Unit test experiment",
                "seed": 42,
                "repeated_trials": 3,
                "output_dir": "results",
            },
            "dataset": {
                "name": "ag_news",
                "task": "text_classification",
                "split": "test",
                "sample_size": 50,
            },
            "prompt": {
                "strategy": "hybrid",
                "instruction_text": "Classify input text.",
            },
            "few_shot": {
                "num_examples": 3,
                "ordering": "random",
                "diversity": "medium",
            },
            "model": {
                "name": "gpt-3.5-turbo",
                "provider": "openai",
                "temperature": 0.0,
                "max_tokens": 256,
            },
            "evaluation": {
                "metrics": ["accuracy", "f1_score"],
            },
        }
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_config_validation(self):
        """Test that a valid config passes validation without error."""
        self.assertTrue(ConfigValidator.validate(self.valid_config_dict))

    def test_invalid_task_validation(self):
        """Test that an invalid task raises ConfigurationValidationError."""
        invalid_dict = dict(self.valid_config_dict)
        invalid_dict["dataset"] = dict(self.valid_config_dict["dataset"], task="unsupported_task")
        with self.assertRaises(ConfigurationValidationError):
            ConfigValidator.validate(invalid_dict)

    def test_invalid_temperature_validation(self):
        """Test out-of-range temperature raises ConfigurationValidationError."""
        invalid_dict = dict(self.valid_config_dict)
        invalid_dict["model"] = dict(self.valid_config_dict["model"], temperature=3.5)
        with self.assertRaises(ConfigurationValidationError):
            ConfigValidator.validate(invalid_dict)

    def test_id_generator(self):
        """Test deterministic ID and hash generation."""
        exp_id, hash_digest = ExperimentIDGenerator.generate_id(self.valid_config_dict)
        self.assertTrue(exp_id.startswith("EXP_TEXT_CLASSIFICATION_3SHOT_GPT35TURBO_"))
        self.assertEqual(len(hash_digest), 12)

    def test_seed_manager(self):
        """Test SeedManager trial seed generation."""
        seed_mgr = SeedManager(base_seed=100, total_trials=3)
        self.assertEqual(seed_mgr.get_trial_seed(0), 100)
        self.assertEqual(seed_mgr.get_trial_seed(1), 101)
        self.assertEqual(seed_mgr.get_trial_seed(2), 102)

    def test_path_manager_and_snapshot(self):
        """Test OutputPathManager creates subdirectories and config snapshots."""
        path_mgr = OutputPathManager(experiment_id="EXP_TEST_001", base_dir=self.temp_dir)
        paths = path_mgr.create_directories()

        self.assertTrue(paths["responses"].exists())
        self.assertTrue(paths["metrics"].exists())

        path_mgr.save_config_snapshot(self.valid_config_dict)
        self.assertTrue((paths["config"] / "experiment.json").exists())
        self.assertTrue((paths["config"] / "experiment.yaml").exists())

    def test_configuration_manager_integration(self):
        """Test full ConfigurationManager flow."""
        config_mgr = ConfigurationManager(self.valid_config_dict)
        self.assertIsNotNone(config_mgr.config)
        self.assertTrue(config_mgr.experiment_id.startswith("EXP_TEXT_CLASSIFICATION"))

        paths = config_mgr.prepare_experiment_environment()
        self.assertIn("responses", paths)


if __name__ == "__main__":
    unittest.main()
