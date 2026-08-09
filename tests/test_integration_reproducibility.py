"""
Unit tests for Module 12 — Integration & Reproducibility.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from pipeline import ReproducibleExperimentPipeline
from utilities.helpers import load_json
from model_interfaces.schema import ModelResponse


class TestIntegrationReproducibility(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_config_dict = {
            "experiment": {
                "name": "Integration Test Run",
                "description": "Integration and reproducibility verification",
                "seed": 42,
                "repeated_trials": 2,
                "output_dir": self.temp_dir,
            },
            "dataset": {
                "name": "ag_news",
                "task": "text_classification",
                "split": "test",
                "sample_size": 2,
            },
            "prompt": {
                "strategy": "hybrid",
                "instruction_text": "Classify input text.",
            },
            "few_shot": {
                "num_examples": 2,
                "ordering": "random",
                "diversity": "medium",
            },
            "model": {
                "name": "gemini-2.5-flash",
                "provider": "google",
                "temperature": 0.0,
                "max_tokens": 10,
            },
            "evaluation": {
                "metrics": ["accuracy"],
            },
        }
        self.mock_response = ModelResponse(
            generated_text="Business",
            model_name="gemini-2.5-flash",
            provider="google",
            latency_seconds=0.04,
            prompt_tokens=20,
            completion_tokens=2,
            total_tokens=22,
            estimated_cost_usd=0.0,
            success=True,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("model_interfaces.execution_manager.ModelExecutionManager.execute_with_retry")
    def test_run_pipeline_end_to_end(self, mock_exec):
        """Test full end-to-end execution of all 11 modules."""
        mock_exec.return_value = self.mock_response
        pipeline = ReproducibleExperimentPipeline(config_source=self.sample_config_dict)
        summary = pipeline.run_pipeline(sample_limit=2)

        self.assertIn(summary["status"], ["SUCCESS", "completed"])
        self.assertTrue(Path(summary["manifest_path"]).exists())

        manifest = load_json(Path(summary["manifest_path"]))
        self.assertIn("config_hash", manifest)
        self.assertIn("environment", manifest)
        self.assertIn("output_checksums_sha256", manifest)

    @patch("model_interfaces.execution_manager.ModelExecutionManager.execute_with_retry")
    def test_reproducibility_verification(self, mock_exec):
        """Test dual-run bit-for-bit reproducibility verification."""
        mock_exec.return_value = self.mock_response
        is_reproducible = ReproducibleExperimentPipeline.verify_reproducibility(
            config_source=self.sample_config_dict
        )
        self.assertTrue(is_reproducible)


if __name__ == "__main__":
    unittest.main()
