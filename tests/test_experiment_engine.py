"""
Unit tests for Module 6 — Experiment Execution Engine.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from config.manager import ConfigurationManager
from datasets.schema import StandardRecord
from experiment_engine.schema import ExperimentTrialResult, ProgressState
from experiment_engine.progress_tracker import ProgressTracker
from experiment_engine.trial_manager import TrialManager
from experiment_engine.controller import ExperimentController
from model_interfaces.schema import ModelResponse


class TestExperimentEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_config_dict = {
            "experiment": {
                "name": "Engine Test Run",
                "description": "Unit test experiment execution",
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
                "instruction_text": "Classify text into news category.",
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

    def test_progress_tracker(self):
        """Test ProgressTracker metric updates and status formatting."""
        tracker = ProgressTracker(total_tasks=5)
        tracker.start_task()
        tracker.complete_task()

        tracker.start_task()
        tracker.fail_task()

        self.assertEqual(tracker.state.completed, 1)
        self.assertEqual(tracker.state.failed, 1)
        self.assertEqual(tracker.state.running, 0)

        log_path = Path(self.temp_dir) / "progress.json"
        tracker.save_progress_state(log_path)
        self.assertTrue(log_path.exists())

    @patch("model_interfaces.execution_manager.ModelExecutionManager.execute_with_retry")
    def test_trial_manager(self, mock_exec):
        """Test TrialManager running a single trial."""
        mock_exec.return_value = self.mock_response
        cfg_mgr = ConfigurationManager(self.sample_config_dict)
        train_recs = [
            StandardRecord(id="tr_1", task_type="text_classification", input_text="Oil prices rise.", answer="Business", label="Business"),
            StandardRecord(id="tr_2", task_type="text_classification", input_text="Match won by team.", answer="Sports", label="Sports"),
        ]
        eval_recs = [
            StandardRecord(id="ev_1", task_type="text_classification", input_text="Market index surges.", answer="Business", label="Business"),
        ]

        trial_mgr = TrialManager(cfg_mgr, train_recs, eval_recs)
        results = trial_mgr.run_trial(trial_index=0)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record_id, "ev_1")
        self.assertIn("prompt_object", results[0].to_dict())

    @patch("model_interfaces.execution_manager.ModelExecutionManager.execute_with_retry")
    def test_experiment_controller_pipeline(self, mock_exec):
        """Test ExperimentController executing end-to-end experiment pipeline."""
        mock_exec.return_value = self.mock_response
        controller = ExperimentController(config_source=self.sample_config_dict)
        summary = controller.run_experiment(sample_limit=2)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["total_trials"], 2)

        resp_dir = Path(summary["output_paths"]["responses"])
    @patch("model_interfaces.execution_manager.ModelExecutionManager.execute_with_retry")
    def test_experiment_trial_result_schema_qa_pipeline(self, mock_exec):
        """Test ExperimentTrialResult schema processing for QA pipeline without AttributeError."""
        qa_resp = ModelResponse(
            generated_text="The answer is: Mars",
            model_name="gemini-2.5-flash",
            provider="google",
            latency_seconds=0.04,
            prompt_tokens=20,
            completion_tokens=4,
            total_tokens=24,
            estimated_cost_usd=0.0,
            success=True,
        )
        mock_exec.return_value = qa_resp
        qa_config_dict = dict(self.sample_config_dict)
        qa_config_dict["dataset"] = {
            "name": "squad_v2",
            "task": "question_answering",
            "split": "test",
            "sample_size": 1,
        }
        
        from pipeline import ReproducibleExperimentPipeline
        pipeline = ReproducibleExperimentPipeline(config_source=qa_config_dict, base_dir=Path(self.temp_dir))
        res = pipeline.run_pipeline(sample_limit=1)
        self.assertIn(res["status"], ["SUCCESS", "completed", "PARTIAL"])
        self.assertNotIn("context", res.get("reason", ""))


if __name__ == "__main__":
    unittest.main()
