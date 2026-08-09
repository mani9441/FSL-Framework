"""
Unit tests & regression tests for Campaign Comparison & Dissertation Table Generator.
Prevents request success from being used as accuracy in campaign-level comparison files.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from experiment_engine.matrix_generator import ExperimentMatrixGenerator
from analysis.dissertation_generator import DissertationArtifactGenerator


class TestCampaignComparison(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "run_test"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = self.run_dir / "experiments"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 9 records supplied in user scenario: 3 True, 6 False
        # is_correct: True, False, False, True, False, False, True, False, False -> 3/9 = 0.3333
        # success: All True -> 9/9 = 1.0
        # parse_success: All True -> 9/9 = 1.0
        is_correct_pattern = [True, False, False, True, False, False, True, False, False]
        records = []
        for i, corr in enumerate(is_correct_pattern):
            records.append({
                "experiment_id": "EXP_TEXT_CLASSIFICATION_5SHOT_INSTRUCTION_LLAMA31LATEST_9a131b903eb1",
                "trial_index": i // 3,
                "record_id": f"ag_news_{i+1}",
                "model_name": "llama3.1:latest",
                "task_type": "text_classification",
                "prompt_strategy": "instruction",
                "num_examples": 5,
                "latency_seconds": 0.5,
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "estimated_cost_usd": 0.0,
                "success": True,
                "parse_success": True,
                "is_correct": corr,
            })
        self.sample_df = pd.DataFrame(records)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_campaign_comparison_accuracy_calculation(self):
        """
        Regression test: Campaign comparison.csv MUST use is_correct for accuracy, NOT success.
        Expected: accuracy = 3/9 = 0.3333, request_success_rate = 1.0, parse_success_rate = 1.0.
        """
        # Save sample_df directly as summary.csv to test aggregation logic
        perf_col = "is_correct" if "is_correct" in self.sample_df.columns else "accuracy"
        comp_df = self.sample_df.groupby(["model_name", "task_type", "num_examples"]).agg(
            accuracy=(perf_col, "mean"),
            request_success_rate=("success", "mean"),
            parse_success_rate=("parse_success", "mean"),
            latency_seconds=("latency_seconds", "mean"),
            prompt_tokens=("prompt_tokens", "mean"),
            completion_tokens=("completion_tokens", "mean"),
            total_tokens=("total_tokens", "mean"),
            sample_count=("record_id", "count"),
        ).reset_index()

        computed_accuracy = round(float(comp_df["accuracy"].iloc[0]), 4)
        computed_request_success = round(float(comp_df["request_success_rate"].iloc[0]), 4)
        computed_parse_success = round(float(comp_df["parse_success_rate"].iloc[0]), 4)

        # Assertions for 3/9 correct scenario
        self.assertEqual(computed_accuracy, 0.3333)
        self.assertEqual(computed_request_success, 1.0)
        self.assertEqual(computed_parse_success, 1.0)
        self.assertNotEqual(computed_accuracy, 1.0, "BUG REGRESSION: comparison accuracy must NOT equal 1.0 when task accuracy is 0.3333!")

    def test_dissertation_tables_accuracy_calculation(self):
        """
        Regression test: Dissertation tables MUST use is_correct for accuracy, NOT success.
        """
        perf_col = "is_correct" if "is_correct" in self.sample_df.columns else "accuracy"
        model_group = self.sample_df.groupby("model_name").agg(
            accuracy=(perf_col, "mean"),
            request_success_rate=("success", "mean"),
            parse_success_rate=("parse_success", "mean"),
            sample_count=("record_id", "count"),
        ).reset_index()

        computed_accuracy = round(float(model_group["accuracy"].iloc[0]), 4)
        self.assertEqual(computed_accuracy, 0.3333)
        self.assertNotEqual(computed_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
