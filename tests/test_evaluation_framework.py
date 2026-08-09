"""
Unit tests for Module 8 — Evaluation Framework.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from evaluation.schema import EvaluationMetricsResult
from evaluation.classification import ClassificationEvaluator
from evaluation.qa_eval import QAEvaluator
from evaluation.generation import GenerationEvaluator
from evaluation.efficiency import EfficiencyEvaluator
from evaluation.reliability import ReliabilityEvaluator
from evaluation.engine import EvaluationEngine
from response_repository import ResponseRecord


class TestEvaluationFramework(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classification_evaluator(self):
        """Test ClassificationEvaluator accuracy, precision, recall, F1."""
        preds = ["Business", "Sci/Tech", "Sports", "World"]
        refs = ["Business", "Sci/Tech", "Sports", "Sports"]

        metrics = ClassificationEvaluator.evaluate(preds, refs)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics)

    def test_qa_evaluator(self):
        """Test QAEvaluator Exact Match, Substring Match, F1, Completeness, Relevance."""
        preds = ["The Apollo program", "1969"]
        refs = ["Apollo program", "1969"]

        metrics = QAEvaluator.evaluate(preds, refs)
        self.assertGreater(metrics["response_correctness"], 0.5)
        self.assertGreater(metrics["exact_match"], 0.0)
        self.assertIn("completeness", metrics)

    def test_generation_evaluator(self):
        """Test GenerationEvaluator BLEU and ROUGE metric calculation."""
        preds = ["Solar power adoption grew 30 percent in Europe."]
        refs = ["Solar energy adoption grew 30 percent across Europe."]

        metrics = GenerationEvaluator.evaluate(preds, refs)
        self.assertGreater(metrics["bleu_1"], 0.0)
        self.assertGreater(metrics["rouge_1"], 0.0)
        self.assertGreater(metrics["rouge_l"], 0.0)

    def test_efficiency_evaluator(self):
        """Test EfficiencyEvaluator latency, token, and cost statistics."""
        rec1 = ResponseRecord(
            experiment_id="exp_1",
            trial_index=0,
            record_id="r1",
            prompt_text="p1",
            generated_text="g1",
            ground_truth="gt1",
            latency_seconds=0.10,
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
            estimated_cost_usd=0.00001,
        )
        rec2 = ResponseRecord(
            experiment_id="exp_1",
            trial_index=0,
            record_id="r2",
            prompt_text="p2",
            generated_text="g2",
            ground_truth="gt2",
            latency_seconds=0.20,
            prompt_tokens=30,
            completion_tokens=10,
            total_tokens=40,
            estimated_cost_usd=0.00002,
        )

        eff = EfficiencyEvaluator.evaluate([rec1, rec2])
        self.assertEqual(eff["latency_mean"], 0.15)
        self.assertEqual(eff["total_tokens_sum"], 65)
        self.assertAlmostEqual(eff["cost_total_usd"], 0.00003, places=5)

    def test_reliability_evaluator(self):
        """Test ReliabilityEvaluator output consistency and stability."""
        rec_t1 = ResponseRecord(experiment_id="e1", trial_index=0, record_id="r1", prompt_text="p", generated_text="Business", ground_truth="Business")
        rec_t2 = ResponseRecord(experiment_id="e1", trial_index=1, record_id="r1", prompt_text="p", generated_text="Business", ground_truth="Business")

        rel = ReliabilityEvaluator.evaluate(
            trial_records_map={0: [rec_t1], 1: [rec_t2]},
            trial_performance_scores=[1.0, 1.0],
        )
        self.assertEqual(rel["output_consistency"], 1.0)
        self.assertEqual(rel["trial_stability_std"], 0.0)

    def test_evaluation_engine_integration(self):
        """Test EvaluationEngine end-to-end evaluation pipeline and file saving."""
        rec = ResponseRecord(
            experiment_id="EXP_EVAL_TEST",
            trial_index=0,
            record_id="r1",
            prompt_text="Prompt text",
            generated_text="Business",
            ground_truth="Business",
            task_type="text_classification",
            latency_seconds=0.05,
            total_tokens=20,
        )
        engine = EvaluationEngine("EXP_EVAL_TEST", output_dir=self.temp_dir)
        res = engine.evaluate_responses({0: [rec]}, task_type="text_classification")

        self.assertIn("accuracy", res.performance_metrics)
        self.assertTrue((Path(self.temp_dir) / "metrics.json").exists())
        self.assertTrue((Path(self.temp_dir) / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
