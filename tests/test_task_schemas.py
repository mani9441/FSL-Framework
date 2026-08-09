"""
Unit and integration tests for Task-Specific Schemas, Task Metric Registry, Prompt-Limit Validation,
and Task-Aware Propagation across Evaluation, Comparison, Statistics, Visualization, and Reports.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from evaluation.registry import TaskMetricRegistry
from evaluation.classification import ClassificationEvaluator
from evaluation.qa_eval import QAEvaluator
from evaluation.generation import GenerationEvaluator
from evaluation.engine import EvaluationEngine
from response_repository import ResponseRecord, ResponseRepository
from experiment_engine.matrix_generator import ExperimentMatrixGenerator
from analysis.engine import StatisticalAnalysisEngine
from visualization.engine import VisualizationEngine
from reports.engine import ReportGenerationEngine
from config.manager import ConfigurationManager
from prompt_engine.validator import PromptValidator
from prompt_engine.generator import PromptGenerator
from datasets.schema import StandardRecord


class TestTaskSchemasAndValidation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_text_classification_metrics(self):
        """Test A: text_classification metrics contain accuracy/F1, and no generation metrics."""
        preds = ["World", "Sports"]
        refs = ["World", "Business"]
        metrics = ClassificationEvaluator.evaluate(preds, refs)

        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertNotIn("bleu_4", metrics)
        self.assertNotIn("rouge_l", metrics)
        self.assertNotIn("response_correctness", metrics)

    def test_question_answering_metrics(self):
        """Test B: question_answering metrics contain correctness/completeness/relevance, and no classification metrics."""
        preds = ["Washington D.C."]
        refs = ["Washington D.C."]
        metrics = QAEvaluator.evaluate(preds, refs)

        self.assertIn("response_correctness", metrics)
        self.assertIn("response_completeness", metrics)
        self.assertIn("response_relevance", metrics)
        self.assertNotIn("accuracy", metrics)
        self.assertNotIn("bleu_4", metrics)

    def test_text_generation_metrics(self):
        """Test C: text_generation metrics contain BLEU/ROUGE, and no accuracy/F1/is_correct."""
        preds = ["The government announced new economic policies today."]
        refs = ["New economic policies were announced by the government."]
        metrics = GenerationEvaluator.evaluate(preds, refs)

        self.assertIn("bleu_4", metrics)
        self.assertIn("rouge_1", metrics)
        self.assertIn("rouge_l", metrics)
        self.assertNotIn("accuracy", metrics)
        self.assertNotIn("is_correct", metrics)
        self.assertNotIn("response_correctness", metrics)

    def test_comparison_task_awareness(self):
        """Test D: comparison generation exports only applicable task metrics."""
        rec_gen = ResponseRecord(
            experiment_id="exp_gen",
            trial_index=0,
            record_id="r1",
            prompt_text="p",
            generated_text="A summary.",
            ground_truth="A reference summary.",
            task_type="text_generation",
            parsed_prediction="A summary.",
            parse_success=True,
            success=True,
            latency_seconds=1.0,
            actual_prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )

        run_dir = Path(self.temp_dir) / "run_test"
        exp_dir = run_dir / "experiments" / "exp_gen" / "responses"
        exp_dir.mkdir(parents=True, exist_ok=True)

        ResponseRepository.save_aggregate_responses("exp_gen", [rec_gen], exp_dir)
        ExperimentMatrixGenerator.generate_campaign_comparisons(run_dir, {})

        comp_csv = run_dir / "comparison.csv"
        self.assertTrue(comp_csv.exists())

        df_comp = pd.read_csv(comp_csv)
        self.assertIn("bleu_4", df_comp.columns)
        self.assertIn("rouge_l", df_comp.columns)
        self.assertNotIn("accuracy", df_comp.columns)
        self.assertNotIn("is_correct", df_comp.columns)

    def test_statistics_target_metric(self):
        """Test E: target statistical metric comes from TaskMetricRegistry."""
        target_cls = TaskMetricRegistry.get_statistical_target("text_classification")
        target_qa = TaskMetricRegistry.get_statistical_target("question_answering")
        target_gen = TaskMetricRegistry.get_statistical_target("text_generation")

        self.assertEqual(target_cls, "accuracy")
        self.assertEqual(target_qa, "response_correctness")
        self.assertEqual(target_gen, "rouge_l")

        df = pd.DataFrame([
            {"experiment_id": "e1", "task_type": "text_generation", "rouge_l": 0.5, "num_examples": 0},
            {"experiment_id": "e2", "task_type": "text_generation", "rouge_l": 0.7, "num_examples": 1},
        ])
        engine = StatisticalAnalysisEngine(output_dir=self.temp_dir)
        res = engine.analyze_dataframe(df, task_type="text_generation")
        self.assertIn("overall", res.descriptive_stats)

    def test_visualization_task_awareness(self):
        """Test F: VisualizationEngine selects task-aware figures."""
        viz = VisualizationEngine(output_dir=self.temp_dir)

        gen_metrics = {"bleu_4": 0.15, "rouge_1": 0.35, "rouge_l": 0.25}
        eff = {"latency_mean": 1.2}
        rel = {"output_consistency": 0.95}

        files = viz.generate_all_figures_from_metrics(gen_metrics, eff, rel, experiment_id="exp_gen", task_type="text_generation")
        file_names = [f.name for f in files]

        self.assertTrue(any("bleu" in n for n in file_names))
        self.assertTrue(any("rouge" in n for n in file_names))
        self.assertFalse(any("accuracy" in n for n in file_names))

    def test_reports_task_awareness(self):
        """Test G: ReportGenerationEngine includes task-specific metrics without accuracy fallback for generation."""
        rec = ResponseRecord(
            experiment_id="EXP_REP_GEN",
            trial_index=0,
            record_id="r1",
            prompt_text="p",
            generated_text="Output summary.",
            ground_truth="Ground truth summary.",
            task_type="text_generation",
        )
        eval_engine = EvaluationEngine("EXP_REP_GEN", output_dir=self.temp_dir)
        eval_res = eval_engine.evaluate_responses({0: [rec]}, task_type="text_generation")

        df = ResponseRepository.to_dataframe([rec])
        stat_engine = StatisticalAnalysisEngine(output_dir=self.temp_dir)
        stat_res = stat_engine.analyze_dataframe(df, task_type="text_generation")

        rep_engine = ReportGenerationEngine("EXP_REP_GEN", output_dir=self.temp_dir)
        exported = rep_engine.generate_report(eval_res, stat_res, df)

        self.assertTrue(exported["md"].exists())
        content = exported["md"].read_text(encoding="utf-8")
        self.assertIn("ROUGE-L", content)
        self.assertNotIn("Accuracy of 1.0000", content)

    def test_prompt_limit_validation(self):
        """Test H: over-limit prompt is rejected before inference."""
        rec = StandardRecord(id="s1", input_text="Query", context="Very long context " * 500, answer="Ans", task_type="text_generation")
        prompt_obj = PromptGenerator.generate(
            query_record=rec,
            demonstrations=[],
            strategy="instruction",
            instruction_text="Summarize.",
            max_token_limit=100,  # Deliberately low limit
        )

        self.assertFalse(prompt_obj.is_valid)
        self.assertTrue(any("exceeds model limit" in w for w in prompt_obj.validation_warnings))


if __name__ == "__main__":
    unittest.main()
