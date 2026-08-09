"""
Forensic Audit & Scientific Validation Test Suite for FSL Research Framework.
Verifies prompt pre-validation, non-executed metric handling, data leakage prevention,
exact shot-count enforcement, visualization fix, and offline post-processing rebuilding.
"""

import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

from evaluation.generation import GenerationEvaluator
from evaluation.classification import ClassificationEvaluator
from evaluation.qa_eval import QAEvaluator
from visualization.engine import VisualizationEngine
from reports.latex_generator import LaTeXTableGenerator
from experiment_engine.matrix_generator import ExperimentMatrixGenerator
from datasets.sampler import DatasetSampler
from datasets.schema import StandardRecord
from prompt_engine.few_shot import ContextBuilder


class TestForensicValidation(unittest.TestCase):
    """Forensic validation test suite enforcing scientific framework guarantees."""

    def test_failed_request_returns_none_metrics(self):
        """Verify that failed/unexecuted requests return None instead of fake 0.0 metrics."""
        preds = ["", ""]
        refs = ["Reference text 1", "Reference text 2"]
        req_succs = [False, False]

        gen_metrics = GenerationEvaluator.evaluate(preds, refs, request_successes=req_succs)
        self.assertIsNone(gen_metrics["bleu_4"])
        self.assertIsNone(gen_metrics["rouge_l"])

        cls_metrics = ClassificationEvaluator.evaluate(preds, refs, request_successes=req_succs)
        self.assertIsNone(cls_metrics["accuracy"])
        self.assertIsNone(cls_metrics["f1_score"])

        qa_metrics = QAEvaluator.evaluate(preds, refs, request_successes=req_succs)
        self.assertIsNone(qa_metrics["qa_exact_match"])
        self.assertIsNone(qa_metrics["qa_f1"])

    def test_no_train_test_data_leakage(self):
        """Verify that train set demonstration IDs and test set evaluation IDs are strictly disjoint."""
        records = [
            StandardRecord(
                id=f"rec_{i}",
                input_text=f"Input text {i}",
                task_type="text_classification",
                label="World",
            )
            for i in range(20)
        ]
        train_recs, test_recs = DatasetSampler.split_train_test(records, test_size=0.3, seed=42)

        train_ids = set(r.id for r in train_recs)
        test_ids = set(r.id for r in test_recs)

        # Intersection MUST be empty
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)

        # Check few-shot candidate selection does not pull from test set
        query = test_recs[0]
        demos = ContextBuilder.build_few_shot_context(
            candidate_pool=train_recs,
            query_record=query,
            num_examples=3,
            seed=42,
        )
        demo_ids = set(d.id for d in demos)
        self.assertNotIn(query.id, demo_ids)
        self.assertEqual(len(demo_ids.intersection(test_ids)), 0)

    def test_shot_count_exactness(self):
        """Verify exact shot counts (0-shot = 0 demos, K-shot = K demos)."""
        train_recs = [
            StandardRecord(
                id=f"rec_{i}",
                input_text=f"Input text {i}",
                task_type="text_classification",
                label="World",
            )
            for i in range(10)
        ]
        query = train_recs[0]

        d0 = ContextBuilder.build_few_shot_context(train_recs, query, num_examples=0)
        self.assertEqual(len(d0), 0)

        d1 = ContextBuilder.build_few_shot_context(train_recs, query, num_examples=1)
        self.assertEqual(len(d1), 1)

        d3 = ContextBuilder.build_few_shot_context(train_recs, query, num_examples=3)
        self.assertEqual(len(d3), 3)

        d5 = ContextBuilder.build_few_shot_context(train_recs, query, num_examples=5)
        self.assertEqual(len(d5), 5)

    def test_visualization_engine_dataframe_tolist_fix(self):
        """Verify visualization engine processes DataFrame without 'DataFrame object has no attribute tolist'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            vis_engine = VisualizationEngine(output_dir=tmp_dir)

            df = pd.DataFrame([
                {
                    "experiment_id": "exp_1",
                    "model_name": "llama3.1:latest",
                    "task_type": "text_generation",
                    "num_examples": 0,
                    "latency_seconds": 1.5,
                    "success": True,
                },
                {
                    "experiment_id": "exp_2",
                    "model_name": "llama3.1:latest",
                    "task_type": "text_generation",
                    "num_examples": 1,
                    "latency_seconds": 2.5,
                    "success": True,
                },
            ])

            # Should complete cleanly without raising AttributeError
            files = vis_engine.generate_all_figures_from_dataframe(df)
            self.assertGreater(len(files), 0)

    def test_latex_table_formats_nan_as_dash(self):
        """Verify LaTeX table generator formats NaN/None as em-dash rather than 'nan' or '0.0000'."""
        df = pd.DataFrame([
            {"task": "Generation", "0-shot": 0.35, "3-shot": np.nan, "5-shot": None}
        ])
        tex = LaTeXTableGenerator.generate_latex_table(df, caption="Test Scaling Matrix")
        self.assertIn("---", tex)
        self.assertNotIn("0.0000", tex)

    def test_rebuild_campaign_from_responses_offline(self):
        """Verify offline rebuilding of campaign comparison artifacts from response files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run_test"
            exp_dir = run_dir / "experiments" / "EXP_001" / "responses"
            exp_dir.mkdir(parents=True, exist_ok=True)

            res_rec = {
                "experiment_id": "EXP_001",
                "trial_index": 0,
                "record_id": "rec_1",
                "prompt_text": "Prompt text",
                "generated_text": "Output label",
                "parsed_prediction": "World",
                "parse_success": True,
                "ground_truth": "World",
                "timestamp": "2026-08-09T01:00:00",
                "model_name": "llama3.1:latest",
                "provider": "ollama",
                "dataset_name": "ag_news",
                "task_type": "text_classification",
                "prompt_strategy": "standard_few_shot",
                "num_examples": 0,
                "ordering_strategy": "random",
                "diversity_strategy": "medium",
                "estimated_prompt_tokens": 100,
                "actual_prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "latency_seconds": 0.5,
                "success": True,
            }
            import json
            with open(exp_dir / "responses_trial_0.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps(res_rec) + "\n")

            res = ExperimentMatrixGenerator.rebuild_campaign_from_responses(run_dir)
            self.assertEqual(res["status"], "rebuilt")
            self.assertTrue((run_dir / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
