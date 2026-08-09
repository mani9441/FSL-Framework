"""
Unit tests for Module 9 — Statistical Analysis Engine.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from analysis.schema import StatisticalTestResult, StatisticalAnalysisResult
from analysis.aggregator import MetricsAggregator
from analysis.comparator import FactorComparator
from analysis.significance import SignificanceTester, compute_cohens_d
from analysis.engine import StatisticalAnalysisEngine


class TestStatisticalAnalysis(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_df = pd.DataFrame([
            {"experiment_id": "exp1", "prompt_strategy": "instruction", "num_examples": 1, "model_name": "gpt-3.5-turbo", "task_type": "text_classification", "success": 1.0},
            {"experiment_id": "exp1", "prompt_strategy": "instruction", "num_examples": 1, "model_name": "gpt-3.5-turbo", "task_type": "text_classification", "success": 0.0},
            {"experiment_id": "exp2", "prompt_strategy": "hybrid", "num_examples": 3, "model_name": "gpt-3.5-turbo", "task_type": "text_classification", "success": 1.0},
            {"experiment_id": "exp2", "prompt_strategy": "hybrid", "num_examples": 3, "model_name": "gpt-3.5-turbo", "task_type": "text_classification", "success": 1.0},
        ])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metrics_aggregator(self):
        """Test MetricsAggregator mean, median, variance, std dev."""
        values = [10.0, 20.0, 30.0, 40.0]
        stats = MetricsAggregator.compute_descriptive_stats(values)

        self.assertEqual(stats["mean"], 25.0)
        self.assertEqual(stats["median"], 25.0)
        self.assertEqual(stats["count"], 4.0)
        self.assertGreater(stats["variance"], 0.0)

    def test_factor_comparator(self):
        """Test FactorComparator grouping across experimental dimensions."""
        comparisons = FactorComparator.compare_factors(self.sample_df, metric_col="success")

        self.assertIn("num_examples", comparisons)
        self.assertEqual(len(comparisons["num_examples"]), 2)  # 1-shot vs 3-shot

    def test_significance_tester(self):
        """Test SignificanceTester T-test, ANOVA, Mann-Whitney U, and Cohen's d."""
        g1 = [0.8, 0.85, 0.9, 0.88, 0.82]
        g2 = [0.4, 0.45, 0.5, 0.48, 0.42]

        # T-Test
        tt_res = SignificanceTester.two_sample_ttest(g1, g2)
        self.assertTrue(tt_res.is_significant)
        self.assertGreater(tt_res.effect_size_cohens_d, 1.0)

        # ANOVA
        anova_res = SignificanceTester.one_way_anova([g1, g2])
        self.assertTrue(anova_res.is_significant)

        # Mann-Whitney U
        mwu_res = SignificanceTester.mann_whitney_u(g1, g2)
        self.assertTrue(mwu_res.is_significant)

    def test_statistical_analysis_engine(self):
        """Test StatisticalAnalysisEngine end-to-end execution and file saving."""
        engine = StatisticalAnalysisEngine(output_dir=self.temp_dir)
        result = engine.analyze_dataframe(self.sample_df, target_metric="success")

        self.assertIn("num_examples", result.comparative_analysis)
        self.assertTrue((Path(self.temp_dir) / "statistical_summary.json").exists())
        self.assertTrue((Path(self.temp_dir) / "comparative_analysis.csv").exists())


if __name__ == "__main__":
    unittest.main()
