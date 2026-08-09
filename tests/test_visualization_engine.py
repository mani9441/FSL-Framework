"""
Unit tests for Module 10 — Visualisation Engine.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from visualization.charts import (
    plot_bar_chart,
    plot_line_chart,
    plot_scatter_chart,
    plot_heatmap,
    plot_reliability_chart,
)
from visualization.engine import VisualizationEngine


class TestVisualizationEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plot_bar_chart(self):
        """Test bar chart generation and PNG/SVG export."""
        stem = Path(self.temp_dir) / "test_bar"
        paths = plot_bar_chart(
            categories=["Instruction", "Example", "Hybrid"],
            values=[0.65, 0.78, 0.88],
            title="Test Bar Chart",
            xlabel="Strategy",
            ylabel="Accuracy",
            output_path_stem=stem,
        )

        self.assertTrue(paths[0].exists())  # PNG
        self.assertTrue(paths[1].exists())  # SVG

    def test_plot_line_chart(self):
        """Test line chart generation."""
        stem = Path(self.temp_dir) / "test_line"
        paths = plot_line_chart(
            x_values=["1-shot", "3-shot", "5-shot"],
            series_dict={"GPT-3.5": [0.70, 0.80, 0.85], "Gemini": [0.68, 0.79, 0.84]},
            title="Shot Count Scaling",
            xlabel="Few-Shot Size",
            ylabel="F1 Score",
            output_path_stem=stem,
        )

        self.assertTrue(paths[0].exists())
        self.assertTrue(paths[1].exists())

    def test_visualization_engine_from_metrics(self):
        """Test VisualizationEngine generating figures from metric dicts."""
        engine = VisualizationEngine(output_dir=self.temp_dir)
        perf = {"accuracy": 0.85, "f1_score": 0.84, "bleu_4": 0.45, "rouge_1": 0.60}
        eff = {"latency_mean": 0.12, "latency_median": 0.10, "cost_total_usd": 0.005}
        rel = {"output_consistency": 0.95, "trial_stability_std": 0.02}

        files = engine.generate_all_figures_from_metrics(perf, eff, rel, experiment_id="exp_test")
        self.assertGreater(len(files), 0)
        for f in files:
            self.assertTrue(f.exists())

    def test_visualization_engine_from_dataframe(self):
        """Test VisualizationEngine generating comparative figures from DataFrame."""
        df = pd.DataFrame([
            {"num_examples": 1, "ordering_strategy": "original", "diversity_strategy": "low", "model_name": "gpt-3.5-turbo", "latency_seconds": 0.10, "success": 0.70},
            {"num_examples": 3, "ordering_strategy": "random", "diversity_strategy": "medium", "model_name": "gpt-3.5-turbo", "latency_seconds": 0.15, "success": 0.82},
            {"num_examples": 5, "ordering_strategy": "performance_based", "diversity_strategy": "high", "model_name": "gemini-1.5-flash", "latency_seconds": 0.08, "success": 0.88},
        ])

        engine = VisualizationEngine(output_dir=self.temp_dir)
        files = engine.generate_all_figures_from_dataframe(df)
        self.assertGreater(len(files), 0)
        for f in files:
            self.assertTrue(f.exists())


if __name__ == "__main__":
    unittest.main()
