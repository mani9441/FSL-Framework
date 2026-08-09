"""
Unit tests for Module 11 — Report Generation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from reports.schema import ResearchReportPayload
from reports.latex_generator import LaTeXTableGenerator
from reports.markdown_generator import MarkdownReportGenerator
from reports.exporter import ReportExporter
from reports.engine import ReportGenerationEngine
from evaluation.schema import EvaluationMetricsResult
from analysis.schema import StatisticalAnalysisResult, StatisticalTestResult


class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_df = pd.DataFrame([
            {"experiment_id": "EXP_01", "model_name": "gpt-3.5-turbo", "success": 0.85, "latency_seconds": 0.12},
            {"experiment_id": "EXP_01", "model_name": "gemini-1.5-flash", "success": 0.90, "latency_seconds": 0.08},
        ])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_latex_table_generator(self):
        """Test LaTeXTableGenerator generating booktabs LaTeX syntax."""
        tex = LaTeXTableGenerator.generate_latex_table(
            df=self.sample_df,
            caption="Test Model Results",
            label="tab:test_models",
        )
        self.assertIn("\\begin{table}", tex)
        self.assertIn("\\caption{Test Model Results}", tex)
        self.assertIn("\\toprule", tex)
        self.assertIn("\\bottomrule", tex)

    def test_markdown_report_generator(self):
        """Test MarkdownReportGenerator generating report text."""
        payload = ResearchReportPayload(
            experiment_id="EXP_TEST_MD",
            experiment_summary="Test Executive Summary",
            performance_report={"accuracy": 0.88, "f1_score": 0.86},
            research_observations=["Observation 1", "Observation 2"],
        )

        md_text = MarkdownReportGenerator.generate_markdown(payload)
        self.assertIn("# Comprehensive Research Report", md_text)
        self.assertIn("EXP_TEST_MD", md_text)
        self.assertIn("Observation 1", md_text)

    def test_report_exporter(self):
        """Test ReportExporter creating .md, .html, .tex, .json, .csv files."""
        payload = ResearchReportPayload(
            experiment_id="EXP_EXPORT_TEST",
            experiment_summary="Export Test Summary",
            performance_report={"accuracy": 0.92},
            evaluation_tables_tex="% Sample TeX",
        )

        paths = ReportExporter.export_all(payload, self.temp_dir)
        self.assertTrue(paths["md"].exists())
        self.assertTrue(paths["html"].exists())
        self.assertTrue(paths["tex"].exists())
        self.assertTrue(paths["json"].exists())
        self.assertTrue(paths["csv"].exists())

    def test_report_generation_engine(self):
        """Test ReportGenerationEngine generating artifacts from evaluation and analysis results."""
        eval_res = EvaluationMetricsResult(
            experiment_id="EXP_ENGINE_TEST",
            task_type="text_classification",
            performance_metrics={"accuracy": 0.95, "f1_score": 0.94},
            efficiency_metrics={"latency_mean": 0.10, "cost_total_usd": 0.002},
            reliability_metrics={"output_consistency": 0.98},
        )
        test_obj = StatisticalTestResult("One-Way ANOVA", 12.5, 0.001, True, 1.2, "Significant difference.")
        analysis_res = StatisticalAnalysisResult(
            experiment_ids=["EXP_ENGINE_TEST"],
            descriptive_stats={"overall": {"mean": 0.95, "count": 20.0}},
            hypothesis_tests=[test_obj.to_dict()],
        )

        engine = ReportGenerationEngine("EXP_ENGINE_TEST", output_dir=self.temp_dir)
        exported = engine.generate_report(eval_res, analysis_res, self.sample_df)

        self.assertTrue(exported["md"].exists())
        self.assertTrue(exported["html"].exists())
        self.assertTrue(exported["tex"].exists())


if __name__ == "__main__":
    unittest.main()
