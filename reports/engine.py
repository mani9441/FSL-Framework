"""
Report Generation Engine for FSL Research Framework.
Master engine synthesizing task-aware evaluation, statistical analysis, and visualization into final research artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd

from evaluation.registry import TaskMetricRegistry
from reports.schema import ResearchReportPayload
from reports.latex_generator import LaTeXTableGenerator
from reports.exporter import ReportExporter
from evaluation import EvaluationEngine, EvaluationMetricsResult
from analysis import StatisticalAnalysisEngine, StatisticalAnalysisResult
from response_repository import ResponseRepository
from utilities.constants import RUNS_DIR
from utilities.logger import get_logger

logger = get_logger("report_generation_engine")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Formats DataFrame into clean GitHub-flavored markdown table without external tabulate dependency."""
    if df.empty:
        return "_No records available._\n"
    cols = df.columns.tolist()
    lines = [
        "| " + " | ".join([str(c).replace("_", " ").title() for c in cols]) + " |",
        "| " + " | ".join([":---" for _ in cols]) + " |",
    ]
    for _, row in df.iterrows():
        formatted_vals = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                formatted_vals.append(f"`{val:.4f}`")
            else:
                formatted_vals.append(f"`{val}`")
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines) + "\n"


class ReportGenerationEngine:
    """Master engine orchestrating multi-format task-aware research report generation."""

    def __init__(self, experiment_id: str, output_dir: Optional[Union[str, Path]] = None):
        self.experiment_id = experiment_id
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = RUNS_DIR / "latest" / "experiments" / experiment_id / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        eval_result: EvaluationMetricsResult,
        analysis_result: StatisticalAnalysisResult,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Path]:
        """
        Synthesizes task-specific evaluation metrics and statistical analysis into research report artifacts.
        """
        logger.info(f"Synthesizing research report artifacts for Experiment '{self.experiment_id}'...")

        task_type = eval_result.task_type
        task_str = eval_result.task_type.replace("_", " ").title()

        # Build task-aware metric description
        if task_type == "text_generation":
            main_metric_name = "ROUGE-L"
            main_val = eval_result.performance_metrics.get("rouge_l", eval_result.performance_metrics.get("bleu_4", 0.0))
        elif task_type == "question_answering":
            main_metric_name = "Response Correctness"
            main_val = eval_result.performance_metrics.get("response_correctness", eval_result.performance_metrics.get("qa_f1", 0.0))
        else:
            main_metric_name = "Accuracy"
            main_val = eval_result.performance_metrics.get("accuracy", eval_result.performance_metrics.get("f1_score", 0.0))

        # 1. Build Executive Summary Text
        count_val = analysis_result.descriptive_stats.get('overall', {}).get('count', 0)
        exec_summary = (
            f"This experiment evaluated Few-Shot Learning (FSL) performance on the **{task_str}** task "
            f"for Experiment ID `{self.experiment_id}`. A total of {count_val} "
            f"responses were analyzed across repeated experimental trials. "
            f"Overall performance reached a mean {main_metric_name} of **{main_val:.4f}** "
            f"with an inference latency mean of **{eval_result.efficiency_metrics.get('latency_mean', 0.0):.4f}s**."
        )

        # 2. Build Markdown & LaTeX Tables
        if df is not None and not df.empty:
            summary_cols = [c for c in ["experiment_id", "prompt_strategy", "num_examples", "model_name", "total_tokens", "latency_seconds", "success"] if c in df.columns]
            sample_table_df = df[summary_cols].head(10)
            latex_table = LaTeXTableGenerator.generate_latex_table(
                df=sample_table_df,
                caption=f"Evaluation Output Sample ({self.experiment_id})",
                label=f"tab:{self.experiment_id}_samples",
            )
            md_table = dataframe_to_markdown(sample_table_df)
        else:
            latex_table = "% No sample dataframe provided.\n"
            md_table = "_No sample records available._\n"

        # 3. Generate Research Observations
        observations: List[str] = []

        if task_type == "text_generation":
            r_l = eval_result.performance_metrics.get("rouge_l", 0.0)
            b_4 = eval_result.performance_metrics.get("bleu_4", 0.0)
            observations.append(f"Generation metrics recorded ROUGE-L={r_l:.4f} and BLEU-4={b_4:.4f}.")
        elif task_type == "question_answering":
            corr = eval_result.performance_metrics.get("response_correctness", 0.0)
            comp = eval_result.performance_metrics.get("response_completeness", 0.0)
            rel = eval_result.performance_metrics.get("response_relevance", 0.0)
            observations.append(f"QA evaluation recorded Correctness={corr:.4f}, Completeness={comp:.4f}, Relevance={rel:.4f}.")
        else:
            acc = eval_result.performance_metrics.get("accuracy", 0.0)
            f1 = eval_result.performance_metrics.get("f1_score", 0.0)
            observations.append(f"Classification recorded Accuracy={acc:.4f} and F1-Score={f1:.4f}.")

        lat = eval_result.efficiency_metrics.get("latency_mean", 0.0)
        observations.append(f"Average inference response latency was measured at {lat:.4f} seconds per query.")

        sig_tests = [t for t in analysis_result.hypothesis_tests if t.get("is_significant")]
        if sig_tests:
            observations.append(f"Identified {len(sig_tests)} statistically significant parameter variations (p < 0.05).")
        else:
            observations.append("No statistically significant metric divergence detected across evaluated parameter levels.")

        cons = eval_result.reliability_metrics.get("output_consistency", 1.0)
        observations.append(f"Multi-trial response output consistency measured at {cons:.2%}.")

        # 4. Construct Payload
        payload = ResearchReportPayload(
            experiment_id=self.experiment_id,
            experiment_summary=exec_summary,
            performance_report=eval_result.performance_metrics,
            evaluation_tables_md=md_table,
            evaluation_tables_tex=latex_table,
            statistical_report={
                "efficiency": eval_result.efficiency_metrics,
                "reliability": eval_result.reliability_metrics,
                "hypothesis_tests": analysis_result.hypothesis_tests,
            },
            comparison_report=analysis_result.comparative_analysis,
            research_observations=observations,
        )

        # 5. Export All Formats
        exported_paths = ReportExporter.export_all(payload, self.output_dir)
        logger.info(f"Report Generation Engine completed for '{self.experiment_id}'. Formats saved: {list(exported_paths.keys())}")
        return exported_paths
