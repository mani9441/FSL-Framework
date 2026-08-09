"""
Visualization Engine for FSL Research Framework.
Master engine coordinating task-aware publication figure generation and file exports.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd

from evaluation.registry import TaskMetricRegistry
from visualization.charts import (
    plot_bar_chart,
    plot_line_chart,
    plot_scatter_chart,
    plot_heatmap,
    plot_reliability_chart,
)
from response_repository import ResponseRecord, ResponseRepository
from utilities.constants import RUNS_DIR
from utilities.logger import get_logger

logger = get_logger("visualization_engine")


class VisualizationEngine:
    """Master visualization engine building task-aware publication-quality figures."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = RUNS_DIR / "latest" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_figures_from_metrics(
        self,
        perf_metrics: Dict[str, float],
        eff_metrics: Dict[str, float],
        rel_metrics: Dict[str, float],
        experiment_id: str = "exp_run",
        task_type: Optional[str] = None,
    ) -> List[Path]:
        """
        Generates task-specific figures (Classification, QA, or Generation) from metrics dicts.
        """
        logger.info(f"Generating publication figures from metrics for '{experiment_id}' ({task_type})...")
        generated_files: List[Path] = []

        norm_task = task_type.lower().strip() if task_type else ""

        # 1. Classification Performance Chart
        if norm_task == "text_classification" or ("accuracy" in perf_metrics and norm_task != "text_generation"):
            cls_keys = [k for k in ["accuracy", "f1_score", "precision", "recall"] if k in perf_metrics]
            if cls_keys:
                cats = [k.replace("_", " ").title() for k in cls_keys]
                vals = [perf_metrics[k] for k in cls_keys]
                files = plot_bar_chart(
                    categories=cats,
                    values=vals,
                    title=f"Classification Performance Metrics ({experiment_id})",
                    xlabel="Metric Name",
                    ylabel="Score",
                    output_path_stem=self.output_dir / f"{experiment_id}_accuracy_chart",
                )
                generated_files.extend(files)

        # 2. Question Answering Performance Chart
        elif norm_task == "question_answering" or "response_correctness" in perf_metrics:
            qa_keys = [k for k in ["response_correctness", "response_completeness", "response_relevance", "qa_exact_match", "qa_f1"] if k in perf_metrics]
            if qa_keys:
                cats = [k.replace("response_", "").replace("qa_", "").replace("_", " ").title() for k in qa_keys]
                vals = [perf_metrics[k] for k in qa_keys]
                files = plot_bar_chart(
                    categories=cats,
                    values=vals,
                    title=f"QA Evaluation Metrics ({experiment_id})",
                    xlabel="QA Metric Name",
                    ylabel="Score",
                    output_path_stem=self.output_dir / f"{experiment_id}_qa_chart",
                )
                generated_files.extend(files)

        # 3. BLEU & ROUGE Generation Charts
        if norm_task == "text_generation" or any("bleu" in k or "rouge" in k for k in perf_metrics):
            bleu_keys = [k for k in perf_metrics.keys() if "bleu" in k]
            if bleu_keys:
                vals = [perf_metrics[k] for k in bleu_keys]
                files = plot_bar_chart(
                    categories=[k.upper() for k in bleu_keys],
                    values=vals,
                    title=f"BLEU N-Gram Precision Scores ({experiment_id})",
                    xlabel="BLEU Metric",
                    ylabel="Score",
                    output_path_stem=self.output_dir / f"{experiment_id}_bleu_chart",
                )
                generated_files.extend(files)

            rouge_keys = [k for k in perf_metrics.keys() if "rouge" in k]
            if rouge_keys:
                vals = [perf_metrics[k] for k in rouge_keys]
                files = plot_bar_chart(
                    categories=[k.upper() for k in rouge_keys],
                    values=vals,
                    title=f"ROUGE Summarization Overlap Scores ({experiment_id})",
                    xlabel="ROUGE Metric",
                    ylabel="Score",
                    output_path_stem=self.output_dir / f"{experiment_id}_rouge_chart",
                )
                generated_files.extend(files)

        # 4. Latency & Efficiency Chart
        if "latency_mean" in eff_metrics:
            lat_cats = ["Mean", "Median", "Min", "Max"]
            lat_vals = [eff_metrics.get("latency_mean", 0), eff_metrics.get("latency_median", 0), eff_metrics.get("latency_min", 0), eff_metrics.get("latency_max", 0)]
            files = plot_bar_chart(
                categories=lat_cats,
                values=lat_vals,
                title=f"Inference Latency Profile ({experiment_id})",
                xlabel="Latency Metric",
                ylabel="Time (seconds)",
                output_path_stem=self.output_dir / f"{experiment_id}_latency_chart",
            )
            generated_files.extend(files)

        # 5. Reliability & Trial Stability Chart
        if "output_consistency" in rel_metrics:
            rel_cats = ["Consistency", "Trial Stability"]
            rel_vals = [rel_metrics.get("output_consistency", 1.0), 1.0 - rel_metrics.get("trial_stability_std", 0.0)]
            rel_errs = [0.0, rel_metrics.get("trial_stability_std", 0.0)]
            files = plot_reliability_chart(
                categories=rel_cats,
                values=rel_vals,
                errors=rel_errs,
                title=f"Reliability & Output Consistency ({experiment_id})",
                xlabel="Dimension",
                ylabel="Score / Stability",
                output_path_stem=self.output_dir / f"{experiment_id}_reliability_chart",
            )
            generated_files.extend(files)

        logger.info(f"Generated {len(generated_files)} figure files in {self.output_dir}")
        return generated_files

    def generate_all_figures_from_dataframe(self, df: pd.DataFrame) -> List[Path]:
        """
        Generates comparative multi-factor figures from response record DataFrame.
        """
        logger.info(f"Generating comparative figures from DataFrame ({len(df)} rows)...")
        generated_files: List[Path] = []

        task_type = df["task_type"].iloc[0] if "task_type" in df.columns and not df.empty else None

        if task_type == "text_generation":
            perf_col = "rouge_l" if "rouge_l" in df.columns else ("bleu_4" if "bleu_4" in df.columns else "latency_seconds")
            ylabel_str = "ROUGE-L Score"
        elif task_type == "question_answering":
            perf_col = "response_correctness" if "response_correctness" in df.columns else ("qa_f1" if "qa_f1" in df.columns else "latency_seconds")
            ylabel_str = "Response Correctness"
        else:
            perf_col = "is_correct" if "is_correct" in df.columns else ("accuracy" if "accuracy" in df.columns else "success")
            ylabel_str = "Mean Task Accuracy"

        # 1. Few-Shot Scaling Trajectories (1-shot -> 3-shot -> 5-shot)
        if "num_examples" in df.columns and perf_col in df.columns:
            shot_grouped = df.groupby("num_examples")[perf_col].mean()
            shots = [str(x) + "-shot" for x in shot_grouped.index]
            vals = shot_grouped.values.tolist()

            files = plot_line_chart(
                x_values=shots,
                series_dict={ylabel_str: vals},
                title="Performance Scaling Across Few-Shot Example Counts",
                xlabel="Demonstration Count (K-Shot)",
                ylabel=ylabel_str,
                output_path_stem=self.output_dir / "shot_count_scaling",
            )
            generated_files.extend(files)

        # 2. Latency vs Performance Scatter Plot
        if "latency_seconds" in df.columns and perf_col in df.columns and "model_name" in df.columns:
            if perf_col == "latency_seconds":
                grouped_series = df.groupby("model_name")["latency_seconds"].mean()
                x_vals = grouped_series.tolist()
                y_vals = grouped_series.tolist()
                labels_list = grouped_series.index.tolist()
            else:
                grouped_df = df.groupby("model_name")[["latency_seconds", perf_col]].mean()
                x_vals = grouped_df["latency_seconds"].tolist()
                y_vals = grouped_df[perf_col].tolist()
                labels_list = grouped_df.index.tolist()

            files = plot_scatter_chart(
                x_values=x_vals,
                y_values=y_vals,
                labels=labels_list,
                title=f"Efficiency Trade-Off: Inference Latency vs {ylabel_str}",
                xlabel="Mean Latency (seconds)",
                ylabel=ylabel_str,
                output_path_stem=self.output_dir / "latency_vs_accuracy_scatter",
            )
            generated_files.extend(files)

        # 3. Ordering vs Diversity Heatmap
        if "ordering_strategy" in df.columns and "diversity_strategy" in df.columns and perf_col in df.columns:
            pivot = df.pivot_table(index="ordering_strategy", columns="diversity_strategy", values=perf_col, aggfunc="mean").fillna(0.0)
            files = plot_heatmap(
                matrix_df=pivot,
                title="Parameter Matrix: Demonstration Ordering vs Diversity",
                xlabel="Diversity Strategy",
                ylabel="Ordering Strategy",
                output_path_stem=self.output_dir / "ordering_vs_diversity_heatmap",
            )
            generated_files.extend(files)

        logger.info(f"Generated {len(generated_files)} comparative figure files.")
        return generated_files
