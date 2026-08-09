"""
Statistical Analysis Engine for FSL Research Framework.
Master engine coordinating task-aware descriptive aggregation, factor comparisons, and hypothesis testing.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd

from evaluation.registry import TaskMetricRegistry
from analysis.schema import StatisticalAnalysisResult, StatisticalTestResult
from analysis.aggregator import MetricsAggregator
from analysis.comparator import FactorComparator
from analysis.significance import SignificanceTester
from response_repository import ResponseRecord, ResponseRepository
from utilities.constants import RUNS_DIR
from utilities.helpers import save_json
from utilities.logger import get_logger

logger = get_logger("statistical_analysis_engine")


class StatisticalAnalysisEngine:
    """Master engine orchestrating statistical data processing and reporting."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = RUNS_DIR / "latest" / "statistics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_dataframe(
        self, df: pd.DataFrame, target_metric: Optional[str] = None, task_type: Optional[str] = None
    ) -> StatisticalAnalysisResult:
        """
        Executes statistical analysis on a DataFrame of response records using task-aware target metrics.
        """
        logger.info(f"Starting Statistical Analysis Engine on DataFrame ({len(df)} rows)...")

        # Resolve task_type and target_metric
        inferred_task = task_type
        if not inferred_task and "task_type" in df.columns and not df["task_type"].empty:
            inferred_task = str(df["task_type"].iloc[0])

        if not target_metric or target_metric == "is_correct" and inferred_task == "text_generation":
            if inferred_task:
                target_metric = TaskMetricRegistry.get_statistical_target(inferred_task)
            else:
                target_metric = "is_correct"

        # Fallback if metric column absent in DataFrame
        if target_metric not in df.columns:
            possible_cols = [c for c in ["accuracy", "response_correctness", "rouge_l", "bleu_4", "f1", "latency_seconds"] if c in df.columns]
            if possible_cols:
                target_metric = possible_cols[0]

        logger.info(f"Target statistical metric for analysis: '{target_metric}' (task_type='{inferred_task}')")

        # 1. Descriptive Aggregation
        if target_metric and target_metric in df.columns:
            target_scores = pd.to_numeric(df[target_metric], errors="coerce").dropna().tolist()
            desc_stats = {"overall": MetricsAggregator.compute_descriptive_stats(target_scores)}
        else:
            desc_stats = {}

        # 2. Comparative Factor Analysis
        comparative = FactorComparator.compare_factors(df, metric_col=target_metric)

        # 3. Statistical Significance Testing
        hypothesis_tests: List[Dict[str, Any]] = []

        if target_metric and target_metric in df.columns and "num_examples" in df.columns:
            shot_groups = []
            for _, group in df.groupby("num_examples"):
                vals = pd.to_numeric(group[target_metric], errors="coerce").dropna().tolist()
                if vals:
                    shot_groups.append(vals)

            if len(shot_groups) >= 2:
                # Check for zero variance across all groups
                all_vals = [v for g in shot_groups for v in g]
                if len(set(all_vals)) > 1:
                    try:
                        anova_res = SignificanceTester.one_way_anova(shot_groups)
                        hypothesis_tests.append(anova_res.to_dict())
                    except Exception as e:
                        logger.warning(f"ANOVA hypothesis testing skipped: {e}")

        exp_ids = df["experiment_id"].unique().tolist() if "experiment_id" in df.columns else []

        result = StatisticalAnalysisResult(
            experiment_ids=exp_ids,
            descriptive_stats=desc_stats,
            comparative_analysis=comparative,
            hypothesis_tests=hypothesis_tests,
        )

        self.save_analysis(result, df)
        logger.info(f"Statistical Analysis completed. Outputs persisted to {self.output_dir}")
        return result

    def save_analysis(
        self, result: StatisticalAnalysisResult, df: pd.DataFrame
    ) -> Dict[str, Path]:
        """Saves statistical summary JSON and comparative CSV tables."""
        json_path = self.output_dir / "statistical_summary.json"
        csv_path = self.output_dir / "comparative_analysis.csv"

        save_json(result.to_dict(), json_path)

        rows: List[Dict[str, Any]] = []
        for factor, levels in result.comparative_analysis.items():
            for lvl_info in levels:
                rows.append(lvl_info)

        if rows:
            comp_df = pd.DataFrame(rows)
            comp_df.to_csv(csv_path, index=False, encoding="utf-8")

        return {"json": json_path, "csv": csv_path}

    @classmethod
    def analyze_experiment_folder(
        cls, experiment_id: str, responses_dir: Optional[Union[str, Path]] = None
    ) -> StatisticalAnalysisResult:
        """
        Loads responses from disk for experiment_id and executes statistical analysis.
        """
        if responses_dir:
            resp_dir = Path(responses_dir)
        else:
            matching = list(RUNS_DIR.rglob(f"experiments/{experiment_id}/responses"))
            if matching:
                resp_dir = matching[0]
            else:
                resp_dir = RUNS_DIR / "latest" / "experiments" / experiment_id / "responses"

        records = ResponseRepository.load_all_responses(resp_dir)
        df = ResponseRepository.to_dataframe(records)

        task_type = records[0].task_type if records else None

        engine = cls()
        return engine.analyze_dataframe(df, task_type=task_type)
