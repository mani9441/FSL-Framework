"""
Factor Comparator for FSL Research Framework.
Performs comparative multi-factor analysis across experimental dimensions.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from analysis.aggregator import MetricsAggregator
from utilities.logger import get_logger

logger = get_logger("factor_comparator")


class FactorComparator:
    """Comparator analyzing performance variations across experimental factors."""

    FACTORS = [
        "num_examples",
        "ordering_strategy",
        "diversity_strategy",
        "model_name",
        "task_type",
    ]

    @classmethod
    def compare_factors(
        cls, df: pd.DataFrame, metric_col: str = "is_correct"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups DataFrame by each available factor and computes summary statistics.
        """
        if df.empty or metric_col not in df.columns:
            logger.warning(f"DataFrame is empty or missing metric column '{metric_col}'.")
            return {}

        results: Dict[str, List[Dict[str, Any]]] = {}

        for factor in cls.FACTORS:
            if factor not in df.columns:
                continue

            factor_summary: List[Dict[str, Any]] = []
            grouped = df.groupby(factor)

            for val, group in grouped:
                scores = group[metric_col].dropna().tolist()
                stats = MetricsAggregator.compute_descriptive_stats(scores)
                entry = {
                    "factor": factor,
                    "level": str(val),
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "variance": stats["variance"],
                    "std_dev": stats["std_dev"],
                    "count": int(stats["count"]),
                }
                factor_summary.append(entry)

            results[factor] = factor_summary
            logger.info(f"Compared factor '{factor}' ({len(factor_summary)} levels).")

        return results
