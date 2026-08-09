"""
Metrics Aggregator for FSL Research Framework.
Calculates Mean, Median, Variance, and Standard Deviation across experimental metrics.
"""

import numpy as np
from typing import List, Dict, Any, Union
from utilities.logger import get_logger

logger = get_logger("metrics_aggregator")


class MetricsAggregator:
    """Aggregator calculating descriptive statistical metrics."""

    @classmethod
    def compute_descriptive_stats(cls, values: List[Union[int, float]]) -> Dict[str, float]:
        """
        Computes Mean, Median, Variance, Standard Deviation, Min, Max, and Count.
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0.0,
            }

        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        var_val = float(np.var(arr))
        std_val = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        count_val = float(len(arr))

        stats = {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "variance": round(var_val, 4),
            "std_dev": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "count": count_val,
        }

        logger.debug(f"Descriptive Stats (N={len(values)}): Mean={mean_val:.4f}, Std={std_val:.4f}")
        return stats
