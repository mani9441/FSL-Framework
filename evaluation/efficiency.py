"""
Efficiency Evaluator for FSL Research Framework.
Calculates Latency (mean, median, std), Token Usage, and Financial Computational Cost.
"""

import numpy as np
from typing import List, Dict, Any
from response_repository.schema import ResponseRecord
from utilities.logger import get_logger

logger = get_logger("efficiency_evaluator")


class EfficiencyEvaluator:
    """Evaluator analyzing latency, token consumption, and financial costs."""

    @classmethod
    def evaluate(cls, records: List[ResponseRecord]) -> Dict[str, float]:
        """
        Computes efficiency metrics across response records.
        """
        if not records:
            return {
                "latency_mean": 0.0,
                "latency_median": 0.0,
                "latency_min": 0.0,
                "latency_max": 0.0,
                "latency_std": 0.0,
                "prompt_tokens_mean": 0.0,
                "completion_tokens_mean": 0.0,
                "total_tokens_mean": 0.0,
                "total_tokens_sum": 0.0,
                "cost_total_usd": 0.0,
                "cost_mean_usd": 0.0,
            }

        latencies = [r.latency_seconds for r in records]
        p_tokens = [r.prompt_tokens for r in records]
        c_tokens = [r.completion_tokens for r in records]
        t_tokens = [r.total_tokens for r in records]
        costs = [r.estimated_cost_usd for r in records]

        metrics = {
            "latency_mean": round(float(np.mean(latencies)), 4),
            "latency_median": round(float(np.median(latencies)), 4),
            "latency_min": round(float(np.min(latencies)), 4),
            "latency_max": round(float(np.max(latencies)), 4),
            "latency_std": round(float(np.std(latencies)), 4),
            "prompt_tokens_mean": round(float(np.mean(p_tokens)), 2),
            "completion_tokens_mean": round(float(np.mean(c_tokens)), 2),
            "total_tokens_mean": round(float(np.mean(t_tokens)), 2),
            "total_tokens_sum": int(np.sum(t_tokens)),
            "cost_total_usd": round(float(np.sum(costs)), 6),
            "cost_mean_usd": round(float(np.mean(costs)), 6),
        }

        logger.info(f"Efficiency Evaluation: Mean Latency={metrics['latency_mean']}s, Cost=${metrics['cost_total_usd']:.6f}")
        return metrics
