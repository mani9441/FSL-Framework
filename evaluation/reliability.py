"""
Reliability Evaluator for FSL Research Framework.
Calculates Output Consistency, Response Variance, and Multi-Trial Stability.
"""

import numpy as np
from typing import List, Dict, Any
from response_repository.schema import ResponseRecord
from utilities.logger import get_logger

logger = get_logger("reliability_evaluator")


class ReliabilityEvaluator:
    """Evaluator assessing output consistency, length variance, and trial stability."""

    @classmethod
    def evaluate(
        cls,
        trial_records_map: Dict[int, List[ResponseRecord]],
        trial_performance_scores: List[float],
    ) -> Dict[str, float]:
        """
        Computes Output Consistency, Response Variance, and Trial Stability.
        `trial_records_map`: dict mapping trial_index -> list of ResponseRecord objects.
        `trial_performance_scores`: list of performance metric values per trial (e.g. F1 or accuracy scores).
        """
        if not trial_records_map:
            return {
                "output_consistency": 1.0,
                "response_length_variance": 0.0,
                "trial_stability_std": 0.0,
            }

        # 1. Output Consistency & Length Variance across trials
        # Align records by record_id
        record_predictions: Dict[str, List[str]] = {}
        record_lengths: Dict[str, List[int]] = {}

        for trial_idx, records in trial_records_map.items():
            for r in records:
                record_predictions.setdefault(r.record_id, []).append(r.generated_text.strip().lower())
                record_lengths.setdefault(r.record_id, []).append(len(r.generated_text))

        consistency_rates: List[float] = []
        length_variances: List[float] = []

        for rec_id, preds in record_predictions.items():
            if len(preds) > 1:
                # Consistency = frequency of most common prediction / total trials for this record
                most_common_freq = max(preds.count(p) for p in set(preds))
                consistency_rates.append(most_common_freq / float(len(preds)))
            else:
                consistency_rates.append(1.0)

        for rec_id, lengths in record_lengths.items():
            if len(lengths) > 1:
                length_variances.append(float(np.var(lengths)))
            else:
                length_variances.append(0.0)

        mean_consistency = float(np.mean(consistency_rates)) if consistency_rates else 1.0
        mean_length_var = float(np.mean(length_variances)) if length_variances else 0.0

        # 2. Multi-Trial Performance Stability (Standard Deviation of scores)
        if trial_performance_scores and len(trial_performance_scores) > 1:
            stability_std = float(np.std(trial_performance_scores))
        else:
            stability_std = 0.0

        metrics = {
            "output_consistency": round(mean_consistency, 4),
            "response_length_variance": round(mean_length_var, 4),
            "trial_stability_std": round(stability_std, 4),
        }

        logger.info(
            f"Reliability Evaluation: Consistency={mean_consistency:.4f}, "
            f"Length Var={mean_length_var:.4f}, Stability Std={stability_std:.4f}"
        )
        return metrics
