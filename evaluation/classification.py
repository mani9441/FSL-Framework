"""
Classification Evaluator for FSL Research Framework.
Calculates Accuracy, Precision, Recall, and F1-Score across classification outputs.
"""

import re
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utilities.logger import get_logger

logger = get_logger("classification_evaluator")


def normalize_label(label: str) -> str:
    """Normalizes string label for comparison."""
    if label is None:
        return ""
    text = str(label).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


class ClassificationEvaluator:
    """Evaluator for Text Classification tasks."""

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        references: List[str],
        parse_successes: Optional[List[bool]] = None,
        request_successes: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Computes Accuracy, Precision, Recall, F1 scores, and failure breakdowns over valid executions.
        """
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("Empty or mismatched predictions/references for classification evaluation.")
            return {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "total_samples": 0,
                "parse_success_rate": 0.0,
                "request_success_rate": 0.0,
            }

        total_count = len(predictions)
        req_success_count = sum(1 for s in request_successes if s) if request_successes else total_count
        parse_success_count = sum(1 for s in parse_successes if s) if parse_successes else total_count

        valid_preds, valid_refs = [], []
        if request_successes is not None:
            for p, r, s in zip(predictions, references, request_successes):
                if s:
                    valid_preds.append(normalize_label(p))
                    valid_refs.append(normalize_label(r))
        else:
            valid_preds = [normalize_label(p) for p in predictions]
            valid_refs = [normalize_label(r) for r in references]

        if not valid_preds:
            logger.warning("No valid successful execution records for classification evaluation.")
            return {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "total_samples": total_count,
                "successful_requests": 0,
                "successful_parses": 0,
                "request_success_rate": 0.0,
                "parse_success_rate": 0.0,
            }

        # Overall Accuracy over valid executions
        acc = float(accuracy_score(valid_refs, valid_preds))

        # Precision, Recall, F1 (macro and weighted)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            valid_refs, valid_preds, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            valid_refs, valid_preds, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f1_macro), 4),
            "precision_weighted": round(float(p_weighted), 4),
            "recall_weighted": round(float(r_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "total_samples": total_count,
            "successful_requests": req_success_count,
            "successful_parses": parse_success_count,
            "request_success_rate": round(req_success_count / float(total_count), 4) if total_count > 0 else 0.0,
            "parse_success_rate": round(parse_success_count / float(total_count), 4) if total_count > 0 else 0.0,
        }

        logger.info(f"Classification Evaluation: Acc={acc:.4f}, F1={f1_macro:.4f}, Parse Rate={metrics['parse_success_rate']:.4f}")
        return metrics
