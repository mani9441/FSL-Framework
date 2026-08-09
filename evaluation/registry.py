"""
Task Metric Registry for FSL Research Framework.
Defines explicit metric mappings, primary evaluation metrics, and statistical target metrics for each NLP task type.
"""

from typing import Dict, List, Any, Optional


class TaskMetricRegistry:
    """Registry specifying metric applicability and statistical targets per task type."""

    TASK_METRICS: Dict[str, Dict[str, Any]] = {
        "text_classification": {
            "task_name": "Text Classification",
            "primary_metrics": ["accuracy", "f1_score"],
            "performance_metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ],
            "sample_fields": [
                "parsed_prediction",
                "is_correct",
            ],
            "primary_statistical_metric": "accuracy",
            "has_binary_correctness": True,
        },
        "question_answering": {
            "task_name": "Question Answering",
            "primary_metrics": [
                "response_correctness",
                "response_completeness",
                "response_relevance",
            ],
            "performance_metrics": [
                "response_correctness",
                "response_completeness",
                "response_relevance",
                "qa_exact_match",
                "qa_f1",
            ],
            "sample_fields": [
                "parsed_prediction",
                "response_correctness",
                "response_completeness",
                "response_relevance",
                "qa_exact_match",
                "qa_f1",
            ],
            "primary_statistical_metric": "response_correctness",
            "has_binary_correctness": False,
        },
        "text_generation": {
            "task_name": "Text Generation",
            "primary_metrics": ["bleu_4", "rouge_1", "rouge_l"],
            "performance_metrics": [
                "bleu_1",
                "bleu_2",
                "bleu_4",
                "rouge_1",
                "rouge_2",
                "rouge_l",
            ],
            "sample_fields": [
                "bleu_4",
                "rouge_1",
                "rouge_l",
            ],
            "primary_statistical_metric": "rouge_l",
            "has_binary_correctness": False,
        },
    }

    @classmethod
    def get_task_config(cls, task_type: str) -> Dict[str, Any]:
        """Returns metric configuration dictionary for a given task_type."""
        norm_task = task_type.lower().strip() if task_type else "text_classification"
        if norm_task not in cls.TASK_METRICS:
            return cls.TASK_METRICS["text_classification"]
        return cls.TASK_METRICS[norm_task]

    @classmethod
    def get_primary_metrics(cls, task_type: str) -> List[str]:
        """Returns list of primary metric names for task_type."""
        return cls.get_task_config(task_type)["primary_metrics"]

    @classmethod
    def get_performance_metrics(cls, task_type: str) -> List[str]:
        """Returns full list of valid performance metric names for task_type."""
        return cls.get_task_config(task_type)["performance_metrics"]

    @classmethod
    def get_sample_fields(cls, task_type: str) -> List[str]:
        """Returns list of sample-level evaluation field names for task_type."""
        return cls.get_task_config(task_type)["sample_fields"]

    @classmethod
    def get_statistical_target(cls, task_type: str) -> str:
        """Returns target metric column for statistical significance testing."""
        return cls.get_task_config(task_type)["primary_statistical_metric"]

    @classmethod
    def is_metric_applicable(cls, task_type: str, metric_name: str) -> bool:
        """Checks if a metric name is applicable to the given task_type."""
        valid_metrics = cls.get_performance_metrics(task_type) + [
            "request_success_rate",
            "parse_success_rate",
            "latency_seconds",
        ]
        return metric_name.lower().strip() in [m.lower() for m in valid_metrics]
