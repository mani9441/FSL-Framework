"""
Evaluation Framework Package for FSL Research Framework.
Provides classification, QA, generation, efficiency, and reliability evaluators.
"""

from evaluation.parser import PredictionParser
from evaluation.schema import EvaluationMetricsResult
from evaluation.registry import TaskMetricRegistry
from evaluation.classification import ClassificationEvaluator
from evaluation.qa_eval import QAEvaluator
from evaluation.generation import GenerationEvaluator
from evaluation.efficiency import EfficiencyEvaluator
from evaluation.reliability import ReliabilityEvaluator
from evaluation.engine import EvaluationEngine

__all__ = [
    "PredictionParser",
    "EvaluationMetricsResult",
    "TaskMetricRegistry",
    "ClassificationEvaluator",
    "QAEvaluator",
    "GenerationEvaluator",
    "EfficiencyEvaluator",
    "ReliabilityEvaluator",
    "EvaluationEngine",
]

