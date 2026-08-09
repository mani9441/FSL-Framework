"""
Evaluation Schema and Data Models for FSL Research Framework.
Defines EvaluationMetricsResult dataclass capturing performance, efficiency, and reliability metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List


@dataclass
class EvaluationMetricsResult:
    """Dataclass capturing evaluation output across all metric categories."""
    experiment_id: str
    task_type: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    reliability_metrics: Dict[str, float] = field(default_factory=dict)
    sample_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Converts EvaluationMetricsResult to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "task_type": self.task_type,
            "performance_metrics": self.performance_metrics,
            "efficiency_metrics": self.efficiency_metrics,
            "reliability_metrics": self.reliability_metrics,
            "sample_evaluations": self.sample_evaluations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationMetricsResult":
        """Instantiates EvaluationMetricsResult from dictionary."""
        return cls(
            experiment_id=data.get("experiment_id", ""),
            task_type=data.get("task_type", ""),
            performance_metrics=data.get("performance_metrics", {}),
            efficiency_metrics=data.get("efficiency_metrics", {}),
            reliability_metrics=data.get("reliability_metrics", {}),
            sample_evaluations=data.get("sample_evaluations", []),
            timestamp=data.get("timestamp", ""),
        )
