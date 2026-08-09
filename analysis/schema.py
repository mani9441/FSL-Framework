"""
Statistical Analysis Schema and Data Models for FSL Research Framework.
Defines StatisticalTestResult and StatisticalAnalysisResult dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class StatisticalTestResult:
    """Dataclass storing the output of a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size_cohens_d: float = 0.0
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts StatisticalTestResult to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "effect_size_cohens_d": round(self.effect_size_cohens_d, 4),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalTestResult":
        """Instantiates StatisticalTestResult from dictionary."""
        return cls(
            test_name=data.get("test_name", ""),
            statistic=data.get("statistic", 0.0),
            p_value=data.get("p_value", 1.0),
            is_significant=data.get("is_significant", False),
            effect_size_cohens_d=data.get("effect_size_cohens_d", 0.0),
            interpretation=data.get("interpretation", ""),
        )


@dataclass
class StatisticalAnalysisResult:
    """Dataclass encapsulating aggregated descriptive statistics, factor comparisons, and hypothesis tests."""
    experiment_ids: List[str] = field(default_factory=list)
    descriptive_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    comparative_analysis: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    hypothesis_tests: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Converts StatisticalAnalysisResult to dictionary."""
        return {
            "experiment_ids": self.experiment_ids,
            "descriptive_stats": self.descriptive_stats,
            "comparative_analysis": self.comparative_analysis,
            "hypothesis_tests": self.hypothesis_tests,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalAnalysisResult":
        """Instantiates StatisticalAnalysisResult from dictionary."""
        return cls(
            experiment_ids=data.get("experiment_ids", []),
            descriptive_stats=data.get("descriptive_stats", {}),
            comparative_analysis=data.get("comparative_analysis", {}),
            hypothesis_tests=data.get("hypothesis_tests", []),
            timestamp=data.get("timestamp", ""),
        )
