"""
Report Generation Schema and Data Models for FSL Research Framework.
Defines ResearchReportPayload dataclass capturing research report artifacts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List


@dataclass
class ResearchReportPayload:
    """Dataclass encapsulating all generated research artifacts and report sections."""
    experiment_id: str
    experiment_summary: str
    performance_report: Dict[str, Any] = field(default_factory=dict)
    evaluation_tables_md: str = ""
    evaluation_tables_tex: str = ""
    statistical_report: Dict[str, Any] = field(default_factory=dict)
    comparison_report: Dict[str, Any] = field(default_factory=dict)
    research_observations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Converts ResearchReportPayload to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_summary": self.experiment_summary,
            "performance_report": self.performance_report,
            "evaluation_tables_md": self.evaluation_tables_md,
            "evaluation_tables_tex": self.evaluation_tables_tex,
            "statistical_report": self.statistical_report,
            "comparison_report": self.comparison_report,
            "research_observations": self.research_observations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchReportPayload":
        """Instantiates ResearchReportPayload from dictionary."""
        return cls(
            experiment_id=data.get("experiment_id", ""),
            experiment_summary=data.get("experiment_summary", ""),
            performance_report=data.get("performance_report", {}),
            evaluation_tables_md=data.get("evaluation_tables_md", ""),
            evaluation_tables_tex=data.get("evaluation_tables_tex", ""),
            statistical_report=data.get("statistical_report", {}),
            comparison_report=data.get("comparison_report", {}),
            research_observations=data.get("research_observations", []),
            timestamp=data.get("timestamp", ""),
        )
