"""
Experiment Schema and Data Models for FSL Research Framework.
Defines ExperimentTrialResult, ProgressState, ExperimentStatus, FailureRecord, and ExperimentManifest dataclasses.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


class ExperimentStatus(str, Enum):
    """Execution status enum for experiments."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"


@dataclass
class FailureRecord:
    """Dataclass capturing failure details for debugging and auditability."""
    time: str = field(default_factory=lambda: datetime.now().isoformat())
    stage: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""
    retry_count: int = 0
    final_status: str = "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
            "retry_count": self.retry_count,
            "final_status": self.final_status,
        }


@dataclass
class ExperimentManifest:
    """Dataclass capturing self-contained experiment status and stage summary."""
    experiment_id: str
    status: str = "PENDING"
    stage: str = "INITIALIZATION"
    reason: str = ""
    completed_trials: int = 0
    total_trials: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    stage_summary: Dict[str, str] = field(default_factory=lambda: {
        "inference": "PENDING",
        "responses": "PENDING",
        "metrics": "PENDING",
        "statistics": "PENDING",
        "figures": "PENDING",
        "reports": "PENDING",
        "dissertation": "PENDING",
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "completed_trials": self.completed_trials,
            "total_trials": self.total_trials,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stage_summary": self.stage_summary,
        }


@dataclass
class ExperimentTrialResult:
    """Dataclass capturing the output and metadata of a single experiment trial sample."""
    experiment_id: str
    trial_index: int
    record_id: str
    prompt_object: Dict[str, Any]
    model_response: Dict[str, Any]
    ground_truth: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Converts ExperimentTrialResult to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "trial_index": self.trial_index,
            "record_id": self.record_id,
            "prompt_object": self.prompt_object,
            "model_response": self.model_response,
            "ground_truth": self.ground_truth,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentTrialResult":
        """Instantiates ExperimentTrialResult from dictionary."""
        return cls(
            experiment_id=data.get("experiment_id", ""),
            trial_index=data.get("trial_index", 0),
            record_id=data.get("record_id", ""),
            prompt_object=data.get("prompt_object", {}),
            model_response=data.get("model_response", {}),
            ground_truth=data.get("ground_truth", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class ProgressState:
    """Dataclass tracking real-time experiment execution progress."""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0
    status: str = "idle"  # idle, running, completed, failed
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts ProgressState to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "running": self.running,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
        }
