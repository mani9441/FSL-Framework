"""
Experiment Engine Package for FSL Research Framework.
Provides TrialManager, ProgressTracker, ExperimentController, ExperimentMatrixGenerator, and Schema models.
"""

from experiment_engine.schema import (
    ExperimentTrialResult,
    ProgressState,
    ExperimentStatus,
    FailureRecord,
    ExperimentManifest,
)
from experiment_engine.progress_tracker import ProgressTracker
from experiment_engine.trial_manager import TrialManager
from experiment_engine.controller import ExperimentController
from experiment_engine.matrix_generator import ExperimentMatrixGenerator

__all__ = [
    "ExperimentTrialResult",
    "ProgressState",
    "ExperimentStatus",
    "FailureRecord",
    "ExperimentManifest",
    "ProgressTracker",
    "TrialManager",
    "ExperimentController",
    "ExperimentMatrixGenerator",
]
