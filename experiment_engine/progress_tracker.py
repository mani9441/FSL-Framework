"""
Progress Tracker for FSL Research Framework.
Monitors experiment execution state, running/completed/failed task metrics, and progress logs.
"""

import time
from pathlib import Path
from typing import Union, Optional
from experiment_engine.schema import ProgressState
from utilities.helpers import save_json
from utilities.logger import get_logger

logger = get_logger("progress_tracker")


class ProgressTracker:
    """Tracker recording real-time experiment progress metrics."""

    def __init__(self, total_tasks: int):
        self.state = ProgressState(total_tasks=total_tasks, status="running")
        self.start_time = time.perf_counter()

    def start_task(self) -> None:
        """Increments running task count."""
        self.state.running += 1

    def complete_task(self) -> None:
        """Increments completed task count."""
        if self.state.running > 0:
            self.state.running -= 1
        self.state.completed += 1
        self._update_status()

    def fail_task(self) -> None:
        """Increments failed task count."""
        if self.state.running > 0:
            self.state.running -= 1
        self.state.failed += 1
        self._update_status()

    def _update_status(self) -> None:
        """Updates internal status state and elapsed seconds."""
        self.state.elapsed_seconds = round(time.perf_counter() - self.start_time, 2)
        total_finished = self.state.completed + self.state.failed
        if total_finished >= self.state.total_tasks:
            self.state.status = "completed" if self.state.failed == 0 else "completed_with_errors"

        pct = (total_finished / self.state.total_tasks * 100.0) if self.state.total_tasks > 0 else 100.0
        logger.info(
            f"Progress [{pct:.1f}%]: {self.state.completed} Completed, {self.state.failed} Failed, {self.state.running} Running (Elapsed: {self.state.elapsed_seconds}s)"
        )

    def save_progress_state(self, output_file: Union[str, Path]) -> None:
        """Saves current progress state to JSON file."""
        self.state.elapsed_seconds = round(time.perf_counter() - self.start_time, 2)
        save_json(self.state.to_dict(), output_file)
