"""
Output Path Manager for FSL Research Framework.
Manages per-experiment isolated directory structures and configuration snapshot archiving.
"""

from pathlib import Path
from typing import Dict, Any, Union
import yaml

from utilities.constants import RESULTS_DIR
from utilities.helpers import save_json
from utilities.logger import get_logger

logger = get_logger("path_manager")


class OutputPathManager:
    """Allocates self-contained experiment directory paths under results/runs/run_<timestamp>/experiments/EXP_<ID>/."""

    def __init__(self, experiment_id: str, base_dir: Union[str, Path] = None):
        self.experiment_id = experiment_id

        if base_dir is None or str(base_dir) == "results" or Path(base_dir) == RESULTS_DIR:
            from utilities.constants import create_run_directory
            self.run_dir = create_run_directory()
        else:
            self.run_dir = Path(base_dir)

        if self.run_dir.name == "experiments":
            self.exp_root = self.run_dir / experiment_id
        elif self.run_dir.name == experiment_id and self.run_dir.parent.name == "experiments":
            self.exp_root = self.run_dir
        else:
            self.exp_root = self.run_dir / "experiments" / experiment_id

        # Isolated subdirectories inside exp_root
        self.config_dir = self.exp_root / "config"
        self.logs_dir = self.exp_root / "logs"
        self.responses_dir = self.exp_root / "responses"
        self.metrics_dir = self.exp_root / "metrics"
        self.statistics_dir = self.exp_root / "statistics"
        self.figures_dir = self.exp_root / "figures"
        self.reports_dir = self.exp_root / "reports"
        self.observations_dir = self.exp_root / "observations"
        self.dissertation_dir = self.exp_root / "dissertation"

        self.subdirs = [
            self.exp_root,
            self.config_dir,
            self.logs_dir,
            self.responses_dir,
            self.metrics_dir,
            self.statistics_dir,
            self.figures_dir,
            self.reports_dir,
            self.observations_dir,
            self.dissertation_dir,
        ]

    def create_directories(self) -> Dict[str, Path]:
        """Creates all required self-contained experiment subdirectories."""
        for path in self.subdirs:
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Allocated self-contained output directory structure at '{self.exp_root}'")
        return self.get_paths()

    def get_paths(self) -> Dict[str, Path]:
        """Returns dictionary of all allocated experiment paths."""
        return {
            "root": self.exp_root,
            "config": self.config_dir,
            "logs": self.logs_dir,
            "responses": self.responses_dir,
            "metrics": self.metrics_dir,
            "statistics": self.statistics_dir,
            "figures": self.figures_dir,
            "reports": self.reports_dir,
            "observations": self.observations_dir,
            "dissertation": self.dissertation_dir,
        }

    def save_config_snapshot(self, config_dict: Dict[str, Any]) -> None:
        """
        Saves snapshot copy of configuration into config/ folder inside experiment root.
        """
        self.create_directories()

        # Save JSON snapshot
        json_path = self.config_dir / "experiment.json"
        save_json(config_dict, json_path)

        # Save YAML snapshot
        yaml_path = self.config_dir / "experiment.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)

        logger.info(f"Saved experiment configuration snapshots to '{json_path}' and '{yaml_path}'")
