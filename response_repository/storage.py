"""
Response Storage and Repository Manager for FSL Research Framework.
Manages saving and loading experiment responses across JSON, JSONL, and CSV formats.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd

from response_repository.schema import ResponseRecord
from utilities.helpers import save_jsonl, load_jsonl, save_json
from utilities.logger import get_logger

logger = get_logger("response_repository")


class ResponseRepository:
    """Repository manager for persisting and loading structured experiment responses."""

    @classmethod
    def save_trial_responses(
        cls,
        experiment_id: str,
        trial_index: int,
        records: List[ResponseRecord],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Saves trial responses in both JSONL and CSV formats under the responses output directory.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_path / f"responses_trial_{trial_index + 1}.jsonl"
        csv_path = out_path / f"responses_trial_{trial_index + 1}.csv"

        # 1. Save JSONL format
        save_jsonl([rec.to_dict() for rec in records], jsonl_path)

        # 2. Save CSV format
        flat_dicts = [rec.to_flat_dict() for rec in records]
        df = pd.DataFrame(flat_dicts)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(
            f"Saved {len(records)} response records for Trial {trial_index + 1} "
            f"to JSONL ({jsonl_path.name}) and CSV ({csv_path.name})"
        )

        return {"jsonl": jsonl_path, "csv": csv_path}

    @classmethod
    def save_aggregate_responses(
        cls,
        experiment_id: str,
        all_records: List[ResponseRecord],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Saves aggregated response records across all trial runs in JSONL and CSV formats.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_path / "experiment_all_responses.jsonl"
        csv_path = out_path / "experiment_all_responses.csv"

        save_jsonl([rec.to_dict() for rec in all_records], jsonl_path)

        flat_dicts = [rec.to_flat_dict() for rec in all_records]
        df = pd.DataFrame(flat_dicts)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(
            f"Saved aggregate experiment repository ({len(all_records)} total records) "
            f"to JSONL ({jsonl_path.name}) and CSV ({csv_path.name})"
        )

        return {"jsonl": jsonl_path, "csv": csv_path}

    @classmethod
    def load_trial_responses(
        cls, output_dir: Union[str, Path], trial_index: int
    ) -> List[ResponseRecord]:
        """Loads saved response records for a specific trial."""
        jsonl_path = Path(output_dir) / f"responses_trial_{trial_index + 1}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Trial response file not found: {jsonl_path}")

        items = load_jsonl(jsonl_path)
        return [ResponseRecord.from_dict(item) for item in items]

    @classmethod
    def load_all_responses(cls, output_dir: Union[str, Path]) -> List[ResponseRecord]:
        """Loads all aggregate response records for an experiment or across all experiment folders."""
        base_path = Path(output_dir)
        jsonl_path = base_path / "experiment_all_responses.jsonl"

        if jsonl_path.exists():
            items = load_jsonl(jsonl_path)
            return [ResponseRecord.from_dict(item) for item in items]

        # Recursive search across subdirectories (e.g. results/responses/*)
        all_records: List[ResponseRecord] = []
        matching_files = list(base_path.rglob("experiment_all_responses.jsonl"))
        if not matching_files:
            matching_files = list(base_path.rglob("responses_trial_*.jsonl"))

        for mf in matching_files:
            try:
                items = load_jsonl(mf)
                all_records.extend([ResponseRecord.from_dict(item) for item in items])
            except Exception as e:
                logger.warning(f"Could not load response records from '{mf}': {e}")

        return all_records

    @staticmethod
    def to_dataframe(records: List[ResponseRecord]) -> pd.DataFrame:
        """Converts list of ResponseRecord objects into a pandas DataFrame."""
        flat_dicts = [rec.to_flat_dict() for rec in records]
        return pd.DataFrame(flat_dicts)
