"""
Evaluation Engine for FSL Research Framework.
Master evaluator coordinating task-specific performance, efficiency, and reliability metric calculation and file persistence.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd

from evaluation.schema import EvaluationMetricsResult
from evaluation.registry import TaskMetricRegistry
from evaluation.classification import ClassificationEvaluator
from evaluation.qa_eval import QAEvaluator
from evaluation.generation import GenerationEvaluator
from evaluation.efficiency import EfficiencyEvaluator
from evaluation.reliability import ReliabilityEvaluator
from response_repository import ResponseRecord, ResponseRepository
from utilities.constants import RUNS_DIR
from utilities.helpers import save_json
from utilities.logger import get_logger

logger = get_logger("evaluation_engine")


class EvaluationEngine:
    """Master evaluation engine managing full task-aware metric suite computation."""

    def __init__(self, experiment_id: str, output_dir: Optional[Union[str, Path]] = None):
        self.experiment_id = experiment_id
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = RUNS_DIR / "latest" / "experiments" / experiment_id / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_responses(
        self, records_by_trial: Dict[int, List[ResponseRecord]], task_type: str
    ) -> EvaluationMetricsResult:
        """
        Computes task-specific performance, efficiency, and reliability metrics across trial responses.
        """
        logger.info(f"Starting Evaluation Engine for Experiment '{self.experiment_id}' ({task_type})...")

        all_records: List[ResponseRecord] = []
        trial_perf_scores: List[float] = []
        sample_evals: List[Dict[str, Any]] = []

        primary_stat_col = TaskMetricRegistry.get_statistical_target(task_type)
        sample_fields = TaskMetricRegistry.get_sample_fields(task_type)

        # 1. Compute Performance Metrics per Trial and Overall
        for trial_idx, records in records_by_trial.items():
            all_records.extend(records)
            preds = [r.parsed_prediction if r.parse_success else r.generated_text for r in records]
            refs = [r.ground_truth for r in records]
            parse_succs = [r.parse_success for r in records]
            req_succs = [r.success for r in records]

            for r in records:
                s_entry = {
                    "experiment_id": r.experiment_id,
                    "trial_index": r.trial_index,
                    "sample_id": r.record_id,
                    "prompt_text": r.prompt_text,
                    "generated_text": r.generated_text,
                    "ground_truth": r.ground_truth,
                    "request_success": r.success,
                    "parse_success": r.parse_success,
                }
                # Attach only task-applicable sample fields
                for sf in sample_fields:
                    val = getattr(r, sf, None)
                    if val is None:
                        val = r.task_metrics.get(sf)
                    if val is not None:
                        s_entry[sf] = val
                sample_evals.append(s_entry)

            if task_type == "text_classification":
                t_metrics = ClassificationEvaluator.evaluate(
                    preds, refs, parse_successes=parse_succs, request_successes=req_succs
                )
            elif task_type == "question_answering":
                contexts = [r.prompt_text for r in records]
                t_metrics = QAEvaluator.evaluate(preds, refs, contexts=contexts, request_successes=req_succs)
            elif task_type == "text_generation":
                t_metrics = GenerationEvaluator.evaluate(preds, refs, request_successes=req_succs)
            else:
                t_metrics = ClassificationEvaluator.evaluate(
                    preds, refs, parse_successes=parse_succs, request_successes=req_succs
                )

            val = t_metrics.get(primary_stat_col)
            if val is not None:
                trial_perf_scores.append(val)

        # Overall aggregate performance metrics across all records
        all_preds = [r.parsed_prediction if r.parse_success else r.generated_text for r in all_records]
        all_refs = [r.ground_truth for r in all_records]
        all_parse_succs = [r.parse_success for r in all_records]
        all_req_succs = [r.success for r in all_records]

        if task_type == "text_classification":
            overall_perf = ClassificationEvaluator.evaluate(
                all_preds, all_refs, parse_successes=all_parse_succs, request_successes=all_req_succs
            )
        elif task_type == "question_answering":
            all_contexts = [r.prompt_text for r in all_records]
            overall_perf = QAEvaluator.evaluate(all_preds, all_refs, contexts=all_contexts, request_successes=all_req_succs)
        elif task_type == "text_generation":
            overall_perf = GenerationEvaluator.evaluate(all_preds, all_refs, request_successes=all_req_succs)
        else:
            overall_perf = ClassificationEvaluator.evaluate(
                all_preds, all_refs, parse_successes=all_parse_succs, request_successes=all_req_succs
            )

        # 2. Compute Efficiency Metrics
        overall_eff = EfficiencyEvaluator.evaluate(all_records)

        # 3. Compute Reliability Metrics
        overall_rel = ReliabilityEvaluator.evaluate(
            trial_records_map=records_by_trial,
            trial_performance_scores=trial_perf_scores,
        )

        # Build EvaluationMetricsResult
        result = EvaluationMetricsResult(
            experiment_id=self.experiment_id,
            task_type=task_type,
            performance_metrics=overall_perf,
            efficiency_metrics=overall_eff,
            reliability_metrics=overall_rel,
            sample_evaluations=sample_evals,
        )

        # Persist metrics to disk
        self.save_metrics(result)
        logger.info(f"Completed evaluation for '{self.experiment_id}'. Metrics saved to {self.output_dir}")
        return result

    def save_metrics(self, result: EvaluationMetricsResult) -> Dict[str, Path]:
        """Saves evaluation metrics result to JSON and CSV files."""
        json_path = self.output_dir / "metrics.json"
        csv_path = self.output_dir / "metrics.csv"

        # Save JSON
        save_json(result.to_dict(), json_path)

        # Save flattened CSV summary
        flat_summary = {
            "experiment_id": result.experiment_id,
            "task_type": result.task_type,
            **{f"perf_{k}": v for k, v in result.performance_metrics.items()},
            **{f"eff_{k}": v for k, v in result.efficiency_metrics.items()},
            **{f"rel_{k}": v for k, v in result.reliability_metrics.items()},
        }
        df = pd.DataFrame([flat_summary])
        df.to_csv(csv_path, index=False, encoding="utf-8")

        return {"json": json_path, "csv": csv_path}

    @classmethod
    def evaluate_experiment_folder(
        cls, experiment_id: str, responses_dir: Optional[Union[str, Path]] = None
    ) -> EvaluationMetricsResult:
        """
        Loads responses from disk for experiment_id and executes evaluation engine.
        """
        if responses_dir:
            resp_dir = Path(responses_dir)
        else:
            matching = list(RUNS_DIR.rglob(f"experiments/{experiment_id}/responses"))
            if matching:
                resp_dir = matching[0]
            else:
                resp_dir = RUNS_DIR / "latest" / "experiments" / experiment_id / "responses"

        if not resp_dir.exists():
            raise FileNotFoundError(f"Response directory not found for experiment: {resp_dir}")

        trial_files = sorted(resp_dir.glob("responses_trial_*.jsonl"))
        records_by_trial: Dict[int, List[ResponseRecord]] = {}

        task_type = "text_classification"
        for idx, tf in enumerate(trial_files):
            recs = ResponseRepository.load_trial_responses(resp_dir, trial_index=idx)
            records_by_trial[idx] = recs
            if recs and recs[0].task_type:
                task_type = recs[0].task_type

        engine = cls(experiment_id=experiment_id)
        return engine.evaluate_responses(records_by_trial, task_type=task_type)
