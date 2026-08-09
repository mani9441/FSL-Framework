"""
Experiment Controller for FSL Research Framework.
Master controller executing single experiments and matrix parameter sweeps.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from config.manager import ConfigurationManager
from datasets.manager import DatasetManager
from datasets.sampler import DatasetSampler
from experiment_engine.trial_manager import TrialManager
from experiment_engine.progress_tracker import ProgressTracker
from experiment_engine.schema import ExperimentTrialResult
from response_repository import ResponseRecord, ResponseRepository
from utilities.logger import get_logger

logger = get_logger("experiment_controller")


class ExperimentController:
    """Master controller orchestrating complete end-to-end experiment pipelines."""

    def __init__(self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None):
        self.config_manager = ConfigurationManager(config_source)
        self.config = self.config_manager.config
        self.output_paths = self.config_manager.prepare_experiment_environment()

    def run_experiment(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes complete experiment pipeline across all configured repeated trials.
        """
        exp_id = self.config_manager.experiment_id
        logger.info(f"==================================================")
        logger.info(f"Executing Experiment Run: '{exp_id}'")
        logger.info(f"==================================================")

        # 1. Prepare Dataset
        ds_manager = DatasetManager(
            dataset_name=self.config.dataset.name,
            task_type=self.config.dataset.task,
        )
        train_records, test_records, metadata = ds_manager.prepare_dataset(
            seed=self.config.experiment.seed
        )

        # Limit eval sample size if specified
        eval_limit = sample_limit or self.config.dataset.sample_size
        eval_samples = DatasetSampler.sample_evaluation_set(
            test_records, sample_size=eval_limit, seed=self.config.experiment.seed
        )

        logger.info(
            f"Dataset '{self.config.dataset.name}' ready: "
            f"Train candidates={len(train_records)}, Eval target samples={len(eval_samples)}"
        )

        # 2. Setup Progress Tracker
        total_tasks = len(eval_samples) * self.config.experiment.repeated_trials
        tracker = ProgressTracker(total_tasks=total_tasks)
        progress_log_file = self.output_paths["logs"] / "progress_state.json"

        # 3. Instantiate TrialManager
        trial_manager = TrialManager(
            config_manager=self.config_manager,
            train_records=train_records,
            eval_records=eval_samples,
        )

        # 4. Execute Repeated Trials
        all_trials_records: List[ResponseRecord] = []

        for trial_idx in range(self.config.experiment.repeated_trials):
            trial_results: List[ExperimentTrialResult] = trial_manager.run_trial(
                trial_index=trial_idx, tracker=tracker
            )

            # Convert ExperimentTrialResult -> ResponseRecord
            trial_response_records: List[ResponseRecord] = []
            for res in trial_results:
                prompt_info = res.prompt_object
                model_info = res.model_response

                resp_rec = ResponseRecord(
                    experiment_id=exp_id,
                    trial_index=trial_idx,
                    record_id=res.record_id,
                    prompt_text=prompt_info.get("prompt_text", ""),
                    generated_text=model_info.get("generated_text", ""),
                    ground_truth=res.ground_truth,
                    timestamp=res.timestamp,
                    model_name=self.config.model.name,
                    provider=self.config.model.provider,
                    dataset_name=self.config.dataset.name,
                    task_type=self.config.dataset.task,
                    prompt_strategy=self.config.prompt.strategy,
                    num_examples=self.config.few_shot.num_examples,
                    ordering_strategy=self.config.few_shot.ordering,
                    diversity_strategy=self.config.few_shot.diversity,
                    prompt_tokens=model_info.get("prompt_tokens", 0),
                    completion_tokens=model_info.get("completion_tokens", 0),
                    total_tokens=model_info.get("total_tokens", 0),
                    latency_seconds=model_info.get("latency_seconds", 0.0),
                    estimated_cost_usd=model_info.get("estimated_cost_usd", 0.0),
                    success=model_info.get("success", True),
                    error_message=model_info.get("error_message", ""),
                    config_snapshot=self.config_manager.raw_config,
                )
                trial_response_records.append(resp_rec)

            all_trials_records.extend(trial_response_records)

            # Persist trial responses in JSONL and CSV via ResponseRepository
            ResponseRepository.save_trial_responses(
                experiment_id=exp_id,
                trial_index=trial_idx,
                records=trial_response_records,
                output_dir=self.output_paths["responses"],
            )
            tracker.save_progress_state(progress_log_file)

        # 5. Persist aggregate experiment repository
        ResponseRepository.save_aggregate_responses(
            experiment_id=exp_id,
            all_records=all_trials_records,
            output_dir=self.output_paths["responses"],
        )

        logger.info(f"==================================================")
        logger.info(f"Experiment Run '{exp_id}' Completed Successfully.")
        logger.info(f"==================================================")

        return {
            "experiment_id": exp_id,
            "status": "completed",
            "total_trials": self.config.experiment.repeated_trials,
            "records_per_trial": len(eval_samples),
            "total_records": len(all_trials_records),
            "output_paths": {k: str(v) for k, v in self.output_paths.items()},
        }

    @classmethod
    def run_matrix_sweep(
        cls, config_sources: List[Union[str, Path, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Executes a matrix parameter sweep across multiple experiment configurations.
        """
        logger.info(f"Starting parameter matrix sweep across {len(config_sources)} experiment configurations...")
        sweep_summaries: List[Dict[str, Any]] = []

        for idx, cfg in enumerate(config_sources):
            logger.info(f"Matrix Sweep Item {idx + 1}/{len(config_sources)}")
            controller = cls(config_source=cfg)
            summary = controller.run_experiment()
            sweep_summaries.append(summary)

        logger.info(f"Matrix sweep complete ({len(sweep_summaries)} experiments executed).")
        return sweep_summaries
