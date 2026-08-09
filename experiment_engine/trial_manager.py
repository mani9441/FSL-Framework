"""
Trial Manager for FSL Research Framework.
Executes repeated experimental trials, prompt assembly, and inference across records.
"""

from typing import List, Optional
from datasets.schema import StandardRecord
from config.manager import ConfigurationManager
from prompt_engine import PromptGenerator, ContextBuilder
from model_interfaces import ModelExecutionManager
from experiment_engine.schema import ExperimentTrialResult
from experiment_engine.progress_tracker import ProgressTracker
from utilities.logger import get_logger

logger = get_logger("trial_manager")


class TrialManager:
    """Manager executing individual multi-trial runs for an experiment."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        train_records: List[StandardRecord],
        eval_records: List[StandardRecord],
    ):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.train_records = train_records
        self.eval_records = eval_records

        # Instantiate ModelExecutionManager
        self.execution_manager = ModelExecutionManager(
            model_name=self.config.model.name,
            provider=self.config.model.provider,
        )

    def run_trial(
        self, trial_index: int, tracker: Optional[ProgressTracker] = None
    ) -> List[ExperimentTrialResult]:
        """
        Executes a single zero-indexed trial across all evaluation records.
        """
        # Set trial seed
        applied_seed = self.config_manager.seed_manager.set_trial_seed(trial_index)
        logger.info(
            f"Starting Trial {trial_index + 1}/{self.config.experiment.repeated_trials} "
            f"(Seed={applied_seed}, Eval Samples={len(self.eval_records)})"
        )

        results: List[ExperimentTrialResult] = []

        for record in self.eval_records:
            if tracker:
                tracker.start_task()

            try:
                # 1. Select few-shot demonstrations
                demos = ContextBuilder.build_few_shot_context(
                    candidate_pool=self.train_records,
                    query_record=record,
                    num_examples=self.config.few_shot.num_examples,
                    diversity=self.config.few_shot.diversity,
                    ordering=self.config.few_shot.ordering,
                    seed=applied_seed,
                )

                # 2. Generate prompt and validate against model context limit
                ctx_limit = getattr(self.config.model, "context_limit", 2048)
                prompt_obj = PromptGenerator.generate(
                    query_record=record,
                    demonstrations=demos,
                    strategy=self.config.prompt.strategy,
                    instruction_text=self.config.prompt.instruction_text,
                    max_token_limit=ctx_limit,
                )

                # 3. Execute model inference ONLY if prompt meets context limit requirements
                if not prompt_obj.is_valid:
                    err_msg = f"Prompt validation failed: {'; '.join(prompt_obj.validation_warnings)}"
                    logger.warning(f"Record '{record.id}' failed prompt pre-validation: {err_msg}")
                    from model_interfaces.schema import ModelResponse
                    model_resp = ModelResponse(
                        generated_text="",
                        model_name=self.config.model.name,
                        provider=self.config.model.provider,
                        latency_seconds=0.0,
                        prompt_tokens=prompt_obj.estimated_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_obj.estimated_tokens,
                        success=False,
                        error_message=err_msg,
                    )
                else:
                    model_resp = self.execution_manager.execute_with_retry(
                        prompt_text=prompt_obj.prompt_text,
                        temperature=self.config.model.temperature,
                        max_tokens=self.config.model.max_tokens,
                        retries=3,
                    )


                # 4. Construct trial result
                trial_res = ExperimentTrialResult(
                    experiment_id=self.config_manager.experiment_id,
                    trial_index=trial_index,
                    record_id=record.id,
                    prompt_object=prompt_obj.to_dict(),
                    model_response=model_resp.to_dict(),
                    ground_truth=record.answer,
                )
                results.append(trial_res)

                if tracker:
                    if model_resp.success:
                        tracker.complete_task()
                    else:
                        tracker.fail_task()

            except Exception as e:
                logger.error(f"Trial {trial_index + 1} execution error on record '{record.id}': {e}")
                if tracker:
                    tracker.fail_task()

        logger.info(f"Completed Trial {trial_index + 1}/{self.config.experiment.repeated_trials} ({len(results)} records).")
        return results
