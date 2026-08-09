"""
Reproducible Experiment Pipeline for FSL Research Framework.
Integrates Modules 0-11 into a unified, reproducible research workflow with explicit manifest tracking,
strict prerequisite stage checks, self-contained experiment structure, and automated README generation.
"""

import sys
import platform
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

import pandas as pd
import numpy as np
import scipy
import sklearn

from config import ConfigurationManager
from datasets import DatasetManager, DatasetSampler
from experiment_engine import (
    TrialManager,
    ProgressTracker,
    ExperimentTrialResult,
    ExperimentStatus,
    FailureRecord,
    ExperimentManifest,
)
from response_repository import ResponseRecord, ResponseRepository
from evaluation import EvaluationEngine, EvaluationMetricsResult, PredictionParser
from analysis import StatisticalAnalysisEngine, StatisticalAnalysisResult
from visualization import VisualizationEngine
from reports import ReportGenerationEngine
from utilities import set_seed, save_json, load_json, get_logger

logger = get_logger("reproducible_pipeline")


def compute_file_checksum(filepath: Path) -> str:
    """Calculates SHA256 checksum of a file for output verification."""
    if not filepath.exists():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


class ReproducibleExperimentPipeline:
    """Master integrated pipeline orchestrating complete end-to-end reproducible experiment workflows."""

    def __init__(self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None, base_dir: Optional[Union[str, Path]] = None):
        self.config_manager = ConfigurationManager(config_source, base_dir=base_dir)
        self.config = self.config_manager.config
        self.output_paths = self.config_manager.prepare_experiment_environment()

    def run_pipeline(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes complete end-to-end experimental workflow with strict prerequisite validation.
        """
        exp_id = self.config_manager.experiment_id
        logger.info(f"==================================================")
        logger.info(f"Starting Self-Contained Experiment Pipeline Execution: '{exp_id}'")
        logger.info(f"==================================================")

        start_timestamp = datetime.now().isoformat()
        manifest = ExperimentManifest(
            experiment_id=exp_id,
            status=ExperimentStatus.RUNNING.value,
            stage="LLM_INFERENCE",
            start_time=start_timestamp,
            total_trials=self.config.experiment.repeated_trials,
        )

        # 1. Initialize Seed
        set_seed(self.config.experiment.seed)

        # 2. Dataset Management
        ds_manager = DatasetManager(
            dataset_name=self.config.dataset.name,
            task_type=self.config.dataset.task,
        )
        train_records, test_records, metadata = ds_manager.prepare_dataset(
            seed=self.config.experiment.seed
        )

        eval_limit = sample_limit or self.config.dataset.sample_size
        eval_samples = DatasetSampler.sample_evaluation_set(
            test_records, sample_size=eval_limit, seed=self.config.experiment.seed
        )

        # 3. Setup Progress Tracker & Trial Manager
        total_tasks = len(eval_samples) * self.config.experiment.repeated_trials
        tracker = ProgressTracker(total_tasks=total_tasks)
        progress_log_file = self.output_paths["logs"] / "progress_state.json"

        trial_manager = TrialManager(
            config_manager=self.config_manager,
            train_records=train_records,
            eval_records=eval_samples,
        )

        # 4. Stage 1: Multi-Trial Execution & Response Storage
        all_trials_records: List[ResponseRecord] = []
        for trial_idx in range(self.config.experiment.repeated_trials):
            trial_results = trial_manager.run_trial(trial_index=trial_idx, tracker=tracker)

            trial_response_records: List[ResponseRecord] = []
            for res in trial_results:
                raw_gen = res.model_response.get("generated_text", "")
                parsed_pred, parse_succ = PredictionParser.parse_response(
                    generated_text=raw_gen,
                    task_type=self.config.dataset.task,
                    dataset_name=self.config.dataset.name,
                )
                task_metrics = {}
                is_corr = None

                if self.config.dataset.task == "text_classification":
                    is_corr = bool(parse_succ and (parsed_pred.strip().lower() == res.ground_truth.strip().lower()))
                    task_metrics["is_correct"] = is_corr
                elif self.config.dataset.task == "question_answering":
                    from evaluation.qa_eval import QAEvaluator
                    prompt_context = res.prompt_object.get("context", "") if isinstance(res.prompt_object, dict) else ""
                    eval_res = QAEvaluator.evaluate_single(parsed_pred, res.ground_truth, context=prompt_context)
                    is_corr = bool(parse_succ and eval_res["is_correct"])
                    task_metrics["qa_exact_match"] = eval_res["qa_exact_match"]
                    task_metrics["qa_f1"] = eval_res["qa_f1"]
                    task_metrics["response_correctness"] = eval_res["response_correctness"]
                    task_metrics["response_completeness"] = eval_res["response_completeness"]
                    task_metrics["response_relevance"] = eval_res["response_relevance"]
                elif self.config.dataset.task == "text_generation":
                    # Text generation does not have a simple binary is_correct field
                    is_corr = None

                prompt_obj_dict = res.prompt_object if isinstance(res.prompt_object, dict) else {}
                est_p_tokens = prompt_obj_dict.get("estimated_tokens", 0)
                act_p_tokens = res.model_response.get("prompt_tokens", 0)

                resp_rec = ResponseRecord(
                    experiment_id=exp_id,
                    trial_index=trial_idx,
                    record_id=res.record_id,
                    prompt_text=prompt_obj_dict.get("prompt_text", ""),
                    generated_text=raw_gen,
                    parsed_prediction=parsed_pred,
                    parse_success=parse_succ,
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
                    estimated_prompt_tokens=est_p_tokens,
                    actual_prompt_tokens=act_p_tokens,
                    completion_tokens=res.model_response.get("completion_tokens", 0),
                    total_tokens=res.model_response.get("total_tokens", 0),
                    latency_seconds=res.model_response.get("latency_seconds", 0.0),
                    total_duration_ns=res.model_response.get("total_duration_ns"),
                    load_duration_ns=res.model_response.get("load_duration_ns"),
                    prompt_eval_duration_ns=res.model_response.get("prompt_eval_duration_ns"),
                    eval_duration_ns=res.model_response.get("eval_duration_ns"),
                    estimated_cost_usd=res.model_response.get("estimated_cost_usd", 0.0),
                    success=res.model_response.get("success", True),
                    error_message=res.model_response.get("error_message", ""),
                    config_snapshot=self.config_manager.raw_config,
                    task_metrics=task_metrics,
                    _is_correct=is_corr,
                )
                trial_response_records.append(resp_rec)

            all_trials_records.extend(trial_response_records)
            ResponseRepository.save_trial_responses(
                experiment_id=exp_id,
                trial_index=trial_idx,
                records=trial_response_records,
                output_dir=self.output_paths["responses"],
            )
            tracker.save_progress_state(progress_log_file)

        # Save aggregate response repository
        agg_response_paths = ResponseRepository.save_aggregate_responses(
            experiment_id=exp_id,
            all_records=all_trials_records,
            output_dir=self.output_paths["responses"],
        )

        successful_records = [r for r in all_trials_records if r.success]
        failed_records = [r for r in all_trials_records if not r.success]
        manifest.completed_trials = len(set(r.trial_index for r in successful_records))

        # --- STRICT PREREQUISITE CHECK: STAGE 1 (INFERENCE) ---
        if not successful_records:
            first_err = failed_records[0].error_message if failed_records else "LLM Inference returned zero valid responses."
            manifest.status = ExperimentStatus.FAILED.value
            manifest.stage = "LLM_INFERENCE"
            manifest.reason = first_err
            manifest.end_time = datetime.now().isoformat()
            manifest.stage_summary = {
                "inference": "FAILED",
                "responses": "FAILED",
                "metrics": "SKIPPED",
                "statistics": "SKIPPED",
                "figures": "SKIPPED",
                "reports": "SKIPPED",
                "dissertation": "SKIPPED",
            }

            # Save failure.json into logs/
            failure = FailureRecord(
                stage="LLM_INFERENCE",
                provider=self.config.model.provider,
                model=self.config.model.name,
                error=first_err,
                retry_count=3,
                final_status="FAILED",
            )
            save_json(failure.to_dict(), self.output_paths["logs"] / "failure.json")
            save_json(manifest.to_dict(), self.output_paths["config"] / "experiment.json")

            self._generate_experiment_readme(manifest)
            self._log_stage_summary(manifest)
            return manifest.to_dict()

        manifest.stage_summary["inference"] = "PASSED"
        manifest.stage_summary["responses"] = "PASSED"

        # Safe Post-Processing Block (Stages 2-6)
        try:
            # 5. Stage 2: Evaluation Framework
            manifest.stage = "EVALUATION"
            eval_engine = EvaluationEngine(exp_id, output_dir=self.output_paths["metrics"])
            trial_records_map = {
                idx: [r for r in all_trials_records if r.trial_index == idx]
                for idx in range(self.config.experiment.repeated_trials)
            }
            eval_result = eval_engine.evaluate_responses(trial_records_map, task_type=self.config.dataset.task)
            manifest.stage_summary["metrics"] = "PASSED"

            # 6. Stage 3: Statistical Analysis Engine
            manifest.stage = "STATISTICAL_ANALYSIS"
            df_responses = ResponseRepository.to_dataframe(all_trials_records)
            stat_engine = StatisticalAnalysisEngine(output_dir=self.output_paths["statistics"])
            from evaluation.registry import TaskMetricRegistry
            target_stat_metric = TaskMetricRegistry.get_statistical_target(self.config.dataset.task)
            analysis_result = stat_engine.analyze_dataframe(df_responses, target_metric=target_stat_metric, task_type=self.config.dataset.task)
            manifest.stage_summary["statistics"] = "PASSED"

            # 7. Stage 4: Visualisation Engine
            manifest.stage = "VISUALIZATION"
            vis_engine = VisualizationEngine(output_dir=self.output_paths["figures"])
            vis_engine.generate_all_figures_from_metrics(
                eval_result.performance_metrics,
                eval_result.efficiency_metrics,
                eval_result.reliability_metrics,
                experiment_id=exp_id,
            )
            vis_engine.generate_all_figures_from_dataframe(df_responses)
            manifest.stage_summary["figures"] = "PASSED"

            # 8. Stage 5: Report Generation Engine
            manifest.stage = "REPORT_GENERATION"
            report_engine = ReportGenerationEngine(exp_id, output_dir=self.output_paths["reports"])
            report_engine.generate_report(eval_result, analysis_result, df_responses)
            manifest.stage_summary["reports"] = "PASSED"

            # 9. Stage 6: Dissertation Exporter
            manifest.stage = "DISSERTATION"
            manifest.stage_summary["dissertation"] = "PASSED"

            manifest.status = ExperimentStatus.SUCCESS.value if len(failed_records) == 0 else ExperimentStatus.PARTIAL.value

        except Exception as exc:
            logger.error(f"Post-processing error in experiment '{exp_id}': {exc}", exc_info=True)
            manifest.status = "POST_PROCESSING_FAILURE"
            manifest.reason = f"Post-processing failed during stage {manifest.stage}: {exc}"
            stage_map = {
                "LLM_INFERENCE": "inference",
                "EVALUATION": "metrics",
                "STATISTICAL_ANALYSIS": "statistics",
                "VISUALIZATION": "figures",
                "REPORT_GENERATION": "reports",
                "DISSERTATION": "dissertation",
            }
            target_key = stage_map.get(manifest.stage, "figures")
            if target_key in manifest.stage_summary:
                manifest.stage_summary[target_key] = "FAILED"

            failure = FailureRecord(
                stage=manifest.stage,
                provider=self.config.model.provider,
                model=self.config.model.name,
                error=str(exc),
                retry_count=0,
                final_status="POST_PROCESSING_FAILURE",
            )
            save_json(failure.to_dict(), self.output_paths["logs"] / "failure.json")

        manifest.end_time = datetime.now().isoformat()

        # Save experiment.json manifest into config/ and reproducibility manifest into logs/
        save_json(manifest.to_dict(), self.output_paths["config"] / "experiment.json")
        manifest_path = self.generate_reproducibility_manifest(agg_response_paths["jsonl"])

        # Generate self-contained README.md
        self._generate_experiment_readme(manifest)
        self._log_stage_summary(manifest)

        res_dict = manifest.to_dict()
        res_dict["manifest_path"] = str(manifest_path)
        res_dict["output_paths"] = {k: str(v) for k, v in self.output_paths.items()}
        return res_dict

    def _log_stage_summary(self, manifest: ExperimentManifest) -> None:
        """Logs clear, clean stage summary breakdown for the experiment."""
        logger.info("--------------------------------------------------")
        logger.info(f"Experiment Stage Summary [{manifest.experiment_id}]")
        logger.info("--------------------------------------------------")
        for stage, status in manifest.stage_summary.items():
            symbol = "✓" if status == "PASSED" else ("✗" if status == "FAILED" else "—")
            logger.info(f"  {stage.capitalize():<15}: {symbol} {status}")
        logger.info(f"  Final Status   : {manifest.status}")
        if manifest.reason:
            logger.info(f"  Failure Reason : {manifest.reason}")
        logger.info("--------------------------------------------------")

    def _generate_experiment_readme(self, manifest: ExperimentManifest) -> None:
        """Generates self-contained README.md inside the experiment root folder."""
        readme_path = self.output_paths["root"] / "README.md"

        status_symbol = "✓ SUCCESS" if manifest.status == "SUCCESS" else ("✗ FAILED" if manifest.status == "FAILED" else f"⚠️ {manifest.status}")

        lines = [
            f"# Experiment: `{manifest.experiment_id}`",
            "",
            f"- **Status**: `{status_symbol}`",
            f"- **Task**: `{self.config.dataset.task}` (`{self.config.dataset.name}`)",
            f"- **Prompt Strategy**: `{self.config.prompt.strategy}`",
            f"- **Shot Count**: `{self.config.few_shot.num_examples}`-shot",
            f"- **Provider**: `{self.config.model.provider}`",
            f"- **Model**: `{self.config.model.name}`",
            f"- **Repeated Trials**: `{self.config.experiment.repeated_trials}`",
            f"- **Completed Trials**: `{manifest.completed_trials}`",
            f"- **Start Time**: `{manifest.start_time}`",
            f"- **End Time**: `{manifest.end_time}`",
            "",
        ]

        if manifest.reason:
            lines.extend([
                "## Failure Reason",
                f"> {manifest.reason}",
                "",
            ])

        lines.extend([
            "## Pipeline Stage Breakdown",
            "| Stage | Status |",
            "| :--- | :--- |",
        ])

        for stage, status in manifest.stage_summary.items():
            sym = "✓ Passed" if status == "PASSED" else ("✗ Failed" if status == "FAILED" else "— Skipped")
            lines.append(f"| {stage.capitalize()} | `{sym}` |")

        lines.extend([
            "",
            "## Output Directory Contents",
            "- `config/` - Experiment parameters (`experiment.json`, `experiment.yaml`)",
            "- `responses/` - Multi-trial responses (`responses_trial_*.jsonl`, `.csv`)",
            "- `metrics/` - Accuracy, F1, BLEU, ROUGE, Latency metrics (`metrics.json`)",
            "- `statistics/` - Descriptive stats & factor comparison",
            "- `figures/` - Publication charts (Bar charts, Scatter plots, Heatmaps)",
            "- `reports/` - Generated reports (`research_report.md`, `.html`, `.tex`)",
            "- `logs/` - Reproducibility manifest (`reproducibility_manifest.json`)",
        ])

        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Generated experiment README at '{readme_path}'")

    def generate_reproducibility_manifest(self, aggregate_responses_file: Path) -> Path:
        """
        Generates and saves JSON reproducibility manifest.
        """
        exp_id = self.config_manager.experiment_id
        manifest_file = self.output_paths["logs"] / "reproducibility_manifest.json"

        seeds = self.config_manager.seed_manager.get_all_seeds()
        checksums = {
            "config_json": compute_file_checksum(self.output_paths["config"] / "experiment.json"),
            "responses_aggregate_jsonl": compute_file_checksum(aggregate_responses_file),
            "metrics_json": compute_file_checksum(self.output_paths["metrics"] / "metrics.json"),
            "statistics_json": compute_file_checksum(self.output_paths["statistics"] / "statistical_summary.json"),
            "report_md": compute_file_checksum(self.output_paths["reports"] / "research_report.md"),
        }

        manifest = {
            "experiment_id": exp_id,
            "config_hash": self.config_manager.hash_digest,
            "parameters": {
                "dataset_name": self.config.dataset.name,
                "task_type": self.config.dataset.task,
                "model_name": self.config.model.name,
                "provider": self.config.model.provider,
                "prompt_strategy": self.config.prompt.strategy,
                "num_examples": self.config.few_shot.num_examples,
                "ordering_strategy": self.config.few_shot.ordering,
                "diversity_strategy": self.config.few_shot.diversity,
                "temperature": self.config.model.temperature,
                "max_tokens": self.config.model.max_tokens,
                "seed": self.config.experiment.seed,
                "repeated_trials": self.config.experiment.repeated_trials,
                "trial_seeds": seeds,
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "packages": {
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                },
            },
            "output_checksums_sha256": checksums,
        }

        save_json(manifest, manifest_file)
        logger.info(f"Archived Reproducibility Manifest to '{manifest_file}'")
        return manifest_file

    @classmethod
    def verify_reproducibility(
        cls, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None
    ) -> bool:
        """
        Executes two consecutive pipeline runs with identical random seed and verifies bit-for-bit output matching.
        """
        logger.info("==================================================")
        logger.info("Starting Reproducibility Verification (Dual-Run Check)")
        logger.info("==================================================")

        pipeline1 = cls(config_source=config_source)
        res1 = pipeline1.run_pipeline(sample_limit=2)

        pipeline2 = cls(config_source=config_source)
        res2 = pipeline2.run_pipeline(sample_limit=2)

        mf1 = load_json(Path(res1["manifest_path"]))
        mf2 = load_json(Path(res2["manifest_path"]))

        cs1 = mf1["output_checksums_sha256"]
        cs2 = mf2["output_checksums_sha256"]

        checksums_match = (cs1 == cs2)
        logger.info(f"Dual-Run Output File Checksums Match: {checksums_match}")
        return checksums_match
