"""
Experiment Matrix Generator & Campaign Manager for FSL Research Framework (Phase E).
Generates multi-factorial experiment matrices, supports CLI filters (--provider, --model, --prompt, --shots, --task),
handles rate-limits, manages campaign-level comparison files, and isolates self-contained experiments.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from utilities.constants import RESULTS_DIR, RUNS_DIR, create_run_directory
from utilities.helpers import save_json, load_json
from utilities.logger import get_logger

logger = get_logger("matrix_generator")


class ExperimentMatrixGenerator:
    """Generates and manages multi-factorial research experimental campaigns with CLI filtering."""

    TASKS = [
        ("ag_news", "text_classification"),
        ("squad_v2", "question_answering"),
        ("cnn_dailymail", "text_generation"),
    ]

    SHOT_COUNTS = [0, 1, 3, 5]
    REPEATED_TRIALS = 3

    @classmethod
    def get_models(cls) -> List[tuple]:
        return [
            (os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), "groq"),
            (os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"), "huggingface"),
            (os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"), "google"),
            ("llama3.1:8b", "ollama"),
            ("qwen3:8b", "ollama"),
            ("mistral:7b", "ollama"),
            ("gemma3:4b", "ollama"),
            ("phi4-mini", "ollama"),
        ]

    @classmethod
    def is_model_enabled(cls, model_name: str, provider: str) -> bool:
        """
        Checks if a model is enabled via environment variables.
        """
        candidate_env_vars = cls._get_model_env_var_names(model_name, provider)
        for env_var in candidate_env_vars:
            if env_var in os.environ:
                val = os.environ[env_var].strip().lower()
                return val in ("true", "1", "yes", "on", "t")
        return True

    @classmethod
    def _get_model_env_var_names(cls, model_name: str, provider: str) -> List[str]:
        candidates = []
        lower_m = model_name.lower()
        if provider.lower() == "ollama":
            if "llama3.1:8b" in lower_m or "llama3.1:latest" in lower_m:
                candidates.extend(["OLLAMA_LLAMA31_8B_ENABLED", "OLLAMA_LLAMA31_LATEST_ENABLED"])
            elif "qwen3:8b" in lower_m or "qwen3" in lower_m:
                candidates.append("OLLAMA_QWEN3_8B_ENABLED")
            elif "mistral:7b" in lower_m or "mistral" in lower_m:
                candidates.append("OLLAMA_MISTRAL_7B_ENABLED")
            elif "gemma3:4b" in lower_m or "gemma3" in lower_m:
                candidates.append("OLLAMA_GEMMA3_4B_ENABLED")
            elif "phi4-mini" in lower_m or "phi4" in lower_m:
                candidates.append("OLLAMA_PHI4_MINI_ENABLED")

        clean_name = model_name.replace(":", "_").replace(".", "").replace("/", "_").replace("-", "_").upper()
        provider_clean = provider.upper()
        if provider_clean == "GOOGLE":
            candidates.extend([
                f"GOOGLE_{clean_name}_ENABLED",
                f"{clean_name}_ENABLED",
                "GOOGLE_GEMINI_35_FLASH_LITE_ENABLED",
                "GOOGLE_GEMINI_35_FLASH_ENABLED",
                "GOOGLE_GEMINI_20_FLASH_ENABLED",
                "GEMINI_MODEL_ENABLED",
            ])
        elif provider_clean == "GROQ":
            candidates.extend([
                f"GROQ_{clean_name}_ENABLED",
                "GROQ_LLAMA31_8B_INSTANT_ENABLED",
                "GROQ_MODEL_ENABLED",
            ])
        elif provider_clean in ["HUGGINGFACE", "HF"]:
            candidates.extend([
                f"HF_{clean_name}_ENABLED",
                f"HUGGINGFACE_{clean_name}_ENABLED",
                "HF_QWEN25_7B_INSTRUCT_ENABLED",
                "HF_MODEL_ENABLED",
            ])
        elif provider_clean == "OLLAMA":
            candidates.append(f"OLLAMA_{clean_name}_ENABLED")

        return candidates

    @classmethod
    def select_models_for_campaign(
        cls,
        provider_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
    ) -> List[tuple]:
        """
        Selects models according to campaign filters and environment enablement rules:
        - If --model is provided (Case C/D):
            - Run explicitly requested model regardless of *_ENABLED flag.
        - If --model is NOT provided (Case A/B):
            - If --provider is provided: Select models belonging to that provider.
            - If --provider is NOT provided: Select models across ALL providers.
            - Keep only ENABLED models (*_ENABLED == true).
        """
        all_registered = cls.get_models()

        if model_filter:
            # Case C / Case D: Explicit model provided
            # Enable flag is IGNORED for explicit model requests
            matched = []
            mf_lower = model_filter.lower()
            for m_name, m_prov in all_registered:
                if provider_filter and provider_filter.lower() != m_prov.lower():
                    continue

                if (mf_lower in m_name.lower() or
                    m_name.lower() in mf_lower or
                    (mf_lower in ["llama3.1:latest", "llama3.1"] and "llama3.1" in m_name.lower())):
                    matched.append((model_filter if mf_lower == "llama3.1:latest" else m_name, m_prov))

            if not matched:
                from model_interfaces.execution_manager import ModelExecutionManager
                prov = provider_filter or ModelExecutionManager._infer_provider(model_filter)
                matched.append((model_filter, prov))

            return matched
        else:
            # Case A / Case B: Automatic model selection (model_filter is None)
            candidates = []
            for m_name, m_prov in all_registered:
                if provider_filter and provider_filter.lower() != m_prov.lower():
                    continue
                candidates.append((m_name, m_prov))

            enabled_models = [
                (m_name, m_prov) for m_name, m_prov in candidates
                if cls.is_model_enabled(m_name, m_prov)
            ]

            logger.info("==================================================")
            logger.info("Automatic Campaign Model Selection")
            logger.info("==================================================")
            logger.info("Enabled models:")
            for m_name, m_prov in enabled_models:
                logger.info(f"  - {m_name} ({m_prov})")
            logger.info("==================================================")

            return enabled_models

    @classmethod
    def generate_experiment_matrix(
        cls,
        sample_size: int = 20,
        base_seed: int = 42,
        provider_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None,
        shot_filter: Optional[int] = None,
        task_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates experiment matrix configurations with optional CLI filtering flags.
        Controlled factor: shot count (0, 1, 3, 5).
        """
        matrix: List[Dict[str, Any]] = []
        idx = 1
        selected_models = cls.select_models_for_campaign(
            provider_filter=provider_filter,
            model_filter=model_filter,
        )

        for ds_name, task_type in cls.TASKS:
            if task_filter and task_filter.lower() not in [task_type.lower(), ds_name.lower()]:
                continue

            for shot in cls.SHOT_COUNTS:
                if shot_filter is not None and int(shot_filter) != shot:
                    continue

                for model_name, provider in selected_models:
                    config_spec = {
                        "experiment_idx": idx,
                        "experiment": {
                            "name": f"Exp {idx:02d} - {task_type} {shot}shot {model_name}",
                            "description": f"Campaign run {idx} evaluating {model_name} on {ds_name} with {shot}-shot standardized prompting",
                            "seed": base_seed,
                            "repeated_trials": cls.REPEATED_TRIALS,
                        },
                        "dataset": {
                            "name": ds_name,
                            "task": task_type,
                            "split": "test",
                            "sample_size": sample_size,
                        },
                        "prompt": {
                            "strategy": "standard_few_shot",
                            "instruction_text": "Classify the input text into exactly one of the designated target categories: World, Sports, Business, Sci/Tech. Respond with ONLY the single chosen category label and nothing else." if task_type == "text_classification" else f"Perform {task_type.replace('_', ' ')} accurately.",
                        },
                        "few_shot": {
                            "num_examples": shot,
                            "ordering": "random",
                            "diversity": "medium",
                        },
                        "model": {
                            "name": model_name,
                            "provider": provider,
                            "temperature": 0.0,
                            "max_tokens": 512,
                        },
                        "evaluation": {
                            "metrics": ["accuracy", "f1", "bleu", "rouge"],
                        },
                    }
                    matrix.append(config_spec)
                    idx += 1

        logger.info(f"Generated filtered campaign matrix: {len(matrix)} experiment configurations.")
        return matrix


    @classmethod
    def run_campaign(
        cls,
        sample_size: int = 10,
        skip_completed: bool = True,
        run_id: Optional[str] = None,
        provider_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None,
        shot_filter: Optional[int] = None,
        task_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete experimental campaign storing outputs in an isolated run directory.
        Campaign level contains campaign.json, summary.csv, and comparison.csv.
        """
        if not run_id:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run_dir = create_run_directory(run_id)
        campaign_manifest_path = run_dir / "campaign.json"

        matrix = cls.generate_experiment_matrix(
            sample_size=sample_size,
            provider_filter=provider_filter,
            model_filter=model_filter,
            strategy_filter=strategy_filter,
            shot_filter=shot_filter,
            task_filter=task_filter,
        )

        completed_runs = {}

        if campaign_manifest_path.exists() and skip_completed:
            try:
                prev_manifest = load_json(campaign_manifest_path)
                completed_runs = prev_manifest.get("completed_runs", {})
                logger.info(f"Resuming campaign from manifest in '{run_dir}'. {len(completed_runs)} runs already completed.")
            except Exception as e:
                logger.warning(f"Could not load existing manifest: {e}")

        total_configs = len(matrix)
        results_summary = []

        from pipeline import ReproducibleExperimentPipeline

        for item in matrix:
            spec_idx = item["experiment_idx"]
            exp_key = f"exp_{spec_idx:03d}"

            if skip_completed and exp_key in completed_runs:
                logger.info(f"Skipping completed campaign run [{spec_idx}/{total_configs}]: '{exp_key}'")
                results_summary.append(completed_runs[exp_key])
                continue

            logger.info(f"==================================================")
            logger.info(f"Executing Campaign Run [{spec_idx}/{total_configs}] in {run_dir.name}: {item['experiment']['name']}")
            logger.info(f"Running model: {item['model']['name']}")
            logger.info(f"==================================================")

            try:
                pipeline = ReproducibleExperimentPipeline(config_source=item, base_dir=run_dir)
                run_summary = pipeline.run_pipeline(sample_limit=sample_size)

                completed_runs[exp_key] = run_summary
                results_summary.append(run_summary)

                # Checkpoint campaign.json
                save_json(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "total_campaign_configs": total_configs,
                        "completed_count": len(completed_runs),
                        "filters": {
                            "provider": provider_filter,
                            "model": model_filter,
                            "strategy": strategy_filter,
                            "shots": shot_filter,
                            "task": task_filter,
                        },
                        "completed_runs": completed_runs,
                    },
                    campaign_manifest_path,
                )

            except Exception as exc:
                logger.error(f"Campaign run [{spec_idx}/{total_configs}] failed: {exc}")
                completed_runs[exp_key] = {
                    "experiment_id": item["experiment"]["name"],
                    "status": "FAILED",
                    "reason": str(exc),
                }

        # Generate Campaign Comparisons ONLY from SUCCESSFUL experiments
        cls.generate_campaign_comparisons(run_dir, completed_runs)

        campaign_status = {
            "status": "completed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "total_configs": total_configs,
            "completed_configs": len(completed_runs),
            "manifest_path": str(campaign_manifest_path),
        }

        logger.info("==================================================")
        logger.info(f"Experimental Campaign Completed for '{run_id}'! ({len(completed_runs)}/{total_configs} runs)")
        logger.info(f"Outputs isolated at: '{run_dir}'")
        logger.info("==================================================")

        return campaign_status

    @classmethod
    def generate_campaign_comparisons(cls, run_dir: Path, completed_runs: Dict[str, Any]) -> None:
        """
        Generates campaign-level summary.csv, comparison.csv, and dissertation tables ONLY from SUCCESSFUL experiments.
        """
        from response_repository import ResponseRepository
        from analysis import DissertationArtifactGenerator

        exp_base = run_dir / "experiments"
        if not exp_base.exists():
            logger.warning(f"No experiments folder found at {exp_base}")
            return

        all_records = ResponseRepository.load_all_responses(exp_base)

        if not all_records:
            logger.warning(f"No valid response records found across experiments in '{exp_base}'. Skipping campaign comparison files.")
            return

        df = ResponseRepository.to_dataframe(all_records)
        if df.empty:
            logger.warning("Empty response DataFrame across experiments. Skipping campaign comparison files.")
            return

        # 1. Export summary.csv
        summary_csv = run_dir / "summary.csv"
        df.to_csv(summary_csv, index=False, encoding="utf-8")
        logger.info(f"Exported campaign summary dataset ({len(df)} records) to '{summary_csv}'")

        # 2. Export task-aware comparison.csv
        from evaluation.registry import TaskMetricRegistry
        from evaluation import ClassificationEvaluator, QAEvaluator, GenerationEvaluator

        group_cols = ["task_type", "model_name", "num_examples"]
        rows = []

        for (t_type, m_name, k_shot), group in df.groupby(group_cols):
            preds = group["parsed_prediction"].fillna("").tolist() if "parsed_prediction" in group.columns else group["generated_text"].fillna("").tolist()
            refs = group["ground_truth"].fillna("").tolist()
            parse_succs = group["parse_success"].tolist() if "parse_success" in group.columns else [True] * len(group)
            req_succs = group["success"].tolist() if "success" in group.columns else [True] * len(group)

            # Calculate ONLY metrics applicable to this task_type over valid requests
            succ_group = group[group["success"] == True] if "success" in group.columns else group
            if not succ_group.empty:
                mean_lat = round(succ_group["latency_seconds"].mean(), 4) if "latency_seconds" in succ_group.columns else None
                mean_p_tok = round(succ_group["prompt_tokens"].mean(), 1) if "prompt_tokens" in succ_group.columns else None
                mean_c_tok = round(succ_group["completion_tokens"].mean(), 1) if "completion_tokens" in succ_group.columns else None
                mean_t_tok = round(succ_group["total_tokens"].mean(), 1) if "total_tokens" in succ_group.columns else None
            else:
                mean_lat = None
                mean_p_tok = None
                mean_c_tok = None
                mean_t_tok = None

            row = {
                "task_type": t_type,
                "model_name": m_name,
                "num_examples": k_shot,
                "sample_count": len(group),
                "request_success_rate": round(group["success"].mean(), 4) if "success" in group.columns else 1.0,
                "parse_success_rate": round(group["parse_success"].mean(), 4) if "parse_success" in group.columns else 1.0,
                "latency_seconds": mean_lat,
                "prompt_tokens": mean_p_tok,
                "completion_tokens": mean_c_tok,
                "total_tokens": mean_t_tok,
            }

            if t_type == "text_classification":
                perf = ClassificationEvaluator.evaluate(preds, refs, parse_successes=parse_succs, request_successes=req_succs)
                row["accuracy"] = perf.get("accuracy")
                row["f1"] = perf.get("f1_score")
                row["precision"] = perf.get("precision")
                row["recall"] = perf.get("recall")
            elif t_type == "question_answering":
                contexts = group["prompt_text"].tolist() if "prompt_text" in group.columns else None
                perf = QAEvaluator.evaluate(preds, refs, contexts=contexts, request_successes=req_succs)
                row["response_correctness"] = perf.get("response_correctness")
                row["response_completeness"] = perf.get("response_completeness")
                row["response_relevance"] = perf.get("response_relevance")
                row["exact_match"] = perf.get("qa_exact_match")
                row["f1"] = perf.get("qa_f1")
            elif t_type == "text_generation":
                perf = GenerationEvaluator.evaluate(preds, refs, request_successes=req_succs)
                row["bleu_1"] = perf.get("bleu_1")
                row["bleu_2"] = perf.get("bleu_2")
                row["bleu_4"] = perf.get("bleu_4")
                row["rouge_1"] = perf.get("rouge_1")
                row["rouge_2"] = perf.get("rouge_2")
                row["rouge_l"] = perf.get("rouge_l")
            else:
                perf = ClassificationEvaluator.evaluate(preds, refs, parse_successes=parse_succs, request_successes=req_succs)
                row["accuracy"] = perf.get("accuracy")
                row["f1"] = perf.get("f1_score")

            rows.append(row)

        comp_df = pd.DataFrame(rows)
        comp_csv = run_dir / "comparison.csv"
        comp_df.to_csv(comp_csv, index=False, encoding="utf-8")
        logger.info(f"Exported task-aware campaign comparison breakdown to '{comp_csv}'")

        # 3. Export Dissertation Tables
        try:
            dis_gen = DissertationArtifactGenerator(dissertation_dir=run_dir / "dissertation")
            dis_gen.generate_all_artifacts(responses_dir=exp_base)
            logger.info(f"Exported campaign dissertation artifacts to '{run_dir / 'dissertation'}'")
        except Exception as e:
            logger.warning(f"Could not generate dissertation artifacts for campaign: {e}")

    @classmethod
    def rebuild_campaign_from_responses(cls, run_dir: Path) -> Dict[str, Any]:
        """
        Rebuilds campaign summary.csv, comparison.csv, figures, and dissertation artifacts offline from stored response files.
        No LLM calls are made.
        """
        run_dir = Path(run_dir)
        manifest_file = run_dir / "campaign.json"
        completed_runs = {}
        if manifest_file.exists():
            data = load_json(manifest_file)
            completed_runs = data.get("completed_runs", {})

        cls.generate_campaign_comparisons(run_dir, completed_runs)
        return {"status": "rebuilt", "run_dir": str(run_dir)}
