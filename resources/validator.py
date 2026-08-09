"""
Resource Validator for Phase 0 Research Resources Setup.
Automates validation of authentic benchmark datasets and production LLM API providers before experiment execution.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import requests

from datasets import DatasetManager, StandardRecord
from model_interfaces import ModelExecutionManager
from utilities.helpers import save_json
from utilities.constants import LOGS_DIR
from utilities.logger import get_logger

logger = get_logger("resource_validator")


class ResourceValidator:
    """Validator performing dataset completeness checks and LLM API connectivity tests."""

    def __init__(self, output_dir: Path = LOGS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_datasets(self) -> Dict[str, Any]:
        """
        Validates authentic benchmark datasets: AG News, SQuAD v2, CNN/DailyMail.
        """
        logger.info("==================================================")
        logger.info("Validating Authentic Benchmark Datasets")
        logger.info("==================================================")

        benchmark_configs = [
            ("ag_news", "text_classification"),
            ("squad_v2", "question_answering"),
            ("cnn_dailymail", "text_generation"),
        ]

        dataset_report = {}
        for ds_name, task in benchmark_configs:
            logger.info(f"Checking dataset '{ds_name}' ({task})...")
            try:
                mgr = DatasetManager(dataset_name=ds_name, task_type=task)
                train, test, metadata = mgr.prepare_dataset(test_size=0.2, seed=42)

                valid = metadata.total_records > 0 and (len(train) + len(test) > 0)
                dataset_report[ds_name] = {
                    "task": task,
                    "valid": valid,
                    "total_records": metadata.total_records,
                    "train_count": len(train),
                    "test_count": len(test),
                    "sample_input": test[0].input_text[:100] if test else "",
                }
                logger.info(f"  ✓ [{ds_name}] Valid={valid}, Total Records={metadata.total_records}")
            except Exception as e:
                logger.error(f"  ✗ [{ds_name}] Dataset validation failed: {e}")
                dataset_report[ds_name] = {
                    "task": task,
                    "valid": False,
                    "error": str(e),
                }

        return dataset_report

    def validate_models(self) -> Dict[str, Any]:
        """
        Validates API reachability and authentication for real LLM providers: Google, Groq, Hugging Face, Ollama.
        """
        logger.info("==================================================")
        logger.info("Validating Production LLM API Providers")
        logger.info("==================================================")

        providers = [
            ("google", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), "GOOGLE_API_KEY"),
            ("groq", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), "GROQ_API_KEY"),
            ("huggingface", os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"), "HF_TOKEN"),
            ("ollama", os.getenv("OLLAMA_MODEL", "llama3"), "OLLAMA_HOST"),
        ]

        model_report = {}
        prompt_test = "Classify headline into Business or World.\nInput: Global stock market rises.\nLabel:"

        for provider, model_name, env_key in providers:
            logger.info(f"Testing Provider '{provider}' (Model: '{model_name}')...")
            key_val = os.getenv(env_key, "")

            if provider == "ollama":
                try:
                    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
                    res = requests.get(f"{host}/api/tags", timeout=3)
                    reachable = res.status_code == 200
                except Exception:
                    reachable = False

                model_report[provider] = {
                    "model_name": model_name,
                    "api_key_configured": True,
                    "reachable": reachable,
                    "test_inference": "available" if reachable else "offline (start ollama serve)",
                }
                logger.info(f"  ✓ [OLLAMA] Host={host}, Reachable={reachable}")
                continue

            if not key_val or key_val.startswith("your_"):
                logger.warning(f"  ! [{provider.upper()}] API Key '{env_key}' not configured in .env")
                model_report[provider] = {
                    "model_name": model_name,
                    "api_key_configured": False,
                    "reachable": False,
                    "test_inference": "unconfigured API key",
                }
                continue

            try:
                mgr = ModelExecutionManager(model_name=model_name, provider=provider)
                resp = mgr.execute_with_retry(prompt_text=prompt_test, max_tokens=10, retries=1)
                model_report[provider] = {
                    "model_name": model_name,
                    "api_key_configured": True,
                    "reachable": resp.success,
                    "latency_seconds": resp.latency_seconds,
                    "generated_output": resp.generated_text[:50] if resp.success else "",
                    "error": resp.error_message if not resp.success else None,
                }
                if resp.success:
                    logger.info(f"  ✓ [{provider.upper()}] Reachable & Test Inference Success! Latency={resp.latency_seconds}s")
                else:
                    logger.warning(f"  ! [{provider.upper()}] API Key set but test call returned error: {resp.error_message}")
            except Exception as e:
                logger.error(f"  ✗ [{provider.upper()}] Test failed: {e}")
                model_report[provider] = {
                    "model_name": model_name,
                    "api_key_configured": True,
                    "reachable": False,
                    "error": str(e),
                }

        return model_report

    def run_full_validation(self) -> bool:
        """Runs dataset and LLM model provider validation and exports report."""
        ds_report = self.validate_datasets()
        m_report = self.validate_models()

        all_ds_valid = all(info.get("valid", False) for info in ds_report.values())
        configured_models = sum(1 for info in m_report.values() if info.get("api_key_configured"))

        report_payload = {
            "status": "passed" if all_ds_valid else "partially_valid",
            "datasets": ds_report,
            "models": m_report,
            "summary": {
                "all_datasets_valid": all_ds_valid,
                "configured_model_providers": configured_models,
            },
        }

        report_file = self.output_dir / "resource_validation_report.json"
        save_json(report_payload, report_file)

        logger.info("==================================================")
        logger.info(f"Phase 0 Resource Validation Summary: {report_payload['status'].upper()}")
        logger.info(f"Saved Validation Report to '{report_file}'")
        logger.info("==================================================")

        return all_ds_valid
