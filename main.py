"""
Main entry point for Few-Shot Learning (FSL) in LLMs Research Framework.
Supports experiment configuration loading, self-diagnostic verification, and modular pipeline invocation.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utilities import (
    get_logger,
    setup_logging,
    set_seed,
    save_json,
    load_json,
    Timer,
    LOGS_DIR,
)
from config import ConfigurationManager
from datasets import DatasetManager, StandardRecord
from prompt_engine import PromptGenerator, ContextBuilder
from model_interfaces import ModelExecutionManager
from experiment_engine import ExperimentController, ExperimentMatrixGenerator
from response_repository import ResponseRecord, ResponseRepository
from evaluation import EvaluationEngine
from analysis import StatisticalAnalysisEngine, SignificanceTester
from visualization import VisualizationEngine
from reports import ReportGenerationEngine
from pipeline import ReproducibleExperimentPipeline
from resources import ResourceValidator

logger = get_logger("main")


def run_diagnostics(config_path: Path = None) -> bool:
    """
    Executes Module 0 Project Foundation verification diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 0 Foundation Verification Diagnostics")
    logger.info("==================================================")

    try:
        config_mgr = ConfigurationManager(config_path)
        logger.info(f"Loaded config successfully. ID: '{config_mgr.experiment_id}', Hash: '{config_mgr.hash_digest}'")
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        return False

    logger.info("Step 2: Testing Random Seed Initialization...")
    try:
        seed = config_mgr.config.experiment.seed
        set_seed(seed)
        logger.info(f"Random seed initialized to {seed}.")
    except Exception as e:
        logger.error(f"Random seed initialization failed: {e}")
        return False

    logger.info("Step 3: Testing File I/O Helpers and Timer...")
    try:
        test_file = LOGS_DIR / "verification_test.json"
        test_payload = {
            "status": "success",
            "module": "Module 0 — Project Foundation",
            "config_id": config_mgr.experiment_id,
            "hash": config_mgr.hash_digest,
        }
        with Timer("JSON roundtrip test"):
            save_json(test_payload, test_file)
            loaded_payload = load_json(test_file)

        assert loaded_payload["status"] == "success"
        test_file.unlink(missing_ok=True)
        logger.info("File I/O helpers & timer operating correctly.")
    except Exception as e:
        logger.error(f"File I/O test failed: {e}")
        return False

    logger.info("==================================================")
    logger.info("Module 0 — Project Foundation Verification: PASSED")
    logger.info("==================================================")
    return True


def run_config_diagnostics(config_path: Path = None) -> bool:
    """
    Executes Module 1 Configuration Management diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 1 Configuration Management Diagnostics")
    logger.info("==================================================")

    sample_configs = [
        Path("config/experiments/exp_classification_baseline.yaml"),
        Path("config/experiments/exp_qa_baseline.yaml"),
        Path("config/experiments/exp_generation_baseline.yaml"),
    ]

    if config_path:
        sample_configs.append(config_path)

    for cfg_p in sample_configs:
        if not cfg_p.exists():
            continue
        logger.info(f"Testing configuration file: {cfg_p}")
        try:
            mgr = ConfigurationManager(cfg_p)
            paths = mgr.prepare_experiment_environment()
            logger.info(f"  ✓ Validated: ID={mgr.experiment_id}")
            logger.info(f"  ✓ Output paths created at: {paths['responses']}")
        except Exception as e:
            logger.error(f"  ✗ Configuration failure on {cfg_p}: {e}")
            return False

    logger.info("==================================================")
    logger.info("Module 1 — Configuration Management Verification: PASSED")
    logger.info("==================================================")
    return True


def run_dataset_diagnostics() -> bool:
    """
    Executes Module 2 Dataset Management diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 2 Dataset Management Diagnostics")
    logger.info("==================================================")

    benchmark_tasks = [
        ("ag_news", "text_classification"),
        ("squad_v2", "question_answering"),
        ("cnn_dailymail", "text_generation"),
    ]

    for name, task in benchmark_tasks:
        logger.info(f"Testing DatasetManager for dataset '{name}' ({task})...")
        try:
            ds_mgr = DatasetManager(dataset_name=name, task_type=task)
            train, test, metadata = ds_mgr.prepare_dataset(test_size=0.25, seed=42)
            logger.info(f"  ✓ Train size: {len(train)}, Test size: {len(test)}")
            logger.info(f"  ✓ Metadata generated: Total records = {metadata.total_records}")
        except Exception as e:
            logger.error(f"  ✗ Dataset preparation failure for '{name}': {e}")
            return False

    logger.info("==================================================")
    logger.info("Module 2 — Dataset Management Verification: PASSED")
    logger.info("==================================================")
    return True


def run_prompt_diagnostics() -> bool:
    """
    Executes Module 3 Prompt Strategy Engine diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 3 Prompt Strategy Engine Diagnostics")
    logger.info("==================================================")

    query_rec = StandardRecord(
        id="q_main",
        task_type="text_classification",
        input_text="Central banks adjust benchmark interest rates.",
        label="Business",
    )
    demos = [
        StandardRecord(id="d1", task_type="text_classification", input_text="Tech startup unveils AI model.", answer="Sci/Tech"),
        StandardRecord(id="d2", task_type="text_classification", input_text="Championship game ends in sudden death.", answer="Sports"),
    ]

    strategies = ["instruction", "example_based", "hybrid"]

    for strat in strategies:
        logger.info(f"Generating prompt for strategy '{strat}'...")
        try:
            p_obj = PromptGenerator.generate(query_rec, demos, strategy=strat)
            logger.info(f"  ✓ [{strat.upper()}] Valid={p_obj.is_valid}, Est Tokens={p_obj.estimated_tokens}")
        except Exception as e:
            logger.error(f"  ✗ Prompt generation failure for '{strat}': {e}")
            return False

    logger.info("==================================================")
    logger.info("Module 3 — Prompt Strategy Engine Verification: PASSED")
    logger.info("==================================================")
    return True


def run_few_shot_diagnostics() -> bool:
    """
    Executes Module 4 Few-Shot Context Builder diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 4 Few-Shot Context Builder Diagnostics")
    logger.info("==================================================")

    query = StandardRecord(
        id="q_oil",
        task_type="text_classification",
        input_text="Oil crude prices fluctuate after supply report.",
        label="Business",
    )
    candidates = [
        StandardRecord(id="c1", task_type="text_classification", input_text="Stock market reaches historic high.", answer="Business", label="Business"),
        StandardRecord(id="c2", task_type="text_classification", input_text="New satellite launched into Earth orbit.", answer="Sci/Tech", label="Sci/Tech"),
        StandardRecord(id="c3", task_type="text_classification", input_text="National football league announces playoffs.", answer="Sports", label="Sports"),
        StandardRecord(id="c4", task_type="text_classification", input_text="Oil barrel production increases in Gulf.", answer="Business", label="Business"),
        StandardRecord(id="c5", task_type="text_classification", input_text="Global diplomatic summit reaches agreement.", answer="World", label="World"),
    ]

    shot_counts = [1, 3, 5]
    diversity_levels = ["low", "medium", "high"]
    ordering_strategies = ["original", "random", "performance_based"]

    for shot in shot_counts:
        for div in diversity_levels:
            for ord_strat in ordering_strategies:
                try:
                    demos = ContextBuilder.build_few_shot_context(
                        candidate_pool=candidates,
                        query_record=query,
                        num_examples=shot,
                        diversity=div,
                        ordering=ord_strat,
                        seed=42,
                    )
                    logger.info(f"  ✓ {shot}-shot | Diversity='{div}' | Ordering='{ord_strat}' => Selected {len(demos)} demos")
                except Exception as e:
                    logger.error(f"  ✗ Few-Shot Context Builder failed for {shot}-shot/{div}/{ord_strat}: {e}")
                    return False

    logger.info("==================================================")
    logger.info("Module 4 — Few-Shot Context Builder Verification: PASSED")
    logger.info("==================================================")
    return True


def run_model_diagnostics() -> bool:
    """
    Executes Module 5 Model Interface Layer diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 5 Model Interface Layer Diagnostics")
    logger.info("==================================================")

    test_models = [
        ("llama-3.1-8b-instant", "groq"),
        ("Qwen/Qwen2.5-7B-Instruct", "huggingface"),
        ("gemini-3.5-flash", "google"),
    ]

    for model_name, provider in test_models:
        logger.info(f"Testing ModelExecutionManager for '{model_name}' ({provider})...")
        try:
            exec_mgr = ModelExecutionManager(model_name=model_name, provider=provider)
            logger.info(f"  ✓ [{provider.upper()}] Initialized model wrapper successfully.")
        except Exception as e:
            logger.error(f"  ✗ Model interface initialization failed for '{model_name}': {e}")
            return False

    logger.info("==================================================")
    logger.info("Module 5 — Model Interface Layer Verification: PASSED")
    logger.info("==================================================")
    return True


def run_engine_diagnostics(config_path: Path = None) -> bool:
    """
    Executes Module 6 Experiment Execution Engine diagnostics.
    """
    logger.info("==================================================")
    logger.info("Starting Module 6 Experiment Execution Engine Diagnostics")
    logger.info("==================================================")

    cfg = config_path or Path("config/experiments/exp_classification_baseline.yaml")
    logger.info(f"Executing dry-run experiment using config: {cfg}")
    try:
        controller = ExperimentController(config_source=cfg)
        summary = controller.run_experiment(sample_limit=2)
        logger.info(f"  ✓ Experiment ID: '{summary['experiment_id']}'")
        logger.info(f"  ✓ Status: '{summary['status']}' across {summary['total_trials']} trials.")
    except Exception as e:
        logger.error(f"  ✗ Experiment execution engine failed: {e}")
        return False

    logger.info("==================================================")
    logger.info("Module 6 — Experiment Execution Engine Verification: PASSED")
    logger.info("==================================================")
    return True


def run_system_verification() -> bool:
    """
    Executes Phase A — System Verification run:
    Runs a 20-sample classification experiment using Groq Llama-3.1-8B (or Gemini) to verify dataset loading,
    prompt generation, API inference, response storage, evaluation, statistical analysis, visualization, and report generation.
    """
    logger.info("==================================================")
    logger.info("Starting Phase A — System Verification Run (20 Samples)")
    logger.info("==================================================")

    verification_spec = {
        "experiment": {
            "name": "Phase A - System Verification Run",
            "description": "Verification of full pipeline behavior on 20 AG News samples",
            "seed": 42,
            "repeated_trials": 1,
        },
        "dataset": {
            "name": "ag_news",
            "task": "text_classification",
            "split": "test",
            "sample_size": 20,
        },
        "prompt": {
            "strategy": "instruction",
            "instruction_text": "Classify news headline into Business, Sci/Tech, Sports, or World.",
        },
        "few_shot": {
            "num_examples": 1,
            "ordering": "random",
            "diversity": "medium",
        },
        "model": {
            "name": "llama-3.1-8b-instant",
            "provider": "groq",
            "temperature": 0.0,
            "max_tokens": 64,
        },
        "evaluation": {
            "metrics": ["accuracy", "f1"],
        },
    }

    try:
        pipeline = ReproducibleExperimentPipeline(config_source=verification_spec)
        summary = pipeline.run_pipeline(sample_limit=20)
        logger.info("==================================================")
        logger.info("Phase A — System Verification Completed Successfully!")
        logger.info(f"  - Experiment ID: {summary['experiment_id']}")
        logger.info(f"  - Research Report: {summary['output_paths']['reports']}/research_report.md")
        logger.info(f"  - HTML Report: {summary['output_paths']['reports']}/research_report.html")
        logger.info("==================================================")
        return True
    except Exception as e:
        logger.error(f"Phase A System Verification failed: {e}")
        return False


def run_resource_validation() -> bool:
    """
    Executes Phase 0 Resource Validation for authentic datasets and production LLMs.
    """
    validator = ResourceValidator()
    return validator.run_full_validation()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-Shot Learning Research Framework Entry Point"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--validate-resources",
        action="store_true",
        help="Execute Phase 0 validation on authentic benchmark datasets and LLM providers",
    )
    parser.add_argument(
        "--verify-system",
        action="store_true",
        help="Execute Phase A — System Verification run (20 samples, full pipeline test)",
    )
    parser.add_argument(
        "--run-campaign",
        action="store_true",
        help="Execute Phase B & E Experimental Campaign (multi-factorial matrix with checkpointing)",
    )
    parser.add_argument(
        "--run-experiment",
        action="store_true",
        help="Execute complete experimental pipeline using specified --config",
    )
    parser.add_argument(
        "--check-reproducibility",
        action="store_true",
        help="Execute dual-run reproducibility check verifying bit-for-bit output match",
    )
    parser.add_argument(
        "--verify-pipeline",
        action="store_true",
        help="Run self-diagnostic verification of Module 12 Integration & Reproducibility Pipeline",
    )
    parser.add_argument(
        "--generate-dissertation-outputs",
        action="store_true",
        help="Generate Chapters 4 & 5 dissertation tables (LaTeX, CSV), figures, and research observation report",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="Filter campaign run by LLM provider (e.g. groq, google, huggingface, ollama)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Filter campaign run by model name (e.g. llama-3.1-8b-instant, gemini-2.0-flash)",
    )
    parser.add_argument(
        "--shots",
        "-s",
        type=int,
        default=None,
        help="Filter campaign run by shot count (0, 1, 3, 5)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="Filter campaign run by task (text_classification, question_answering, text_generation)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Evaluation sample size for campaign runs (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = 10 if args.verbose else 20
    setup_logging(console_level=log_level)

    if args.validate_resources:
        success = run_resource_validation()
        sys.exit(0 if success else 1)

    if args.verify_system:
        success = run_system_verification()
        sys.exit(0 if success else 1)

    if getattr(args, "generate_dissertation_outputs", False):
        from analysis import DissertationArtifactGenerator
        gen = DissertationArtifactGenerator()
        res = gen.generate_all_artifacts()
        logger.info(f"Dissertation artifacts generation completed: {res}")
        sys.exit(0)

    if args.run_campaign or args.provider or args.model or args.shots or args.task:
        summary = ExperimentMatrixGenerator.run_campaign(
            sample_size=args.sample_size,
            provider_filter=args.provider,
            model_filter=args.model,
            shot_filter=args.shots,
            task_filter=args.task,
        )
        logger.info(f"Campaign execution completed: {summary}")
        sys.exit(0)

    if args.verify_pipeline:
        config_path = Path(args.config) if args.config else None
        success = ReproducibleExperimentPipeline.verify_reproducibility(config_path)
        sys.exit(0 if success else 1)

    if args.check_reproducibility:
        config_path = Path(args.config) if args.config else None
        success = ReproducibleExperimentPipeline.verify_reproducibility(config_path)
        sys.exit(0 if success else 1)

    if args.run_experiment:
        config_path = Path(args.config) if args.config else None
        pipeline = ReproducibleExperimentPipeline(config_source=config_path)
        summary = pipeline.run_pipeline()

        logger.info(
            f"Full End-to-End Pipeline Execution Completed for '{summary['experiment_id']}'.\n"
            f"  - Responses: {summary['output_paths']['responses']}\n"
            f"  - Metrics: {summary['output_paths']['metrics']}\n"
            f"  - Statistics: {summary['output_paths']['statistics']}\n"
            f"  - Figures: {summary['output_paths']['figures']}\n"
            f"  - Research Reports: {summary['output_paths']['reports']}\n"
            f"  - Reproducibility Manifest: {summary['manifest_path']}"
        )
        sys.exit(0)

    config_path = Path(args.config) if args.config else None
    mgr = ConfigurationManager(config_path)
    logger.info(f"Loaded configuration for Experiment ID '{mgr.experiment_id}'")
    logger.info("Framework loaded. Pass '--verify-system' for Phase A verification or '--run-campaign' for Phase B campaign.")


if __name__ == "__main__":
    main()
