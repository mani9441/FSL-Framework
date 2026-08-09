"""
Global Constants for Few-Shot Learning (FSL) Research Framework.
Contains directory paths, experimental variables, supported models, and default parameters.
"""

from pathlib import Path
from typing import List

# -----------------------------------------------------------------------------
# Base Directory Structure Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
BASE_DIR: Path = PROJECT_ROOT.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
RAW_DATASETS_DIR: Path = DATASETS_DIR / "raw"
PROCESSED_DATASETS_DIR: Path = DATASETS_DIR / "processed"
METADATA_DATASETS_DIR: Path = DATASETS_DIR / "metadata"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def create_run_directory(run_id: str = None) -> Path:
    """Creates a run-specific output directory under results/runs/run_<timestamp_or_id>/ containing experiments/."""
    if not run_id:
        from datetime import datetime
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    return run_dir

# Ensure essential output directories exist
ALL_DIRS: List[Path] = [
    CONFIG_DIR,
    RAW_DATASETS_DIR,
    PROCESSED_DATASETS_DIR,
    METADATA_DATASETS_DIR,
    RUNS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Supported NLP Tasks
# -----------------------------------------------------------------------------
TASK_TEXT_CLASSIFICATION: str = "text_classification"
TASK_QUESTION_ANSWERING: str = "question_answering"
TASK_TEXT_GENERATION: str = "text_generation"

SUPPORTED_TASKS: List[str] = [
    TASK_TEXT_CLASSIFICATION,
    TASK_QUESTION_ANSWERING,
    TASK_TEXT_GENERATION,
]

PROMPT_STRATEGY_STANDARD: str = "standard_few_shot"
PROMPT_STRATEGY_INSTRUCTION: str = "instruction"
PROMPT_STRATEGY_EXAMPLE: str = "example_based"
PROMPT_STRATEGY_HYBRID: str = "hybrid"

SUPPORTED_PROMPT_STRATEGIES: List[str] = [
    PROMPT_STRATEGY_STANDARD,
    "standard",
    PROMPT_STRATEGY_INSTRUCTION,
    PROMPT_STRATEGY_EXAMPLE,
    PROMPT_STRATEGY_HYBRID,
]

# -----------------------------------------------------------------------------
# Few-Shot Context Variables
# -----------------------------------------------------------------------------
SUPPORTED_SHOT_COUNTS: List[int] = [0, 1, 3, 5]

ORDERING_ORIGINAL: str = "original"
ORDERING_RANDOM: str = "random"
ORDERING_PERFORMANCE_BASED: str = "performance_based"

SUPPORTED_ORDERING_STRATEGIES: List[str] = [
    ORDERING_ORIGINAL,
    ORDERING_RANDOM,
    ORDERING_PERFORMANCE_BASED,
]

DIVERSITY_LOW: str = "low"
DIVERSITY_MEDIUM: str = "medium"
DIVERSITY_HIGH: str = "high"

SUPPORTED_DIVERSITY_LEVELS: List[str] = [
    DIVERSITY_LOW,
    DIVERSITY_MEDIUM,
    DIVERSITY_HIGH,
]

# -----------------------------------------------------------------------------
# Evaluation Metric Identifiers
# -----------------------------------------------------------------------------
METRIC_ACCURACY: str = "accuracy"
METRIC_PRECISION: str = "precision"
METRIC_RECALL: str = "recall"
METRIC_F1: str = "f1_score"

METRIC_CORRECTNESS: str = "response_correctness"
METRIC_COMPLETENESS: str = "completeness"
METRIC_RELEVANCE: str = "relevance"

METRIC_BLEU: str = "bleu"
METRIC_ROUGE: str = "rouge"

METRIC_LATENCY: str = "latency_seconds"
METRIC_TOKEN_USAGE: str = "token_usage"
METRIC_COST: str = "estimated_cost"

METRIC_CONSISTENCY: str = "output_consistency"
METRIC_VARIANCE: str = "output_variance"
METRIC_STABILITY: str = "trial_stability"

# -----------------------------------------------------------------------------
# Default Execution Parameters
# -----------------------------------------------------------------------------
DEFAULT_SEED: int = 42
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_NUM_TRIALS: int = 3
