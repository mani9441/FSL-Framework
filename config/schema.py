"""
Configuration Schema and Data Models for FSL Research Framework.
Defines strongly-typed dataclasses for all 10 configuration items.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Dataset configuration item."""
    name: str
    task: str  # text_classification, question_answering, text_generation
    split: str = "test"
    sample_size: Optional[int] = 100


@dataclass
class PromptConfig:
    """Prompt engineering strategy configuration item."""
    strategy: str  # instruction, example_based, hybrid
    instruction_text: str = ""


@dataclass
class FewShotConfig:
    """Few-shot context configuration item."""
    num_examples: int = 3  # Few-shot size: 1, 3, 5
    ordering: str = "random"  # original, random, performance_based
    diversity: str = "medium"  # low, medium, high


@dataclass
class ModelConfig:
    """Language model configuration item."""
    name: str = "gpt-3.5-turbo"
    provider: str = "openai"
    temperature: float = 0.0  # Range: 0.0 - 2.0
    max_tokens: int = 512  # Range: 1 - 4096
    context_limit: int = 2048  # Declared input context token limit



@dataclass
class ExecutionConfig:
    """Execution and reproducibility configuration item."""
    id: Optional[str] = None
    name: str = "Baseline Few-Shot Evaluation"
    description: str = "Initial research trial"
    seed: int = 42
    repeated_trials: int = 3  # Trial count >= 1
    output_dir: str = "results"


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration item."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score",
        "latency_seconds", "token_usage"
    ])


@dataclass
class ExperimentConfig:
    """Top-level container for all experimental configuration items."""
    experiment: ExecutionConfig
    dataset: DatasetConfig
    prompt: PromptConfig
    few_shot: FewShotConfig
    model: ModelConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiates ExperimentConfig dataclass from raw dictionary."""
        exp_dict = data.get("experiment", {})
        ds_dict = data.get("dataset", {})
        prompt_dict = data.get("prompt", {})
        fs_dict = data.get("few_shot", {})
        model_dict = data.get("model", {})
        eval_dict = data.get("evaluation", {})

        return cls(
            experiment=ExecutionConfig(**exp_dict),
            dataset=DatasetConfig(**ds_dict),
            prompt=PromptConfig(**prompt_dict),
            few_shot=FewShotConfig(**fs_dict),
            model=ModelConfig(**model_dict),
            evaluation=EvaluationConfig(**eval_dict),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass back to nested dictionary."""
        return {
            "experiment": self.experiment.__dict__,
            "dataset": self.dataset.__dict__,
            "prompt": self.prompt.__dict__,
            "few_shot": self.few_shot.__dict__,
            "model": self.model.__dict__,
            "evaluation": self.evaluation.__dict__,
        }
