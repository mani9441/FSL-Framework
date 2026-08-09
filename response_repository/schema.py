"""
Response Repository Schema for FSL Research Framework.
Defines ResponseRecord dataclass encapsulating common core prompt/response metadata
and a task-specific evaluation metric layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class ResponseRecord:
    """
    Dataclass storing complete record for a generated LLM response.
    Encapsulates Common Core operational/experiment fields and Task-Specific evaluation metrics.
    """
    # Common Core Experiment/Inference Fields
    experiment_id: str
    trial_index: int
    record_id: str
    prompt_text: str
    generated_text: str
    ground_truth: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: str = ""
    provider: str = ""
    dataset_name: str = ""
    task_type: str = ""
    prompt_strategy: str = ""
    num_examples: int = 0
    ordering_strategy: str = ""
    diversity_strategy: str = ""
    
    # Token Accounting & Performance Operational Fields
    estimated_prompt_tokens: int = 0
    actual_prompt_tokens: int = 0
    prompt_tokens: int = 0  # Compatible alias field for actual_prompt_tokens
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    # Operational Success Indicators
    success: bool = True  # request_success
    parsed_prediction: str = ""
    parse_success: bool = False
    error_message: str = ""

    # Hardware / Provider Durations (Optional)
    total_duration_ns: Optional[int] = None
    load_duration_ns: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_duration_ns: Optional[int] = None

    # Config Snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Task-Specific Metric Layer (Dict containing metrics valid ONLY for this task)
    task_metrics: Dict[str, Any] = field(default_factory=dict)

    # Legacy field attributes retained as optional/nullable properties for task-specific access
    _is_correct: Optional[bool] = None
    _qa_exact_match: Optional[float] = None
    _qa_f1: Optional[float] = None
    _response_correctness: Optional[float] = None
    _response_completeness: Optional[float] = None
    _response_relevance: Optional[float] = None

    def __post_init__(self):
        if not self.actual_prompt_tokens and self.prompt_tokens:
            self.actual_prompt_tokens = self.prompt_tokens
        elif self.actual_prompt_tokens and not self.prompt_tokens:
            self.prompt_tokens = self.actual_prompt_tokens
        if not self.estimated_prompt_tokens and self.prompt_tokens:
            self.estimated_prompt_tokens = self.prompt_tokens

    @property
    def is_correct(self) -> Optional[bool]:
        """Classification binary correctness indicator (None for non-classification tasks)."""
        if self._is_correct is not None:
            return self._is_correct
        return self.task_metrics.get("is_correct")

    @is_correct.setter
    def is_correct(self, val: Optional[bool]) -> None:
        self._is_correct = val

    @property
    def qa_exact_match(self) -> Optional[float]:
        if self._qa_exact_match is not None:
            return self._qa_exact_match
        return self.task_metrics.get("qa_exact_match")

    @property
    def qa_f1(self) -> Optional[float]:
        if self._qa_f1 is not None:
            return self._qa_f1
        return self.task_metrics.get("qa_f1")

    @property
    def response_correctness(self) -> Optional[float]:
        if self._response_correctness is not None:
            return self._response_correctness
        return self.task_metrics.get("response_correctness")

    @property
    def response_completeness(self) -> Optional[float]:
        if self._response_completeness is not None:
            return self._response_completeness
        return self.task_metrics.get("response_completeness")

    @property
    def response_relevance(self) -> Optional[float]:
        if self._response_relevance is not None:
            return self._response_relevance
        return self.task_metrics.get("response_relevance")

    def to_dict(self) -> Dict[str, Any]:
        """Converts ResponseRecord to dictionary format, embedding common core and active task metrics."""
        d = {
            "experiment_id": self.experiment_id,
            "trial_index": self.trial_index,
            "record_id": self.record_id,
            "prompt_text": self.prompt_text,
            "generated_text": self.generated_text,
            "parsed_prediction": self.parsed_prediction,
            "parse_success": self.parse_success,
            "ground_truth": self.ground_truth,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "provider": self.provider,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "prompt_strategy": self.prompt_strategy,
            "num_examples": self.num_examples,
            "ordering_strategy": self.ordering_strategy,
            "diversity_strategy": self.diversity_strategy,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "actual_prompt_tokens": self.actual_prompt_tokens,
            "prompt_tokens": self.actual_prompt_tokens or self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "total_duration_ns": self.total_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "estimated_cost_usd": self.estimated_cost_usd,
            "success": self.success,
            "error_message": self.error_message,
            "config_snapshot": self.config_snapshot,
        }

        # Dynamically include task-specific metrics
        if self.task_metrics:
            for k, v in self.task_metrics.items():
                d[k] = v

        if self._is_correct is not None:
            d["is_correct"] = self._is_correct
        if self._qa_exact_match is not None:
            d["qa_exact_match"] = self._qa_exact_match
        if self._qa_f1 is not None:
            d["qa_f1"] = self._qa_f1
        if self._response_correctness is not None:
            d["response_correctness"] = self._response_correctness
        if self._response_completeness is not None:
            d["response_completeness"] = self._response_completeness
        if self._response_relevance is not None:
            d["response_relevance"] = self._response_relevance

        return d

    def to_flat_dict(self) -> Dict[str, Any]:
        """Converts ResponseRecord to a flattened dictionary suitable for CSV exports."""
        d = self.to_dict()
        d.pop("config_snapshot", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseRecord":
        """Instantiates ResponseRecord from dictionary."""
        p_tokens = data.get("actual_prompt_tokens", data.get("prompt_tokens", 0))
        est_p_tokens = data.get("estimated_prompt_tokens", p_tokens)

        task_metrics = data.get("task_metrics", {})
        for k in ["is_correct", "qa_exact_match", "qa_f1", "response_correctness",
                  "response_completeness", "response_relevance", "bleu_4", "rouge_1", "rouge_l"]:
            if k in data and k not in task_metrics:
                task_metrics[k] = data[k]

        return cls(
            experiment_id=data.get("experiment_id", ""),
            trial_index=data.get("trial_index", 0),
            record_id=data.get("record_id", ""),
            prompt_text=data.get("prompt_text", ""),
            generated_text=data.get("generated_text", ""),
            parsed_prediction=data.get("parsed_prediction", ""),
            parse_success=data.get("parse_success", False),
            ground_truth=data.get("ground_truth", ""),
            timestamp=data.get("timestamp", ""),
            model_name=data.get("model_name", ""),
            provider=data.get("provider", ""),
            dataset_name=data.get("dataset_name", ""),
            task_type=data.get("task_type", ""),
            prompt_strategy=data.get("prompt_strategy", ""),
            num_examples=data.get("num_examples", 0),
            ordering_strategy=data.get("ordering_strategy", ""),
            diversity_strategy=data.get("diversity_strategy", ""),
            estimated_prompt_tokens=est_p_tokens,
            actual_prompt_tokens=p_tokens,
            prompt_tokens=p_tokens,
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            latency_seconds=data.get("latency_seconds", 0.0),
            total_duration_ns=data.get("total_duration_ns"),
            load_duration_ns=data.get("load_duration_ns"),
            prompt_eval_duration_ns=data.get("prompt_eval_duration_ns"),
            eval_duration_ns=data.get("eval_duration_ns"),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
            config_snapshot=data.get("config_snapshot", {}),
            task_metrics=task_metrics,
            _is_correct=data.get("is_correct") if "is_correct" in data else None,
            _qa_exact_match=data.get("qa_exact_match") if "qa_exact_match" in data else None,
            _qa_f1=data.get("qa_f1") if "qa_f1" in data else None,
            _response_correctness=data.get("response_correctness") if "response_correctness" in data else None,
            _response_completeness=data.get("response_completeness") if "response_completeness" in data else None,
            _response_relevance=data.get("response_relevance") if "response_relevance" in data else None,
        )
