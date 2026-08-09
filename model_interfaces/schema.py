"""
Model Schema and Response Data Models for FSL Research Framework.
Defines ModelResponse dataclass capturing text, latency, token usage, cost, and execution status.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelResponse:
    """Dataclass capturing LLM inference output and metadata."""
    generated_text: str
    model_name: str
    provider: str  # openai, google, huggingface, mock
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    success: bool = True
    error_message: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)

    total_duration_ns: Optional[int] = None
    load_duration_ns: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_duration_ns: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts ModelResponse to dictionary format."""
        return {
            "generated_text": self.generated_text,
            "model_name": self.model_name,
            "provider": self.provider,
            "latency_seconds": self.latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "success": self.success,
            "error_message": self.error_message,
            "total_duration_ns": self.total_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResponse":
        """Instantiates ModelResponse from dictionary."""
        return cls(
            generated_text=data.get("generated_text", ""),
            model_name=data.get("model_name", ""),
            provider=data.get("provider", ""),
            latency_seconds=data.get("latency_seconds", 0.0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
            total_duration_ns=data.get("total_duration_ns"),
            load_duration_ns=data.get("load_duration_ns"),
            prompt_eval_duration_ns=data.get("prompt_eval_duration_ns"),
            eval_duration_ns=data.get("eval_duration_ns"),
            raw_response=data.get("raw_response", {}),
        )
