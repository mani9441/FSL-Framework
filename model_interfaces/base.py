"""
Base Model Wrapper and Pricing Estimator for FSL Research Framework.
Establishes model interface contracts and automated cost calculation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from model_interfaces.schema import ModelResponse
from utilities.logger import get_logger

logger = get_logger("base_model_wrapper")

# Model pricing table per 1,000 tokens in USD (Input / Output)
MODEL_PRICING_USD: Dict[str, Tuple[float, float]] = {
    # OpenAI Models
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4": (0.0300, 0.0600),
    # Google Gemini Models
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro": (0.00125, 0.0050),
    "gemini-pro": (0.0005, 0.0015),
    # Anthropic Models
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-sonnet": (0.0030, 0.0150),
    # Hugging Face Open-Source / Local Models (Free)
    "huggingface": (0.0000, 0.0000),
    "mock": (0.0000, 0.0000),
}


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates estimated financial cost in USD based on input and output token counts.
    """
    clean_name = model_name.lower().strip()
    pricing = MODEL_PRICING_USD.get(clean_name, (0.0005, 0.0015))  # Default fallback rates

    input_cost = (prompt_tokens / 1000.0) * pricing[0]
    output_cost = (completion_tokens / 1000.0) * pricing[1]

    return round(input_cost + output_cost, 6)


class BaseModelWrapper(ABC):
    """Abstract base class for all language model providers."""

    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider

    @abstractmethod
    def generate_response(
        self, prompt_text: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> ModelResponse:
        """
        Executes model inference and returns a ModelResponse object.
        Must be implemented by concrete provider wrappers.
        """
        pass
