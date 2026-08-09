"""
Model Interfaces Package for FSL Research Framework.
Provides BaseModelWrapper, ModelResponse dataclass, cost estimation, OpenAI, Google Gemini, Groq, Hugging Face, and Ollama model wrappers, and ModelExecutionManager.
"""

from model_interfaces.schema import ModelResponse
from model_interfaces.base import BaseModelWrapper, estimate_cost
from model_interfaces.openai_wrapper import OpenAIModelWrapper
from model_interfaces.gemini_wrapper import GoogleGeminiModelWrapper
from model_interfaces.groq_wrapper import GroqModelWrapper
from model_interfaces.huggingface_wrapper import HuggingFaceModelWrapper
from model_interfaces.ollama_wrapper import OllamaModelWrapper
from model_interfaces.execution_manager import ModelExecutionManager

__all__ = [
    "ModelResponse",
    "BaseModelWrapper",
    "estimate_cost",
    "OpenAIModelWrapper",
    "GoogleGeminiModelWrapper",
    "GroqModelWrapper",
    "HuggingFaceModelWrapper",
    "OllamaModelWrapper",
    "ModelExecutionManager",
]
