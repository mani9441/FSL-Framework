"""
Execution Manager & Metadata Recorder for FSL Research Framework.
Handles model instantiation across Google Gemini, Groq, Hugging Face, Ollama, and OpenAI, exponential backoff retries, timeouts, and metadata logging.
"""

import time
from typing import Dict, Any, Optional
from model_interfaces.schema import ModelResponse
from model_interfaces.base import BaseModelWrapper
from model_interfaces.openai_wrapper import OpenAIModelWrapper
from model_interfaces.gemini_wrapper import GoogleGeminiModelWrapper
from model_interfaces.huggingface_wrapper import HuggingFaceModelWrapper
from model_interfaces.groq_wrapper import GroqModelWrapper
from model_interfaces.ollama_wrapper import OllamaModelWrapper
from utilities.logger import get_logger

logger = get_logger("model_execution_manager")


class ModelExecutionManager:
    """Manager orchestrating model execution, retries, timeouts, and metadata recording."""

    def __init__(self, model_name: str = "gemini-2.5-flash", provider: Optional[str] = None):
        self.model_name = model_name
        self.provider = provider or self._infer_provider(model_name)
        self.wrapper: BaseModelWrapper = self._instantiate_wrapper(self.model_name, self.provider)

    @staticmethod
    def _infer_provider(model_name: str) -> str:
        """Infers provider from model name string."""
        lower = model_name.lower()
        ollama_tokens = ["llama3.1", "qwen3", "mistral:7b", "gemma3", "phi4", "ollama", ":8b", ":7b", ":4b"]
        if any(tok in lower for tok in ollama_tokens):
            return "ollama"
        elif "groq" in lower or "llama-3" in lower or "gemma2" in lower:
            return "groq"
        elif "gemini" in lower or "google" in lower:
            return "google"
        elif "qwen" in lower or "mistral" in lower or "hf" in lower:
            return "huggingface"
        elif "gpt" in lower or "openai" in lower:
            return "openai"
        return "google"


    @staticmethod
    def _instantiate_wrapper(model_name: str, provider: str) -> BaseModelWrapper:
        """Instantiates appropriate provider model wrapper."""
        if provider == "groq":
            return GroqModelWrapper(model_name=model_name)
        elif provider in ["google", "gemini"]:
            return GoogleGeminiModelWrapper(model_name=model_name)
        elif provider == "huggingface":
            return HuggingFaceModelWrapper(model_name=model_name)
        elif provider == "ollama":
            return OllamaModelWrapper(model_name=model_name)
        elif provider == "openai":
            return OpenAIModelWrapper(model_name=model_name)
        else:
            logger.warning(f"Unrecognized provider '{provider}'. Defaulting to Google Gemini wrapper.")
            return GoogleGeminiModelWrapper(model_name=model_name)

    def execute_with_retry(
        self,
        prompt_text: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> ModelResponse:
        """
        Executes model inference with exponential backoff retry loop.
        Captures latency, token usage, cost, and error metadata.
        """
        last_error = ""
        start_total = time.perf_counter()

        for attempt in range(retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retries} executing '{self.model_name}' ({self.provider})...")
                response = self.wrapper.generate_response(
                    prompt_text=prompt_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.success:
                    return response
                last_error = response.error_message
            except Exception as exc:
                last_error = str(exc)
                logger.warning(f"Inference attempt {attempt + 1} failed: {exc}")

            if attempt < retries - 1:
                sleep_time = backoff_factor ** (attempt + 1)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

        # Retries exhausted: return error ModelResponse
        total_latency = time.perf_counter() - start_total
        p_tokens = len(prompt_text) // 4
        logger.error(f"Execution failed after {retries} retries for '{self.model_name}': {last_error}")

        return ModelResponse(
            generated_text="",
            model_name=self.model_name,
            provider=self.provider,
            latency_seconds=round(total_latency, 4),
            prompt_tokens=p_tokens,
            completion_tokens=0,
            total_tokens=p_tokens,
            estimated_cost_usd=0.0,
            success=False,
            error_message=f"Failed after {retries} retries: {last_error}",
        )
