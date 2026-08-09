"""
Groq Model Wrapper for FSL Research Framework.
Provides API integration for ultra-fast Llama 3.1 8B, Llama 3.3 70B, and Gemma 2 models via Groq API.
Model name is configured via GROQ_MODEL env var as single source of truth.
"""

import os
import time
from typing import Dict, Any, Optional
from model_interfaces.base import BaseModelWrapper, estimate_cost
from model_interfaces.schema import ModelResponse
from utilities.logger import get_logger

logger = get_logger("groq_wrapper")


class GroqModelWrapper(BaseModelWrapper):
    """Wrapper interfacing directly with production Groq API."""

    def __init__(self, model_name: Optional[str] = None):
        target_model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        super().__init__(model_name=target_model, provider="groq")
        self.api_key = os.getenv("GROQ_API_KEY", "")

    def generate_response(
        self, prompt_text: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> ModelResponse:
        start_time = time.perf_counter()

        if not self.api_key:
            err_msg = "Missing GROQ_API_KEY environment variable. Cannot initialize Groq API wrapper."
            logger.error(err_msg)
            return ModelResponse(
                generated_text="",
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=0.0,
                success=False,
                error_message=err_msg,
            )

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - start_time

            output_text = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            p_tokens = getattr(usage, "prompt_tokens", len(prompt_text) // 4) if usage else len(prompt_text) // 4
            c_tokens = getattr(usage, "completion_tokens", len(output_text) // 4 + 1) if usage else len(output_text) // 4 + 1
            t_tokens = p_tokens + c_tokens
            cost = estimate_cost(self.model_name, p_tokens, c_tokens)

            logger.info(f"Groq API call succeeded ({self.model_name}, latency={latency:.2f}s).")
            return ModelResponse(
                generated_text=output_text.strip(),
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=round(latency, 4),
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                estimated_cost_usd=cost,
                success=True,
            )

        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.error(f"Groq API call failed: {e}")
            return ModelResponse(
                generated_text="",
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=round(latency, 4),
                success=False,
                error_message=str(e),
            )
