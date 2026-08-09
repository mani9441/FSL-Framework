"""
OpenAI Model Wrapper for FSL Research Framework.
Provides direct production API integration for OpenAI models.
"""

import os
import time
from typing import Dict, Any
from model_interfaces.base import BaseModelWrapper, estimate_cost
from model_interfaces.schema import ModelResponse
from utilities.logger import get_logger

logger = get_logger("openai_wrapper")


class OpenAIModelWrapper(BaseModelWrapper):
    """Wrapper interfacing with production OpenAI Chat Completions API."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name=model_name, provider="openai")
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def generate_response(
        self, prompt_text: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> ModelResponse:
        start_time = time.perf_counter()

        if not self.api_key:
            err_msg = "Missing OPENAI_API_KEY environment variable. Cannot initialize OpenAI API."
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
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - start_time

            output_text = response.choices[0].message.content or ""
            p_tokens = response.usage.prompt_tokens if response.usage else len(prompt_text) // 4
            c_tokens = response.usage.completion_tokens if response.usage else len(output_text) // 4
            t_tokens = p_tokens + c_tokens
            cost = estimate_cost(self.model_name, p_tokens, c_tokens)

            logger.info(f"OpenAI API call succeeded ({self.model_name}, latency={latency:.2f}s, tokens={t_tokens}).")
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
                raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.error(f"OpenAI API call failed: {e}")
            return ModelResponse(
                generated_text="",
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=round(latency, 4),
                success=False,
                error_message=str(e),
            )
