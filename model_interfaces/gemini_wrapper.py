"""
Google Gemini Model Wrapper for FSL Research Framework.
Provides production API integration for Google Gemini models using official google.genai SDK.
Model name is configured via GEMINI_MODEL env var as single source of truth.
"""

import os
import time
from typing import Dict, Any, Optional
from model_interfaces.base import BaseModelWrapper, estimate_cost
from model_interfaces.schema import ModelResponse
from utilities.logger import get_logger

logger = get_logger("gemini_wrapper")


class GoogleGeminiModelWrapper(BaseModelWrapper):
    """Wrapper interfacing directly with production Google Gemini Generative AI API."""

    def __init__(self, model_name: Optional[str] = None):
        target_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        super().__init__(model_name=target_model, provider="google")
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")

    def generate_response(
        self, prompt_text: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> ModelResponse:
        start_time = time.perf_counter()

        if not self.api_key:
            err_msg = "Missing GOOGLE_API_KEY environment variable. Cannot initialize production Gemini API."
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
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            latency = time.perf_counter() - start_time

            output_text = response.text if hasattr(response, "text") and response.text else ""
            p_tokens = getattr(response, "usage_metadata", None).prompt_token_count if hasattr(response, "usage_metadata") and response.usage_metadata else len(prompt_text) // 4
            c_tokens = getattr(response, "usage_metadata", None).candidates_token_count if hasattr(response, "usage_metadata") and response.usage_metadata else len(output_text) // 4 + 1
            t_tokens = p_tokens + c_tokens
            cost = estimate_cost(self.model_name, p_tokens, c_tokens)

            logger.info(f"Gemini API call succeeded ({self.model_name}, latency={latency:.2f}s).")
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
            logger.error(f"Gemini API call failed: {e}")
            return ModelResponse(
                generated_text="",
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=round(latency, 4),
                success=False,
                error_message=str(e),
            )
