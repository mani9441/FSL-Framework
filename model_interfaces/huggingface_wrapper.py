"""
Hugging Face Model Wrapper for FSL Research Framework.
Provides fast online serverless inference via huggingface_hub InferenceClient API.
Model name is configured via HF_MODEL env var as single source of truth.
"""

import os
import time
from typing import Dict, Any, Optional
from model_interfaces.base import BaseModelWrapper, estimate_cost
from model_interfaces.schema import ModelResponse
from utilities.logger import get_logger

logger = get_logger("huggingface_wrapper")


class HuggingFaceModelWrapper(BaseModelWrapper):
    """Wrapper interfacing with Hugging Face Serverless Inference API."""

    def __init__(self, model_name: Optional[str] = None):
        target_model = model_name or os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        super().__init__(model_name=target_model, provider="huggingface")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")

    def generate_response(
        self, prompt_text: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> ModelResponse:
        start_time = time.perf_counter()

        if not self.hf_token:
            err_msg = "Missing HF_TOKEN environment variable. Cannot initialize Hugging Face Inference API."
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
            from huggingface_hub import InferenceClient

            client = InferenceClient(api_key=self.hf_token)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            latency = time.perf_counter() - start_time
            res_text = completion.choices[0].message.content or ""

            p_tokens = len(prompt_text) // 4
            c_tokens = len(res_text) // 4 + 1
            t_tokens = p_tokens + c_tokens

            logger.info(f"Hugging Face Inference API succeeded ({self.model_name}, latency={latency:.2f}s).")
            return ModelResponse(
                generated_text=res_text.strip(),
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=round(latency, 4),
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                estimated_cost_usd=0.0,
                success=True,
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.error(f"Hugging Face API call failed for model '{self.model_name}': {e}")
            return ModelResponse(
                generated_text="",
                model_name=self.model_name,
                provider=self.provider,
                latency_seconds=round(latency, 4),
                success=False,
                error_message=str(e),
            )
