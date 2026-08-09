"""
Local Ollama Model Wrapper for FSL Research Framework.

Provides HTTP API integration for local Ollama instances.
Designed for reproducible few-shot learning experiments across
multiple local LLMs using the Ollama REST API.
"""

import os
import time
import requests
from typing import Optional

from model_interfaces.base import BaseModelWrapper
from model_interfaces.schema import ModelResponse
from utilities.logger import get_logger


logger = get_logger("ollama_wrapper")


class OllamaModelWrapper(BaseModelWrapper):
    """Wrapper interfacing with local Ollama service."""

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        think: bool = False,
        seed: int = 42,
    ):
        super().__init__(
            model_name=model_name,
            provider="ollama",
        )

        self.host = os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434"
        ).rstrip("/")

        # Research configuration
        self.think = think
        self.seed = seed

        # Record the Ollama server version for reproducibility
        self.ollama_version = self._get_ollama_version()

        logger.info(
            f"Initialized Ollama wrapper: "
            f"model={self.model_name}, "
            f"think={self.think}, "
            f"seed={self.seed}, "
            f"ollama_version={self.ollama_version}"
        )

    def _get_ollama_version(self) -> Optional[str]:
        """Retrieve the running Ollama server version."""

        try:
            response = requests.get(
                f"{self.host}/api/version",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            return data.get("version")

        except Exception as e:
            logger.warning(
                f"Could not determine Ollama version: {e}"
            )
            return None

    def generate_response(
        self,
        prompt_text: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse:

        start_time = time.perf_counter()

        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt_text,
            "stream": False,

            # Explicitly control thinking for reproducibility.
            # False keeps the primary experiment focused on
            # few-shot prompting rather than reasoning-mode behavior.
            "think": self.think,

            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,

                # Fixed seed for reproducible inference.
                "seed": self.seed,
            },
        }

        try:
            res = requests.post(
                url,
                json=payload,
                timeout=120,
            )

            res.raise_for_status()

            data = res.json()

            latency = time.perf_counter() - start_time

            output_text = data.get(
                "response",
                ""
            ).strip()

            # ---------------------------------------------------------
            # Token counts
            # ---------------------------------------------------------
            #
            # Prefer Ollama's actual token counts.
            # Fall back to rough estimates only if unavailable.
            #

            prompt_tokens_raw = data.get(
                "prompt_eval_count"
            )

            completion_tokens_raw = data.get(
                "eval_count"
            )

            prompt_tokens = (
                prompt_tokens_raw
                if prompt_tokens_raw is not None
                else max(1, len(prompt_text) // 4)
            )

            completion_tokens = (
                completion_tokens_raw
                if completion_tokens_raw is not None
                else max(1, len(output_text) // 4)
            )

            total_tokens = (
                prompt_tokens +
                completion_tokens
            )

            # ---------------------------------------------------------
            # Ollama performance metrics
            # ---------------------------------------------------------

            total_dur = data.get(
                "total_duration"
            )

            load_dur = data.get(
                "load_duration"
            )

            prompt_eval_dur = data.get(
                "prompt_eval_duration"
            )

            eval_dur = data.get(
                "eval_duration"
            )

            # ---------------------------------------------------------
            # Logging
            # ---------------------------------------------------------

            logger.info(
                f"Ollama API call succeeded "
                f"(model={self.model_name}, "
                f"think={self.think}, "
                f"latency={latency:.2f}s)"
            )

            # ---------------------------------------------------------
            # Response
            # ---------------------------------------------------------

            return ModelResponse(
                generated_text=output_text,

                model_name=self.model_name,
                provider=self.provider,

                latency_seconds=round(
                    latency,
                    4
                ),

                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,

                estimated_cost_usd=0.0,

                success=True,

                total_duration_ns=total_dur,
                load_duration_ns=load_dur,
                prompt_eval_duration_ns=prompt_eval_dur,
                eval_duration_ns=eval_dur,

                raw_response=(
                    data
                    if isinstance(data, dict)
                    else {}
                ),
            )

        except Exception as e:

            latency = time.perf_counter() - start_time

            logger.error(
                f"Ollama API call failed "
                f"(model={self.model_name}): {e}"
            )

            return ModelResponse(
                generated_text="",

                model_name=self.model_name,
                provider=self.provider,

                latency_seconds=round(
                    latency,
                    4
                ),

                success=False,

                error_message=str(e),
            )
