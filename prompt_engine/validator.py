"""
Prompt Validator for FSL Research Framework.
Checks prompt structural integrity, section completeness, formatting, and token length limits.
"""

from typing import Tuple, List
from prompt_engine.schema import PromptObject
from utilities.logger import get_logger

logger = get_logger("prompt_validator")


class PromptValidator:
    """Validator inspecting generated prompts for formatting, missing sections, and length."""

    @classmethod
    def validate(cls, prompt_obj: PromptObject, max_token_limit: int = 4096) -> Tuple[bool, List[str]]:
        """
        Inspects PromptObject and returns tuple: (is_valid, list_of_warnings).
        """
        warnings: List[str] = []
        is_valid = True

        prompt_text = prompt_obj.prompt_text
        strategy = prompt_obj.strategy

        # 1. Missing Sections Check
        if not prompt_text or not prompt_text.strip():
            warnings.append("Prompt text is empty.")
            return False, warnings

        if not prompt_obj.query or not prompt_obj.query.strip():
            warnings.append("Missing required target query section.")
            is_valid = False

        if strategy in ["instruction", "hybrid"]:
            if not prompt_obj.instruction or not prompt_obj.instruction.strip():
                warnings.append(f"Strategy '{strategy}' requires a non-empty instruction section.")
                is_valid = False

        if strategy in ["example_based", "hybrid"]:
            if not prompt_obj.demonstrations:
                warnings.append(f"Strategy '{strategy}' expected demonstration examples but received none.")

        # 2. Formatting & Delimiter Check
        if not (prompt_text.strip().endswith(":") or prompt_text.strip().endswith("Label:") or prompt_text.strip().endswith("Answer:") or prompt_text.strip().endswith("Summary:")):
            warnings.append("Prompt does not end with standard target response colon delimiter.")

        # 3. Token Length Check
        # Rough token estimate: 1 token ~ 4 characters
        est_tokens = len(prompt_text) // 4
        prompt_obj.estimated_tokens = est_tokens

        if est_tokens > max_token_limit:
            warnings.append(
                f"Prompt estimated token length ({est_tokens} tokens) exceeds model limit ({max_token_limit} tokens)."
            )
            is_valid = False

        if not is_valid:
            logger.warning(f"Prompt validation failed for strategy '{strategy}': {warnings}")
        else:
            logger.debug(f"Prompt validation passed successfully ({est_tokens} tokens).")

        return is_valid, warnings
