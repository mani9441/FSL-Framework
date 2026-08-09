"""
Prompt Generator for FSL Research Framework.
Assembles, formats, and validates prompts according to configured prompt strategies.
"""

from typing import List, Optional
from datasets.schema import StandardRecord
from prompt_engine.schema import PromptObject
from prompt_engine.templates import (
    InstructionTemplate,
    ExampleTemplate,
    HybridTemplate,
    format_single_demonstration,
    format_query_block,
    BasePromptTemplate,
)
from prompt_engine.validator import PromptValidator
from utilities.logger import get_logger

logger = get_logger("prompt_generator")


class PromptGenerator:
    """Generator constructing validated PromptObject instances from records and configuration."""

    @classmethod
    def generate(
        cls,
        query_record: StandardRecord,
        demonstrations: Optional[List[StandardRecord]] = None,
        strategy: str = "standard_few_shot",
        instruction_text: str = "",
        max_token_limit: int = 4096,
    ) -> PromptObject:
        """
        Assembles a PromptObject for a query record based on the standardized prompt design,
        dynamically ensuring demonstration token budget fits within max_token_limit.
        """
        if demonstrations is None:
            demonstrations = []

        resolved_instruction = BasePromptTemplate.resolve_instruction(query_record, instruction_text)
        
        # Candidate demo context character budgets to attempt if prompt length exceeds max_token_limit
        demo_char_budgets = [600, 400, 200, 100]
        
        prompt_obj = None
        for demo_chars in demo_char_budgets:
            formatted_demos = [format_single_demonstration(rec, max_demo_context_chars=demo_chars) for rec in demonstrations]
            query_str = format_query_block(query_record)

            # Assemble prompt using standardized template
            from prompt_engine.templates import StandardFewShotTemplate
            prompt_text = StandardFewShotTemplate.assemble(
                query_record, demonstrations, resolved_instruction, max_demo_context_chars=demo_chars
            )

            # Build raw PromptObject
            prompt_obj = PromptObject(
                prompt_text=prompt_text,
                strategy=strategy,
                task_type=query_record.task_type,
                instruction=resolved_instruction,
                context=query_record.context,
                demonstrations=formatted_demos,
                query=query_str,
                estimated_tokens=len(prompt_text) // 4,
            )

            # Validate generated prompt
            is_valid, warnings = PromptValidator.validate(prompt_obj, max_token_limit=max_token_limit)
            prompt_obj.is_valid = is_valid
            prompt_obj.validation_warnings = warnings

            if is_valid or demo_chars == demo_char_budgets[-1]:
                break

        logger.debug(f"Generated {strategy} prompt for task '{query_record.task_type}' ({prompt_obj.estimated_tokens} est tokens, valid={prompt_obj.is_valid}).")
        return prompt_obj

