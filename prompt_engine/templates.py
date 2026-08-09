"""
Prompt Templates for FSL Research Framework.
Provides Instruction, Example-Based, and Hybrid strategy template assemblers.
"""

from typing import List, Dict, Any
from datasets.schema import StandardRecord
from utilities.logger import get_logger

logger = get_logger("prompt_templates")

# Default task instructions if not custom specified in config
DEFAULT_TASK_INSTRUCTIONS: Dict[str, str] = {
    "text_classification": "Classify the input text into exactly one of the designated target categories: World, Sports, Business, Sci/Tech. Respond with ONLY the single chosen category label and nothing else.",
    "question_answering": "Answer the question accurately based on the provided reference context. Keep the answer concise.",
    "text_generation": "Summarize the given passage into concise highlight bullet points capturing main information.",
}


def format_single_demonstration(record: StandardRecord, max_demo_context_chars: int = 600) -> str:
    """Formats a single StandardRecord into a clean demonstration example block."""
    if record.task_type == "text_classification":
        return f"Input: {record.input_text}\nLabel: {record.answer}"

    elif record.task_type == "question_answering":
        if record.context:
            ctx = record.context
            if len(ctx) > max_demo_context_chars:
                ctx = ctx[:max_demo_context_chars].rsplit(" ", 1)[0] + "..."
            return f"Context: {ctx}\nQuestion: {record.question}\nAnswer: {record.answer}"
        return f"Question: {record.question}\nAnswer: {record.answer}"

    elif record.task_type == "text_generation":
        ctx = record.context or record.input_text
        if len(ctx) > max_demo_context_chars:
            ctx = ctx[:max_demo_context_chars].rsplit(" ", 1)[0] + "..."
        return f"Article: {ctx}\nSummary: {record.answer}"

    return f"Input: {record.input_text}\nOutput: {record.answer}"


def format_query_block(record: StandardRecord) -> str:
    """Formats the target evaluation record query block."""
    if record.task_type == "text_classification":
        return f"Input: {record.input_text}\nLabel:"

    elif record.task_type == "question_answering":
        if record.context:
            return f"Context: {record.context}\nQuestion: {record.question}\nAnswer:"
        return f"Question: {record.question}\nAnswer:"

    elif record.task_type == "text_generation":
        if record.context:
            return f"Article: {record.context}\nSummary:"
        return f"Input: {record.input_text}\nSummary:"

    return f"Input: {record.input_text}\nOutput:"


class BasePromptTemplate:
    """Base prompt template class."""

    @staticmethod
    def resolve_instruction(record: StandardRecord, custom_instruction: str = "") -> str:
        """Resolves task instruction string."""
        if custom_instruction and custom_instruction.strip():
            return custom_instruction.strip()
        return DEFAULT_TASK_INSTRUCTIONS.get(record.task_type, "Perform the requested task accurately.")


class InstructionTemplate(BasePromptTemplate):
    """
    Instruction Prompt Template (Zero-Shot Strategy).
    Format:
    [Instruction]

    [Query]
    """

    @classmethod
    def assemble(cls, query_record: StandardRecord, instruction_text: str = "") -> str:
        instruction = cls.resolve_instruction(query_record, instruction_text)
        query = format_query_block(query_record)
        return f"Instruction:\n{instruction}\n\n{query}"


class ExampleTemplate(BasePromptTemplate):
    """
    Example-Based Prompt Template (Pure Few-Shot Strategy).
    Format:
    [Demonstration 1]

    [Demonstration 2]

    [Query]
    """

    @classmethod
    def assemble(cls, query_record: StandardRecord, demonstrations: List[StandardRecord], max_demo_context_chars: int = 600) -> str:
        formatted_demos = [format_single_demonstration(rec, max_demo_context_chars=max_demo_context_chars) for rec in demonstrations]
        demos_block = "\n\n".join(formatted_demos)
        query = format_query_block(query_record)

        if demos_block:
            return f"{demos_block}\n\n{query}"
        return query


class StandardFewShotTemplate(BasePromptTemplate):
    """
    Standardized Few-Shot Prompt Template.
    Format:
    Instruction:
    [Instruction]

    [Examples:
    [Demonstration 1]

    [Demonstration 2]]

    Test Query:
    [Query]
    """

    @classmethod
    def assemble(
        cls,
        query_record: StandardRecord,
        demonstrations: List[StandardRecord],
        instruction_text: str = "",
        max_demo_context_chars: int = 600,
    ) -> str:
        instruction = cls.resolve_instruction(query_record, instruction_text)
        formatted_demos = [format_single_demonstration(rec, max_demo_context_chars=max_demo_context_chars) for rec in demonstrations]
        demos_block = "\n\n".join(formatted_demos)
        query = format_query_block(query_record)

        if demos_block:
            return f"Instruction:\n{instruction}\n\nExamples:\n{demos_block}\n\nTest Query:\n{query}"
        return f"Instruction:\n{instruction}\n\nTest Query:\n{query}"


class InstructionTemplate(BasePromptTemplate):
    """Instruction Prompt Template (Zero-Shot Strategy)."""

    @classmethod
    def assemble(cls, query_record: StandardRecord, instruction_text: str = "") -> str:
        return StandardFewShotTemplate.assemble(query_record, [], instruction_text)


class ExampleTemplate(BasePromptTemplate):
    """Example-Based Prompt Template."""

    @classmethod
    def assemble(cls, query_record: StandardRecord, demonstrations: List[StandardRecord]) -> str:
        return StandardFewShotTemplate.assemble(query_record, demonstrations, "")


class HybridTemplate(StandardFewShotTemplate):
    """Alias for StandardFewShotTemplate."""
    pass
