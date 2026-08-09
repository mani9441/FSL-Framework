"""
Prompt Strategy Engine Package for FSL Research Framework.
Provides prompt models, templates, validation, generation, and few-shot context building.
"""

from prompt_engine.schema import PromptObject
from prompt_engine.templates import (
    InstructionTemplate,
    ExampleTemplate,
    HybridTemplate,
    DEFAULT_TASK_INSTRUCTIONS,
)
from prompt_engine.validator import PromptValidator
from prompt_engine.generator import PromptGenerator
from prompt_engine.few_shot import ExampleSelector, ContextBuilder

__all__ = [
    "PromptObject",
    "InstructionTemplate",
    "ExampleTemplate",
    "HybridTemplate",
    "DEFAULT_TASK_INSTRUCTIONS",
    "PromptValidator",
    "PromptGenerator",
    "ExampleSelector",
    "ContextBuilder",
]
