"""
Prompt Schema and Data Models for FSL Research Framework.
Defines PromptObject dataclass representing generated prompts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PromptObject:
    """Dataclass encapsulating generated prompt components and validation state."""
    prompt_text: str
    strategy: str  # instruction, example_based, hybrid
    task_type: str  # text_classification, question_answering, text_generation
    instruction: str = ""
    context: str = ""
    demonstrations: List[str] = field(default_factory=list)
    query: str = ""
    estimated_tokens: int = 0
    is_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts PromptObject to dictionary."""
        return {
            "prompt_text": self.prompt_text,
            "strategy": self.strategy,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "context": self.context,
            "demonstrations": self.demonstrations,
            "query": self.query,
            "estimated_tokens": self.estimated_tokens,
            "is_valid": self.is_valid,
            "validation_warnings": self.validation_warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptObject":
        """Instantiates PromptObject from dictionary."""
        return cls(
            prompt_text=data.get("prompt_text", ""),
            strategy=data.get("strategy", ""),
            task_type=data.get("task_type", ""),
            instruction=data.get("instruction", ""),
            context=data.get("context", ""),
            demonstrations=data.get("demonstrations", []),
            query=data.get("query", ""),
            estimated_tokens=data.get("estimated_tokens", 0),
            is_valid=data.get("is_valid", True),
            validation_warnings=data.get("validation_warnings", []),
        )
