"""
Dataset Schema and Data Models for FSL Research Framework.
Defines StandardRecord and DatasetMetadata dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


@dataclass
class StandardRecord:
    """Standardized record format used across all NLP tasks."""
    id: str
    task_type: str  # text_classification, question_answering, text_generation
    input_text: str
    question: str = ""
    context: str = ""
    answer: str = ""
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts StandardRecord to dictionary representation."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "input_text": self.input_text,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardRecord":
        """Instantiates StandardRecord from dictionary."""
        return cls(
            id=str(data.get("id", "")),
            task_type=data.get("task_type", ""),
            input_text=data.get("input_text", ""),
            question=data.get("question", ""),
            context=data.get("context", ""),
            answer=data.get("answer", ""),
            label=str(data.get("label", "")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DatasetMetadata:
    """Metadata describing a processed dataset."""
    dataset_name: str
    task_type: str
    total_records: int
    splits: Dict[str, int] = field(default_factory=dict)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts DatasetMetadata to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "total_records": self.total_records,
            "splits": self.splits,
            "label_distribution": self.label_distribution,
            "created_at": self.created_at,
            "additional_info": self.additional_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Instantiates DatasetMetadata from dictionary."""
        return cls(
            dataset_name=data.get("dataset_name", ""),
            task_type=data.get("task_type", ""),
            total_records=data.get("total_records", 0),
            splits=data.get("splits", {}),
            label_distribution=data.get("label_distribution", {}),
            created_at=data.get("created_at", ""),
            additional_info=data.get("additional_info", {}),
        )
