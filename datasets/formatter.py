"""
Dataset Formatter for FSL Research Framework.
Converts authentic dataset records into uniform StandardRecord instances.
"""

from typing import List, Dict, Any
from datasets.schema import StandardRecord
from utilities.logger import get_logger

logger = get_logger("dataset_formatter")

AG_NEWS_LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech",
    "World": "World",
    "Sports": "Sports",
    "Business": "Business",
    "Sci/Tech": "Sci/Tech",
    "Science/Technology": "Sci/Tech",
}


class DatasetFormatter:
    """Formats raw dataset representations into standardized records."""

    @classmethod
    def format_record(
        cls, record: Dict[str, Any], task_type: str, dataset_name: str, index: int
    ) -> StandardRecord:
        """
        Converts a single cleaned dictionary record into a StandardRecord instance.
        """
        rec_id = str(record.get("id", f"{dataset_name}_{index}"))

        if task_type == "text_classification":
            text = record.get("text") or record.get("content") or record.get("sentence") or record.get("input_text", "")
            title = record.get("title", "")
            if title and not text.startswith(title):
                input_text = f"{title}. {text}".strip()
            else:
                input_text = text.strip()

            raw_label = record.get("label") if "label" in record else record.get("category", "")
            label_str = AG_NEWS_LABEL_MAP.get(raw_label, str(raw_label))
            answer_str = label_str

            return StandardRecord(
                id=rec_id,
                task_type=task_type,
                input_text=input_text,
                question="",
                context="",
                answer=answer_str,
                label=label_str,
                metadata={"dataset_name": dataset_name, "raw_label": raw_label},
            )

        elif task_type == "question_answering":
            context = record.get("context") or record.get("passage", "")
            question = record.get("question", "")

            raw_answer = record.get("answer") or record.get("answers", "")
            if isinstance(raw_answer, dict) and "text" in raw_answer:
                texts = raw_answer["text"]
                answer_str = texts[0] if isinstance(texts, list) and texts else str(texts)
            elif isinstance(raw_answer, list):
                answer_str = str(raw_answer[0]) if raw_answer else ""
            else:
                answer_str = str(raw_answer)

            input_text = f"Context: {context}\nQuestion: {question}"

            return StandardRecord(
                id=rec_id,
                task_type=task_type,
                input_text=input_text,
                question=question,
                context=context,
                answer=answer_str,
                label="",
                metadata={"dataset_name": dataset_name},
            )

        elif task_type == "text_generation":
            context = record.get("article") or record.get("document") or record.get("context", "")
            input_text = record.get("input_text") or context
            answer_str = str(
                record.get("highlights") or record.get("summary") or record.get("answer") or record.get("completion", "")
            )

            return StandardRecord(
                id=rec_id,
                task_type=task_type,
                input_text=input_text,
                question="",
                context=context,
                answer=answer_str,
                label="",
                metadata={"dataset_name": dataset_name},
            )

        else:
            raise ValueError(f"Unsupported task_type '{task_type}' for formatting.")

    @classmethod
    def format_dataset(
        cls, records: List[Dict[str, Any]], task_type: str, dataset_name: str
    ) -> List[StandardRecord]:
        """Formats a list of dictionary records into standard records."""
        standard_records = [
            cls.format_record(rec, task_type, dataset_name, idx)
            for idx, rec in enumerate(records)
        ]
        logger.info(f"Formatted {len(standard_records)} records into StandardRecord objects.")
        return standard_records
