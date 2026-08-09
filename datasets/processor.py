"""
Dataset Processor for FSL Research Framework.
Handles text cleaning, record validation, HTML stripping, and missing value handling.
"""

import re
import html
from typing import List, Dict, Any, Tuple
from utilities.logger import get_logger

logger = get_logger("dataset_processor")


class DatasetProcessor:
    """Processor for cleaning and validating raw dataset records."""

    @staticmethod
    def clean_text(text: Any) -> str:
        """
        Cleans text string by stripping HTML tags, unescaping HTML entities,
        normalizing whitespace, and trimming trailing spaces.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # Unescape HTML entities (e.g., &amp; -> &)
        text = html.unescape(text)
        # Strip HTML tags (e.g., <p>text</p> -> text)
        text = re.sub(r"<[^>]+>", " ", text)
        # Replace non-breaking spaces and unicode whitespace
        text = re.sub(r"[\xa0\u200b\t\r\n]+", " ", text)
        # Normalize consecutive spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @classmethod
    def validate_record(cls, record: Dict[str, Any], task_type: str) -> bool:
        """
        Validates whether a raw record contains required non-empty information for the task.
        """
        if not isinstance(record, dict):
            return False

        # Task-specific minimum field requirement check
        if task_type == "text_classification":
            text = record.get("text") or record.get("content") or record.get("sentence") or record.get("input_text")
            label = record.get("label") if "label" in record else record.get("category")
            if text is None or label is None:
                return False
            if not str(text).strip():
                return False

        elif task_type == "question_answering":
            context = record.get("context") or record.get("passage")
            question = record.get("question")
            answer = record.get("answer") or record.get("answers")
            if context is None or question is None or answer is None:
                return False
            if not str(context).strip() or not str(question).strip():
                return False

        elif task_type == "text_generation":
            article = record.get("article") or record.get("document") or record.get("input_text") or record.get("prompt")
            answer = record.get("highlights") or record.get("summary") or record.get("answer") or record.get("completion")
            if article is None or answer is None:
                return False
            if not str(article).strip() or not str(answer).strip():
                return False

        return True

    @classmethod
    def process_records(
        cls, records: List[Dict[str, Any]], task_type: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Processes list of raw records: validates, cleans text fields, and drops invalid records.
        Returns tuple of (cleaned_records, dropped_count).
        """
        cleaned_records: List[Dict[str, Any]] = []
        dropped_count = 0

        for idx, raw in enumerate(records):
            if not cls.validate_record(raw, task_type):
                dropped_count += 1
                continue

            cleaned_rec: Dict[str, Any] = {}
            for key, val in raw.items():
                if isinstance(val, str):
                    cleaned_rec[key] = cls.clean_text(val)
                elif isinstance(val, list):
                    cleaned_rec[key] = [cls.clean_text(v) if isinstance(v, str) else v for v in val]
                else:
                    cleaned_rec[key] = val

            cleaned_records.append(cleaned_rec)

        logger.info(
            f"Processed {len(records)} records for task '{task_type}': "
            f"{len(cleaned_records)} valid, {dropped_count} dropped."
        )
        return cleaned_records, dropped_count
