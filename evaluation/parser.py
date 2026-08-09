"""
Prediction Parser Layer for FSL Research Framework.
Parses raw LLM generated text into standardized predicted labels/outputs.
"""

import re
from typing import Tuple, List, Optional
from utilities.logger import get_logger

logger = get_logger("prediction_parser")

AG_NEWS_LABEL_MAP = {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech",
    "world": "World",
    "sports": "Sports",
    "business": "Business",
    "sci/tech": "Sci/Tech",
    "scitech": "Sci/Tech",
    "science/technology": "Sci/Tech",
    "science and technology": "Sci/Tech",
    "sports news": "Sports",
    "business news": "Business",
    "world news": "World",
}

VALID_AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


class PredictionParser:
    """Parser layer converting raw LLM response strings into clean prediction outputs."""

    @classmethod
    def parse_classification(
        cls,
        generated_text: str,
        valid_labels: Optional[List[str]] = None,
        dataset_name: str = "ag_news",
    ) -> Tuple[str, bool]:
        """
        Parses classification LLM response.
        Returns Tuple of (parsed_prediction_label, parse_success).
        """
        if not generated_text or not generated_text.strip():
            return "", False

        text = generated_text.strip()

        # Clean markdown formatting like **Label:** or **Category:**
        clean_text = re.sub(r"\*\*.*?\*\*", "", text).strip()
        if not clean_text:
            clean_text = text

        # Strip common preambles like "Label:", "Category:", "Answer:", "Output:"
        clean_text = re.sub(r"^(label|category|class|answer|output)\s*:\s*", "", clean_text, flags=re.IGNORECASE).strip()

        # Normalize target labels list
        if valid_labels is None:
            valid_labels = VALID_AG_NEWS_LABELS

        # 1. Exact case-insensitive match
        for label in valid_labels:
            if clean_text.lower() == label.lower():
                return label, True

        # 2. Check mapped dictionary synonyms / numbers
        lower_clean = clean_text.lower()
        if lower_clean in AG_NEWS_LABEL_MAP:
            return AG_NEWS_LABEL_MAP[lower_clean], True

        # 3. Search for label substring presence in response
        # Prioritize exact word boundary matches
        found_matches: List[Tuple[int, str]] = []
        for label in valid_labels:
            pattern = r"\b" + re.escape(label.lower()) + r"\b"
            match = re.search(pattern, lower_clean)
            if match:
                found_matches.append((match.start(), label))

        if found_matches:
            # Pick first matching label by string position
            found_matches.sort(key=lambda x: x[0])
            return found_matches[0][1], True

        # Fallback substring search without word boundaries for Sci/Tech variations
        if "sci" in lower_clean or "tech" in lower_clean:
            return "Sci/Tech", True
        elif "sport" in lower_clean:
            return "Sports", True
        elif "busin" in lower_clean:
            return "Business", True
        elif "world" in lower_clean:
            return "World", True

        logger.warning(f"Failed to parse classification label from LLM output: '{generated_text}'")
        return generated_text.strip(), False

    @classmethod
    def parse_qa(cls, generated_text: str) -> Tuple[str, bool]:
        """
        Parses Question Answering LLM response by extracting concise answer spans
        and removing preambles while preserving the raw response string.
        """
        if not generated_text or not generated_text.strip():
            return "", False

        text = generated_text.strip()

        # Check for explicit preambles and extract answer span after preamble
        preamble_patterns = [
            r"^(?:the\s+)?correct\s+answer\s+(?:to\s+the\s+test\s+query\s+)?is\s*:\s*",
            r"^(?:the\s+)?answer\s+(?:to\s+the\s+(?:test\s+query|question)\s+)?is\s*:\s*",
            r"^so,?\s+the\s+answer\s+is\s*:\s*",
            r"^(?:according|based)\s+to\s+the\s+context,?\s*",
            r"^(?:answer|response|output|result)\s*:\s*",
        ]

        cleaned_text = text
        for pattern in preamble_patterns:
            match = re.search(pattern, cleaned_text, flags=re.IGNORECASE)
            if match:
                cleaned_text = cleaned_text[match.end():].strip()
                break

        # Remove leading/trailing bullet points or formatting
        cleaned_text = re.sub(r"^[*•\-\d+\.]\s*", "", cleaned_text).strip()

        # If multi-line response after preamble removal, extract first non-empty line as main answer span
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        extracted_answer = lines[0] if lines else cleaned_text

        # Strip remaining bold markdown tags if present around the extracted answer
        extracted_answer = re.sub(r"^\*\*(.*?)\*\*$", r"\1", extracted_answer).strip()

        return extracted_answer, bool(extracted_answer)

    @classmethod
    def parse_response(
        cls,
        generated_text: str,
        task_type: str,
        dataset_name: str = "ag_news",
    ) -> Tuple[str, bool]:
        """
        Master dispatcher for task-specific prediction parsing.
        """
        if task_type == "text_classification":
            return cls.parse_classification(generated_text, dataset_name=dataset_name)
        elif task_type == "question_answering":
            return cls.parse_qa(generated_text)
        elif task_type == "text_generation":
            cleaned = generated_text.strip()
            return cleaned, bool(cleaned)
        else:
            return generated_text.strip(), bool(generated_text.strip())
