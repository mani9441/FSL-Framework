"""
Unit tests for Phase 0 Research Resources Setup (Modules 0.1 - 0.7).
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path

from datasets.schema import StandardRecord, DatasetMetadata
from datasets.formatter import DatasetFormatter, AG_NEWS_LABEL_MAP
from resources.validator import ResourceValidator


class TestPhase0Resources(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ag_news_label_mapping(self):
        """Test AG News label integer mapping to official category strings."""
        self.assertEqual(AG_NEWS_LABEL_MAP[0], "World")
        self.assertEqual(AG_NEWS_LABEL_MAP[1], "Sports")
        self.assertEqual(AG_NEWS_LABEL_MAP[2], "Business")
        self.assertEqual(AG_NEWS_LABEL_MAP[3], "Sci/Tech")

    def test_standard_record_conversion(self):
        """Test conversion of raw record into unified StandardRecord schema."""
        raw = {
            "title": "Wall St Rallies",
            "text": "Stock indices rose after interest rate cut.",
            "label": 2,
        }

        record = DatasetFormatter.format_record(
            raw, task_type="text_classification", dataset_name="ag_news", index=0
        )
        self.assertEqual(record.label, "Business")
        self.assertEqual(record.answer, "Business")
        self.assertIn("Wall St Rallies", record.input_text)
        self.assertIn("Stock indices", record.input_text)

    def test_resource_validator_instantiation(self):
        """Test ResourceValidator running dataset checks."""
        validator = ResourceValidator(output_dir=self.temp_dir)
        report = validator.validate_datasets()
        self.assertIn("ag_news", report)
        self.assertIn("squad_v2", report)
        self.assertIn("cnn_dailymail", report)


if __name__ == "__main__":
    unittest.main()
