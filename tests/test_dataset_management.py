"""
Unit tests for Module 2 — Dataset Management.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from datasets.schema import StandardRecord, DatasetMetadata
from datasets.processor import DatasetProcessor
from datasets.formatter import DatasetFormatter
from datasets.sampler import DatasetSampler
from datasets.manager import DatasetManager


class TestDatasetManagement(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.proc_dir = Path(self.temp_dir) / "processed"
        self.meta_dir = Path(self.temp_dir) / "metadata"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_standard_record_serialization(self):
        """Test StandardRecord to_dict and from_dict roundtrip."""
        rec = StandardRecord(
            id="test_1",
            task_type="text_classification",
            input_text="Sample news headline text.",
            question="",
            context="",
            answer="Sports",
            label="Sports",
            metadata={"source": "test"},
        )
        rec_dict = rec.to_dict()
        reconstructed = StandardRecord.from_dict(rec_dict)

        self.assertEqual(reconstructed.id, "test_1")
        self.assertEqual(reconstructed.task_type, "text_classification")
        self.assertEqual(reconstructed.answer, "Sports")
        self.assertEqual(reconstructed.metadata["source"], "test")

    def test_dataset_processor_cleaning(self):
        """Test DatasetProcessor HTML tag stripping and whitespace normalization."""
        raw_text = "<p>  Fears over &amp; oil supplies hit stocks worldwide.  </p>"
        cleaned = DatasetProcessor.clean_text(raw_text)
        self.assertEqual(cleaned, "Fears over & oil supplies hit stocks worldwide.")

    def test_dataset_processor_validation(self):
        """Test DatasetProcessor record validation for missing fields."""
        valid_rec = {"text": "Valid headline text", "label": "Business"}
        invalid_rec = {"text": "   ", "label": "Business"}

        self.assertTrue(DatasetProcessor.validate_record(valid_rec, "text_classification"))
        self.assertFalse(DatasetProcessor.validate_record(invalid_rec, "text_classification"))

    def test_dataset_formatter_classification(self):
        """Test DatasetFormatter formatting classification record."""
        raw_rec = {"id": "cls_1", "text": "Stock market hits high.", "label": "Business"}
        std_rec = DatasetFormatter.format_record(raw_rec, "text_classification", "ag_news", 0)

        self.assertEqual(std_rec.id, "cls_1")
        self.assertEqual(std_rec.input_text, "Stock market hits high.")
        self.assertEqual(std_rec.answer, "Business")
        self.assertEqual(std_rec.label, "Business")

    def test_dataset_formatter_question_answering(self):
        """Test DatasetFormatter formatting QA record."""
        raw_rec = {
            "id": "qa_1",
            "context": "NASA launched Apollo 11 in 1969.",
            "question": "When was Apollo 11 launched?",
            "answers": {"text": ["1969"]},
        }
        std_rec = DatasetFormatter.format_record(raw_rec, "question_answering", "squad_v2", 0)

        self.assertEqual(std_rec.context, "NASA launched Apollo 11 in 1969.")
        self.assertEqual(std_rec.question, "When was Apollo 11 launched?")
        self.assertEqual(std_rec.answer, "1969")
        self.assertIn("Context: NASA launched Apollo 11 in 1969.", std_rec.input_text)

    def test_dataset_sampler(self):
        """Test DatasetSampler train/test split and demonstration pool sampling."""
        records = [
            StandardRecord(id=f"r_{i}", task_type="text_classification", input_text=f"Text {i}", label=f"L_{i%2}")
            for i in range(20)
        ]
        train, test = DatasetSampler.split_train_test(records, test_size=0.2, seed=42)
        self.assertEqual(len(train), 16)
        self.assertEqual(len(test), 4)

        demo_pool = DatasetSampler.sample_demonstration_pool(train, sample_size=6, seed=42)
        self.assertEqual(len(demo_pool), 6)

    def test_dataset_manager_pipeline(self):
        """Test DatasetManager end-to-end dataset preparation pipeline."""
        mgr = DatasetManager(
            dataset_name="sample_ag_news",
            task_type="text_classification",
            processed_dir=self.proc_dir,
            metadata_dir=self.meta_dir,
        )
        train, test, metadata = mgr.prepare_dataset(test_size=0.25, seed=42)

        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.dataset_name, "sample_ag_news")

        # Verify output files exist
        train_file = self.proc_dir / "sample_ag_news_text_classification_train.jsonl"
        metadata_file = self.meta_dir / "sample_ag_news_text_classification_metadata.json"
        self.assertTrue(train_file.exists())
        self.assertTrue(metadata_file.exists())

        # Test reload split
        loaded_test = mgr.load_processed_split("test")
        self.assertEqual(len(loaded_test), len(test))


if __name__ == "__main__":
    unittest.main()
