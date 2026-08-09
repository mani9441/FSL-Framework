"""
Unit tests for Module 7 — Response Repository.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from response_repository.schema import ResponseRecord
from response_repository.storage import ResponseRepository


class TestResponseRepository(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_record = ResponseRecord(
            experiment_id="EXP_TEST_REP_001",
            trial_index=0,
            record_id="rec_101",
            prompt_text="Instruction:\nClassify text.\n\nInput: Stock prices soar.\nLabel:",
            generated_text="Business",
            ground_truth="Business",
            model_name="gpt-3.5-turbo",
            provider="openai",
            dataset_name="ag_news",
            task_type="text_classification",
            prompt_strategy="hybrid",
            num_examples=3,
            ordering_strategy="random",
            diversity_strategy="medium",
            prompt_tokens=30,
            completion_tokens=2,
            total_tokens=32,
            latency_seconds=0.045,
            estimated_cost_usd=0.000018,
            success=True,
            config_snapshot={"test": "snapshot"},
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_response_record_serialization(self):
        """Test ResponseRecord serialization and flat dict conversion."""
        d = self.sample_record.to_dict()
        flat_d = self.sample_record.to_flat_dict()

        self.assertEqual(d["experiment_id"], "EXP_TEST_REP_001")
        self.assertIn("config_snapshot", d)
        self.assertNotIn("config_snapshot", flat_d)

        reconstructed = ResponseRecord.from_dict(d)
        self.assertEqual(reconstructed.record_id, "rec_101")
        self.assertEqual(reconstructed.total_tokens, 32)

    def test_save_and_load_trial_responses(self):
        """Test saving and loading trial responses in JSONL and CSV."""
        records = [self.sample_record]
        paths = ResponseRepository.save_trial_responses(
            experiment_id="EXP_TEST_REP_001",
            trial_index=0,
            records=records,
            output_dir=self.temp_dir,
        )

        self.assertTrue(paths["jsonl"].exists())
        self.assertTrue(paths["csv"].exists())

        loaded = ResponseRepository.load_trial_responses(self.temp_dir, trial_index=0)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].generated_text, "Business")

    def test_save_and_load_aggregate_responses(self):
        """Test saving aggregate responses and loading all responses."""
        rec2 = ResponseRecord(
            experiment_id="EXP_TEST_REP_001",
            trial_index=1,
            record_id="rec_102",
            prompt_text="Prompt 2",
            generated_text="Sci/Tech",
            ground_truth="Sci/Tech",
        )
        all_recs = [self.sample_record, rec2]

        paths = ResponseRepository.save_aggregate_responses(
            experiment_id="EXP_TEST_REP_001",
            all_records=all_recs,
            output_dir=self.temp_dir,
        )

        self.assertTrue(paths["jsonl"].exists())
        self.assertTrue(paths["csv"].exists())

        loaded = ResponseRepository.load_all_responses(self.temp_dir)
        self.assertEqual(len(loaded), 2)

    def test_to_dataframe(self):
        """Test converting records to pandas DataFrame."""
        df = ResponseRepository.to_dataframe([self.sample_record])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["generated_text"].iloc[0], "Business")
        self.assertEqual(df["total_tokens"].iloc[0], 32)


if __name__ == "__main__":
    unittest.main()
