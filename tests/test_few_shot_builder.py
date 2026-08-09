"""
Unit tests for Module 4 — Few-Shot Context Builder.
"""

import unittest
from datasets.schema import StandardRecord
from prompt_engine.few_shot import ExampleSelector, ContextBuilder


class TestFewShotBuilder(unittest.TestCase):

    def setUp(self):
        self.query_rec = StandardRecord(
            id="q_1",
            task_type="text_classification",
            input_text="Global oil crude prices surge in market.",
            label="Business",
        )
        self.candidates = [
            StandardRecord(id="c_1", task_type="text_classification", input_text="Stock market index hits all time record high.", answer="Business", label="Business"),
            StandardRecord(id="c_2", task_type="text_classification", input_text="Spacecraft launches on expedition to Mars.", answer="Sci/Tech", label="Sci/Tech"),
            StandardRecord(id="c_3", task_type="text_classification", input_text="Football team advances to final round victory.", answer="Sports", label="Sports"),
            StandardRecord(id="c_4", task_type="text_classification", input_text="Oil crude barrel prices fluctuate sharply.", answer="Business", label="Business"),
            StandardRecord(id="c_5", task_type="text_classification", input_text="Diplomatic summit concludes with trade treaty.", answer="World", label="World"),
        ]

    def test_shot_count_selection(self):
        """Test selecting 1-shot, 3-shot, and 5-shot example counts."""
        for num_shot in [1, 3, 5]:
            selected = ExampleSelector.select_examples(
                self.candidates, self.query_rec, num_examples=num_shot, ordering="original", seed=42
            )
            self.assertEqual(len(selected), num_shot)

    def test_ordering_strategies(self):
        """Test original, random, and performance-based ordering strategies."""
        orig = ExampleSelector.select_examples(
            self.candidates, self.query_rec, num_examples=3, ordering="original"
        )
        self.assertEqual(orig[0].id, "c_1")

        perf = ExampleSelector.select_examples(
            self.candidates, self.query_rec, num_examples=5, ordering="performance_based"
        )
        # Most relevant (c_4: 'Oil crude barrel prices...') should be last in performance_based order
        self.assertEqual(perf[-1].id, "c_4")

    def test_diversity_strategies(self):
        """Test low, medium, and high diversity selection strategies."""
        low_div = ExampleSelector.select_examples(
            self.candidates, self.query_rec, num_examples=2, diversity="low", ordering="original"
        )
        # Low diversity should pick c_4 (highest oil/crude word overlap with query)
        self.assertEqual(low_div[0].id, "c_4")

        high_div = ExampleSelector.select_examples(
            self.candidates, self.query_rec, num_examples=3, diversity="high", ordering="original", seed=42
        )
        self.assertEqual(len(high_div), 3)

    def test_context_builder_rendering(self):
        """Test ContextBuilder rendering context text block."""
        demos = ContextBuilder.build_few_shot_context(
            self.candidates, self.query_rec, num_examples=2, ordering="original", seed=42
        )
        rendered = ContextBuilder.render_context_text(demos)
        self.assertIn("Input:", rendered)
        self.assertIn("Label:", rendered)


if __name__ == "__main__":
    unittest.main()
