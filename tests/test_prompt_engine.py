"""
Unit tests for Module 3 — Prompt Strategy Engine.
"""

import unittest
from datasets.schema import StandardRecord
from prompt_engine.schema import PromptObject
from prompt_engine.templates import InstructionTemplate, ExampleTemplate, HybridTemplate
from prompt_engine.validator import PromptValidator
from prompt_engine.generator import PromptGenerator


class TestPromptEngine(unittest.TestCase):

    def setUp(self):
        self.cls_query = StandardRecord(
            id="q_1",
            task_type="text_classification",
            input_text="Tech giant reveals quantum processor.",
            label="Sci/Tech",
        )
        self.cls_demos = [
            StandardRecord(id="d_1", task_type="text_classification", input_text="Oil prices surge globally.", answer="Business"),
            StandardRecord(id="d_2", task_type="text_classification", input_text="Local team wins championship.", answer="Sports"),
        ]

        self.qa_query = StandardRecord(
            id="q_2",
            task_type="question_answering",
            input_text="Context: Mars is the fourth planet.\nQuestion: Which planet is fourth?",
            context="Mars is the fourth planet from the Sun.",
            question="Which planet is fourth?",
            answer="Mars",
        )

    def test_instruction_template(self):
        """Test InstructionTemplate zero-shot prompt assembly."""
        prompt = InstructionTemplate.assemble(self.cls_query, "Categorize news.")
        self.assertIn("Instruction:\nCategorize news.", prompt)
        self.assertIn("Input: Tech giant reveals quantum processor.\nLabel:", prompt)

    def test_example_template(self):
        """Test ExampleTemplate few-shot demonstration prompt assembly."""
        prompt = ExampleTemplate.assemble(self.cls_query, self.cls_demos)
        self.assertIn("Input: Oil prices surge globally.\nLabel: Business", prompt)
        self.assertIn("Input: Tech giant reveals quantum processor.\nLabel:", prompt)

    def test_hybrid_template(self):
        """Test HybridTemplate combined instruction + demonstration assembly."""
        prompt = HybridTemplate.assemble(self.cls_query, self.cls_demos, "Classify headlines.")
        self.assertIn("Instruction:\nClassify headlines.", prompt)
        self.assertIn("Examples:\nInput: Oil prices surge globally.", prompt)
        self.assertIn("Test Query:\nInput: Tech giant reveals quantum processor.\nLabel:", prompt)

    def test_prompt_validator_pass(self):
        """Test PromptValidator with valid prompt."""
        prompt_obj = PromptObject(
            prompt_text="Instruction:\nClassify.\n\nInput: Test\nLabel:",
            strategy="instruction",
            task_type="text_classification",
            instruction="Classify.",
            query="Input: Test\nLabel:",
        )
        is_valid, warnings = PromptValidator.validate(prompt_obj, max_token_limit=1000)
        self.assertTrue(is_valid)
        self.assertEqual(len(warnings), 0)

    def test_prompt_validator_overflow(self):
        """Test PromptValidator detecting token length limit overflow."""
        huge_text = "Word " * 2000 + "\nLabel:"
        prompt_obj = PromptObject(
            prompt_text=huge_text,
            strategy="instruction",
            task_type="text_classification",
            instruction="Classify.",
            query="Label:",
        )
        is_valid, warnings = PromptValidator.validate(prompt_obj, max_token_limit=50)
        self.assertFalse(is_valid)
        self.assertTrue(any("exceeds model limit" in w for w in warnings))

    def test_standard_few_shot_template_0shot(self):
        """Test StandardFewShotTemplate 0-shot prompt assembly (0 demonstrations)."""
        from prompt_engine.templates import StandardFewShotTemplate
        prompt = StandardFewShotTemplate.assemble(self.cls_query, [], "Classify headlines.")
        self.assertIn("Instruction:\nClassify headlines.", prompt)
        self.assertNotIn("Examples:", prompt)
        self.assertIn("Test Query:\nInput: Tech giant reveals quantum processor.\nLabel:", prompt)

    def test_standard_few_shot_template_with_demos(self):
        """Test StandardFewShotTemplate prompt assembly with demonstrations."""
        from prompt_engine.templates import StandardFewShotTemplate
        prompt = StandardFewShotTemplate.assemble(self.cls_query, self.cls_demos, "Classify headlines.")
        self.assertIn("Instruction:\nClassify headlines.", prompt)
        self.assertIn("Examples:\nInput: Oil prices surge globally.", prompt)
        self.assertIn("Test Query:\nInput: Tech giant reveals quantum processor.\nLabel:", prompt)

    def test_prompt_generator_integration(self):
        """Test PromptGenerator creating validated PromptObject with standard_few_shot."""
        prompt_obj = PromptGenerator.generate(
            query_record=self.qa_query,
            demonstrations=[],
            strategy="standard_few_shot",
            instruction_text="Answer concisely.",
        )
        self.assertTrue(prompt_obj.is_valid)
        self.assertEqual(prompt_obj.strategy, "standard_few_shot")
        self.assertEqual(prompt_obj.task_type, "question_answering")
        self.assertIn("Question: Which planet is fourth?", prompt_obj.prompt_text)

    def test_text_generation_few_shot_token_budget(self):
        """Test PromptGenerator budget trimming for 5-shot text generation prompts."""
        long_article = "This is a detailed news passage about global economics. " * 30
        gen_query = StandardRecord(
            id="gen_q",
            task_type="text_generation",
            input_text=long_article,
            context=long_article,
            answer="Summary highlights of economics.",
        )
        gen_demos = [
            StandardRecord(
                id=f"gen_d_{i}",
                task_type="text_generation",
                input_text=long_article,
                context=long_article,
                answer=f"Summary point {i}.",
            )
            for i in range(5)
        ]

        prompt_obj = PromptGenerator.generate(
            query_record=gen_query,
            demonstrations=gen_demos,
            strategy="standard_few_shot",
            instruction_text="Summarize the article.",
            max_token_limit=2048,
        )
        self.assertTrue(prompt_obj.is_valid)
        self.assertLessEqual(prompt_obj.estimated_tokens, 2048)
        self.assertIn("Instruction:\nSummarize the article.", prompt_obj.prompt_text)
        self.assertIn("Test Query:", prompt_obj.prompt_text)


if __name__ == "__main__":
    unittest.main()

