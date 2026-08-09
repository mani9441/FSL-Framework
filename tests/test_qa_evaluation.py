"""
Regression unit tests for SQuAD v2-compatible Question Answering Evaluation Layer.
Verifies all 5 required test cases:
1. Semantic / Extractive Match
2. Paraphrased Answer
3. SQuAD v2 Unanswerable Question
4. Wrong Answer
5. Partial Answer
"""

import unittest
from evaluation.qa_eval import QAEvaluator, normalize_answer, compute_token_f1, is_unanswerable_ref, is_no_answer_pred
from evaluation.parser import PredictionParser


class TestQAEvaluation(unittest.TestCase):

    def test_case_1_semantic_extractive_match(self):
        """CASE 1 — semantic/extractive match."""
        ref = "various academic disciplines"
        raw_pred = "The various academic disciplines that University of Chicago scholars played a major part in..."
        extracted_pred, parse_success = PredictionParser.parse_qa(raw_pred)
        
        eval_res = QAEvaluator.evaluate_single(extracted_pred, ref)

        self.assertTrue(parse_success)
        self.assertGreater(eval_res["qa_f1"], 0.0, "Expected positive F1 for semantic match")
        self.assertGreater(eval_res["response_correctness"], 0.0, "Expected positive correctness score")
        self.assertEqual(eval_res["response_completeness"], 1.0, "Expected 100% reference token coverage")
        self.assertTrue(eval_res["is_correct"])

    def test_case_2_paraphrased_answer(self):
        """CASE 2 — paraphrased answer."""
        ref = "a two-page statement"
        raw_pred = "The Kalven Report statement was a two-page document."
        extracted_pred, parse_success = PredictionParser.parse_qa(raw_pred)

        eval_res = QAEvaluator.evaluate_single(extracted_pred, ref)

        self.assertTrue(parse_success)
        self.assertGreater(eval_res["qa_f1"], 0.0, "Expected positive QA F1 score")
        self.assertGreater(eval_res["response_correctness"], 0.0)
        self.assertTrue(eval_res["is_correct"], "Paraphrased match should be evaluated as correct")

    def test_case_3_no_answer(self):
        """CASE 3 — SQuAD v2 unanswerable question."""
        ref = []  # SQuAD v2 unanswerable reference format
        raw_pred = "I cannot find an answer in the provided context."
        extracted_pred, parse_success = PredictionParser.parse_qa(raw_pred)

        eval_res = QAEvaluator.evaluate_single(extracted_pred, ref)

        self.assertTrue(parse_success)
        self.assertEqual(eval_res["qa_exact_match"], 1.0, "Correct no-answer recognition should receive EM=1.0")
        self.assertEqual(eval_res["qa_f1"], 1.0, "Correct no-answer recognition should receive F1=1.0")
        self.assertTrue(eval_res["is_correct"])

    def test_case_4_wrong_answer(self):
        """CASE 4 — wrong answer."""
        ref = "1413"
        raw_pred = "1300"
        extracted_pred, parse_success = PredictionParser.parse_qa(raw_pred)

        eval_res = QAEvaluator.evaluate_single(extracted_pred, ref)

        self.assertEqual(eval_res["qa_exact_match"], 0.0)
        self.assertEqual(eval_res["qa_f1"], 0.0)
        self.assertFalse(eval_res["is_correct"])

    def test_case_5_partial_answer(self):
        """CASE 5 — partial answer."""
        ref = "a two-page statement"
        raw_pred = "two pages"
        extracted_pred, parse_success = PredictionParser.parse_qa(raw_pred)

        eval_res = QAEvaluator.evaluate_single(extracted_pred, ref)

        self.assertGreater(eval_res["qa_f1"], 0.0, "Partial answer should yield positive F1 score")
        self.assertEqual(eval_res["qa_exact_match"], 0.0, "Partial answer should not be exact match")

    def test_prediction_parser_preamble_removal(self):
        """Test PredictionParser removing common LLM preambles."""
        raw = "The correct answer is:\n\nvarious academic disciplines\n\nThis answer can be inferred..."
        extracted, success = PredictionParser.parse_qa(raw)
        self.assertTrue(success)
        self.assertEqual(extracted, "various academic disciplines")


if __name__ == "__main__":
    unittest.main()
