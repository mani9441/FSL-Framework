"""
Unit tests for Module 5 — Model Interface Layer.
"""

import unittest
from unittest.mock import patch
from model_interfaces.schema import ModelResponse
from model_interfaces.base import estimate_cost
from model_interfaces.openai_wrapper import OpenAIModelWrapper
from model_interfaces.gemini_wrapper import GoogleGeminiModelWrapper
from model_interfaces.groq_wrapper import GroqModelWrapper
from model_interfaces.huggingface_wrapper import HuggingFaceModelWrapper
from model_interfaces.execution_manager import ModelExecutionManager


class TestModelInterfaces(unittest.TestCase):

    def test_model_response_serialization(self):
        """Test ModelResponse serialization and deserialization."""
        resp = ModelResponse(
            generated_text="Business",
            model_name="gemini-2.5-flash",
            provider="google",
            latency_seconds=0.12,
            prompt_tokens=40,
            completion_tokens=5,
            total_tokens=45,
            estimated_cost_usd=0.000027,
            success=True,
        )
        resp_dict = resp.to_dict()
        reconstructed = ModelResponse.from_dict(resp_dict)

        self.assertEqual(reconstructed.generated_text, "Business")
        self.assertEqual(reconstructed.prompt_tokens, 40)
        self.assertTrue(reconstructed.success)

    def test_cost_estimation(self):
        """Test financial cost calculation logic."""
        cost = estimate_cost("gpt-3.5-turbo", prompt_tokens=1000, completion_tokens=1000)
        self.assertAlmostEqual(cost, 0.002, places=4)

    def test_unconfigured_api_key_handling(self):
        """Test model wrappers returning clean error ModelResponse when API keys are unconfigured."""
        wrapper = GoogleGeminiModelWrapper(model_name="gemini-2.5-flash")
        wrapper.api_key = ""
        resp = wrapper.generate_response("Test prompt")
        self.assertFalse(resp.success)
        self.assertIn("Missing GOOGLE_API_KEY", resp.error_message)

        groq_wrapper = GroqModelWrapper(model_name="llama-3.1-8b-instant")
        groq_wrapper.api_key = ""
        resp_groq = groq_wrapper.generate_response("Test prompt")
        self.assertFalse(resp_groq.success)
        self.assertIn("Missing GROQ_API_KEY", resp_groq.error_message)

    @patch.object(GoogleGeminiModelWrapper, "generate_response")
    def test_gemini_wrapper_mocked_test(self, mock_gen):
        """Test Google Gemini wrapper execution with mocked API response."""
        mock_gen.return_value = ModelResponse(
            generated_text="Business",
            model_name="gemini-2.5-flash",
            provider="google",
            latency_seconds=0.05,
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            estimated_cost_usd=0.0,
            success=True,
        )
        wrapper = GoogleGeminiModelWrapper(model_name="gemini-2.5-flash")
        resp = wrapper.generate_response("Question: What is NASA?\nAnswer:", temperature=0.0)
        self.assertTrue(resp.success)
        self.assertEqual(resp.generated_text, "Business")

    @patch.object(OpenAIModelWrapper, "generate_response")
    def test_execution_manager_retry(self, mock_gen):
        """Test ModelExecutionManager retry loop and metadata recording."""
        mock_gen.return_value = ModelResponse(
            generated_text="Business",
            model_name="gpt-3.5-turbo",
            provider="openai",
            latency_seconds=0.05,
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            estimated_cost_usd=0.0,
            success=True,
        )
        exec_mgr = ModelExecutionManager(model_name="gpt-3.5-turbo", provider="openai")
        resp = exec_mgr.execute_with_retry(
            prompt_text="Classify headline: Stock market rallies.\nLabel:",
            retries=2,
        )
        self.assertTrue(resp.success)
        self.assertGreater(resp.total_tokens, 0)

    @patch("requests.get")
    @patch("requests.post")
    def test_ollama_wrapper_configuration(self, mock_post, mock_get):
        """Test OllamaModelWrapper explicit thinking, seed, and version tracking configuration."""
        from model_interfaces.ollama_wrapper import OllamaModelWrapper

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"version": "0.5.7"}

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "World",
            "prompt_eval_count": 25,
            "eval_count": 2,
            "total_duration": 1500000000,
        }

        wrapper = OllamaModelWrapper(model_name="qwen3:8b", think=False, seed=42)
        self.assertEqual(wrapper.think, False)
        self.assertEqual(wrapper.seed, 42)
        self.assertEqual(wrapper.ollama_version, "0.5.7")

        resp = wrapper.generate_response("Classify headline: Stock market rallies.\nLabel:")
        self.assertTrue(resp.success)
        self.assertEqual(resp.generated_text, "World")
        self.assertEqual(resp.prompt_tokens, 25)
        self.assertEqual(resp.completion_tokens, 2)

        # Verify payload sent to Ollama REST API
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        self.assertEqual(payload.get("think"), False)
        self.assertEqual(payload.get("options", {}).get("seed"), 42)
        self.assertEqual(kwargs.get("timeout"), 120)


if __name__ == "__main__":
    unittest.main()

