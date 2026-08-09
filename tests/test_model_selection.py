"""
Unit tests for Model Selection & Environment Enablement Logic (Section 16 Verification Suite).
"""

import os
import unittest

from experiment_engine.matrix_generator import ExperimentMatrixGenerator
from model_interfaces.execution_manager import ModelExecutionManager


class TestModelSelection(unittest.TestCase):
    """Test suite verifying environment-controlled model selection across all providers."""

    def setUp(self):
        # Save existing environment variables
        self.original_env = os.environ.copy()

        # Set default enabled flags for testing
        os.environ["OLLAMA_LLAMA31_8B_ENABLED"] = "true"
        os.environ["OLLAMA_QWEN3_8B_ENABLED"] = "true"
        os.environ["OLLAMA_MISTRAL_7B_ENABLED"] = "true"
        os.environ["OLLAMA_GEMMA3_4B_ENABLED"] = "true"
        os.environ["OLLAMA_PHI4_MINI_ENABLED"] = "true"
        os.environ["GROQ_LLAMA31_8B_INSTANT_ENABLED"] = "true"
        os.environ["HF_QWEN25_7B_INSTRUCT_ENABLED"] = "true"
        os.environ["GOOGLE_GEMINI_35_FLASH_LITE_ENABLED"] = "true"

    def tearDown(self):
        # Restore environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_1_no_provider_no_model(self):
        """Test Case 1 — No provider AND no model: selects ALL enabled models across ALL providers."""
        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter=None,
            model_filter=None,
        )
        model_names = [m[0] for m in models]
        providers = set(m[1] for m in models)

        self.assertIn("llama3.1:8b", model_names)
        self.assertIn("qwen3:8b", model_names)
        self.assertIn("mistral:7b", model_names)
        self.assertIn("gemma3:4b", model_names)
        self.assertIn("phi4-mini", model_names)
        self.assertTrue({"groq", "huggingface", "google", "ollama"}.issubset(providers))

    def test_2_ollama_provider_only(self):
        """Test Case 2 — Provider specified (ollama), model not specified: selects only enabled Ollama models."""
        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter="ollama",
            model_filter=None,
        )
        for m_name, m_prov in models:
            self.assertEqual(m_prov, "ollama")

        model_names = [m[0] for m in models]
        self.assertEqual(len(models), 5)
        self.assertIn("llama3.1:8b", model_names)
        self.assertIn("qwen3:8b", model_names)
        self.assertIn("mistral:7b", model_names)
        self.assertIn("gemma3:4b", model_names)
        self.assertIn("phi4-mini", model_names)

    def test_3_groq_provider_only(self):
        """Test Case 3 — Another provider (groq): selects only enabled Groq models."""
        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter="groq",
            model_filter=None,
        )
        for m_name, m_prov in models:
            self.assertEqual(m_prov, "groq")

    def test_4_explicit_enabled_model(self):
        """Test Case 4 — Explicit enabled model: runs requested model."""
        os.environ["OLLAMA_QWEN3_8B_ENABLED"] = "true"
        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter=None,
            model_filter="qwen3:8b",
        )
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0][0], "qwen3:8b")
        self.assertEqual(models[0][1], "ollama")

    def test_5_explicit_disabled_model(self):
        """
        Test Case 5 — Explicit disabled model:
        With OLLAMA_QWEN3_8B_ENABLED=false, explicit --model qwen3:8b MUST STILL RUN!
        """
        os.environ["OLLAMA_QWEN3_8B_ENABLED"] = "false"
        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter=None,
            model_filter="qwen3:8b",
        )
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0][0], "qwen3:8b")
        self.assertEqual(models[0][1], "ollama")

    def test_6_mixed_enabled_configuration(self):
        """Test Case 6 — Mixed enabled configuration: filters out disabled models during automatic selection."""
        os.environ["OLLAMA_LLAMA31_8B_ENABLED"] = "true"
        os.environ["OLLAMA_QWEN3_8B_ENABLED"] = "false"
        os.environ["OLLAMA_MISTRAL_7B_ENABLED"] = "true"
        os.environ["OLLAMA_GEMMA3_4B_ENABLED"] = "false"
        os.environ["OLLAMA_PHI4_MINI_ENABLED"] = "true"

        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter="ollama",
            model_filter=None,
        )
        model_names = [m[0] for m in models]

        self.assertIn("llama3.1:8b", model_names)
        self.assertIn("mistral:7b", model_names)
        self.assertIn("phi4-mini", model_names)

        self.assertNotIn("qwen3:8b", model_names)
        self.assertNotIn("gemma3:4b", model_names)

    def test_7_alias_and_provider_inference(self):
        """Test backward compatibility alias llama3.1:latest and provider inference."""
        models = ExperimentMatrixGenerator.select_models_for_campaign(
            provider_filter=None,
            model_filter="llama3.1:latest",
        )
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0][0], "llama3.1:latest")
        self.assertEqual(models[0][1], "ollama")

        # Infer provider test
        self.assertEqual(ModelExecutionManager._infer_provider("qwen3:8b"), "ollama")
        self.assertEqual(ModelExecutionManager._infer_provider("mistral:7b"), "ollama")
        self.assertEqual(ModelExecutionManager._infer_provider("gemma3:4b"), "ollama")
        self.assertEqual(ModelExecutionManager._infer_provider("phi4-mini"), "ollama")
        self.assertEqual(ModelExecutionManager._infer_provider("llama3.1:8b"), "ollama")


if __name__ == "__main__":
    unittest.main()
