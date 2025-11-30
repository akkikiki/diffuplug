"""Test model imports and basic functionality."""

import pytest


class TestModelImports:
    """Test that model modules can be imported."""

    def test_import_dream_model(self):
        """Test that Dream model can be imported."""
        from dllm_plugin.models import dream
        assert dream is not None

    def test_import_llada_model(self):
        """Test that LLaDA model can be imported."""
        from dllm_plugin.models import llada
        assert llada is not None

    def test_dream_model_has_class(self):
        """Test that Dream model defines the expected class."""
        from dllm_plugin.models.dream import DreamForDiffusionLMVLLM
        assert DreamForDiffusionLMVLLM is not None

    def test_llada_model_has_class(self):
        """Test that LLaDA model defines the expected class."""
        from dllm_plugin.models.llada import LLaDAForDiffusionLMVLLM
        assert LLaDAForDiffusionLMVLLM is not None
