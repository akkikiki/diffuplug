"""Test LLaDA sampler functionality."""

import pytest


class TestLLaDASampler:
    """Test cases for LLaDA sampler."""

    def test_import_sampler(self):
        """Test that LLaDASampler can be imported."""
        from dllm_plugin.llada_sampler import LLaDASampler
        assert LLaDASampler is not None

    def test_sampler_has_required_methods(self):
        """Test that LLaDASampler has required methods."""
        from dllm_plugin.llada_sampler import LLaDASampler

        # Check for key methods
        assert hasattr(LLaDASampler, '__init__')
        assert hasattr(LLaDASampler, '__call__')
