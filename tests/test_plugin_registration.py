"""Test plugin registration functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestPluginRegistration:
    """Test cases for plugin registration."""

    def test_import_dllm_plugin(self):
        """Test that dllm_plugin can be imported."""
        import dllm_plugin
        assert dllm_plugin is not None

    def test_register_function_exists(self):
        """Test that the register function exists."""
        import dllm_plugin
        assert hasattr(dllm_plugin, 'register')
        assert callable(dllm_plugin.register)

    @patch('dllm_plugin.ModelRegistry')
    @patch('dllm_plugin._patch_kv_cache_manager')
    @patch('dllm_plugin._patch_engine_core')
    @patch('dllm_plugin._patch_llm_generation')
    def test_register_calls_patches(
        self,
        mock_patch_llm,
        mock_patch_engine,
        mock_patch_kv,
        mock_registry
    ):
        """Test that register() calls all patch functions."""
        # Mock the registry methods
        mock_registry.get_supported_archs.return_value = []
        mock_registry.register_model = Mock()

        import dllm_plugin
        dllm_plugin.register()

        # Verify patches were called
        mock_patch_kv.assert_called_once()
        mock_patch_engine.assert_called_once()
        mock_patch_llm.assert_called_once()

    @patch('dllm_plugin.ModelRegistry')
    @patch('dllm_plugin._patch_kv_cache_manager')
    @patch('dllm_plugin._patch_engine_core')
    @patch('dllm_plugin._patch_llm_generation')
    def test_register_registers_models(
        self,
        mock_patch_llm,
        mock_patch_engine,
        mock_patch_kv,
        mock_registry
    ):
        """Test that register() registers all expected models."""
        # Mock the registry
        mock_registry.get_supported_archs.return_value = []
        mock_registry.register_model = Mock()

        import dllm_plugin
        dllm_plugin.register()

        # Check that models were registered
        calls = mock_registry.register_model.call_args_list
        registered_models = [call[0][0] for call in calls]

        assert "DreamForDiffusionLM" in registered_models
        assert "LLaDAForDiffusionLM" in registered_models
        assert "LLaDAModelLM" in registered_models

    @patch('dllm_plugin.ModelRegistry')
    @patch('dllm_plugin._patch_kv_cache_manager')
    @patch('dllm_plugin._patch_engine_core')
    @patch('dllm_plugin._patch_llm_generation')
    def test_register_skips_existing_models(
        self,
        mock_patch_llm,
        mock_patch_engine,
        mock_patch_kv,
        mock_registry
    ):
        """Test that register() skips already registered models."""
        # Mock registry to show DreamForDiffusionLM is already registered
        mock_registry.get_supported_archs.return_value = ["DreamForDiffusionLM"]
        mock_registry.register_model = Mock()

        import dllm_plugin
        dllm_plugin.register()

        # Check that DreamForDiffusionLM was not registered again
        calls = mock_registry.register_model.call_args_list
        registered_models = [call[0][0] for call in calls]

        assert "DreamForDiffusionLM" not in registered_models
        # But LLaDA models should still be registered
        assert "LLaDAForDiffusionLM" in registered_models
        assert "LLaDAModelLM" in registered_models
