"""
vLLM Plugin for Diffusion Language Models

This plugin registers diffusion language models (Dream and LLaDA) with vLLM.
"""

from vllm import ModelRegistry


def register():
    """
    Register diffusion language models with vLLM.

    This function registers the following models:
    - DreamForDiffusionLM: Dream diffusion language model
    - LLaDAForDiffusionLM: LLaDA diffusion language model
    - LLaDAModelLM: Alternative name for LLaDA (used by some checkpoints)
    """
    # Register Dream model
    if "DreamForDiffusionLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "DreamForDiffusionLM",
            "dllm_plugin.models.dream:DreamForDiffusionLMVLLM"
        )

    # Register LLaDA model
    if "LLaDAForDiffusionLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "LLaDAForDiffusionLM",
            "dllm_plugin.models.llada:LLaDAForDiffusionLMVLLM"
        )

    # Register alternative LLaDA architecture name (used in some HF configs)
    if "LLaDAModelLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "LLaDAModelLM",
            "dllm_plugin.models.llada:LLaDAForDiffusionLMVLLM"
        )
