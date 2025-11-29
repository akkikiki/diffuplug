"""
vLLM Plugin for Diffusion Language Models

This plugin registers diffusion language models (Dream and LLaDA) with vLLM.
"""

import sys
from vllm import ModelRegistry


def _patch_kv_cache_manager():
    """
    Patch vLLM's KV cache manager to allow multiple block sizes for diffusion models.
    
    This is a workaround for diffusion models that don't use standard KV caching.
    The assertion in vLLM's kv_cache_manager.py checks for only one block size,
    but diffusion models may have different block sizes. We patch both KVCacheManager.__init__
    and get_kv_cache_coordinator to unify block sizes before the assertion is checked.
    """
    try:
        import vllm.v1.core.kv_cache_manager as kv_cache_manager_module
        import vllm.v1.core.kv_cache_coordinator as coordinator_module
        from dataclasses import replace
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Patch KVCacheManager.__init__
        original_manager_init = kv_cache_manager_module.KVCacheManager.__init__
        
        def patched_manager_init(self, kv_cache_config, *args, **kwargs):
            # Handle empty groups case (attention-free models)
            # The assertion checks len(set(...)) == 1, but if there are no groups,
            # the set is empty (len == 0), which fails the assertion
            if kv_cache_config and hasattr(kv_cache_config, 'kv_cache_groups'):
                if len(kv_cache_config.kv_cache_groups) == 0:
                    # Empty groups - this is an attention-free model (like diffusion models)
                    # The assertion will fail because len(empty_set) == 0, not 1
                    # Create a dummy group to satisfy the assertion
                    logger.info(
                        "dllm_plugin: Empty KV cache groups detected (attention-free model). "
                        "Creating dummy group to satisfy assertion."
                    )
                    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec
                    import torch
                    # Get hash_block_size from kwargs or args
                    # KVCacheManager.__init__ signature: (self, kv_cache_config, max_model_len, hash_block_size, ...)
                    # In our patched function: (self, kv_cache_config, *args, **kwargs)
                    # So args[0] = max_model_len, args[1] = hash_block_size
                    hash_block_size = kwargs.get('hash_block_size', None)
                    if hash_block_size is None and len(args) >= 2:
                        hash_block_size = args[1]  # hash_block_size is args[1] (after max_model_len)
                    if hash_block_size is None:
                        hash_block_size = 16  # Default fallback
                    dummy_spec = FullAttentionSpec(
                        block_size=hash_block_size,
                        num_kv_heads=1,
                        head_size=64,
                        dtype=torch.float16
                    )
                    dummy_group = KVCacheGroupSpec(layer_names=[], kv_cache_spec=dummy_spec)
                    object.__setattr__(kv_cache_config, 'kv_cache_groups', [dummy_group])
                    logger.info(
                        f"dllm_plugin: Created dummy KV cache group with block_size={hash_block_size} "
                        "to satisfy assertion for attention-free model"
                    )
                elif len(kv_cache_config.kv_cache_groups) > 0:
                    # Check for multiple block sizes
                    block_sizes = {
                        group.kv_cache_spec.block_size 
                        for group in kv_cache_config.kv_cache_groups
                    }
                    
                    # Log what we're seeing
                    logger.info(
                        f"dllm_plugin: KVCacheManager init - found {len(kv_cache_config.kv_cache_groups)} groups "
                        f"with block sizes: {block_sizes}"
                    )
                    
                    # If multiple block sizes exist, unify them to the maximum
                    # This is a workaround for diffusion models
                    if len(block_sizes) > 1:
                        max_block_size = max(block_sizes)
                        logger.info(
                            f"dllm_plugin: Unifying multiple block sizes {block_sizes} to {max_block_size} in KVCacheManager"
                        )
                        # Create new groups with unified block sizes
                        # KVCacheGroupSpec is not frozen, but kv_cache_spec is, so we need to create new specs
                        from vllm.v1.kv_cache_interface import KVCacheGroupSpec
                        unified_groups = []
                        for group in kv_cache_config.kv_cache_groups:
                            if group.kv_cache_spec.block_size != max_block_size:
                                # Create a new spec with unified block size
                                new_spec = replace(
                                    group.kv_cache_spec, 
                                    block_size=max_block_size
                                )
                                # Create a new group with the unified spec
                                unified_groups.append(
                                    KVCacheGroupSpec(group.layer_names, new_spec)
                                )
                            else:
                                unified_groups.append(group)
                        # Replace the groups list - use object.__setattr__ to ensure modification
                        object.__setattr__(kv_cache_config, 'kv_cache_groups', unified_groups)
                        
                        # Verify unification worked
                        final_block_sizes = {
                            group.kv_cache_spec.block_size 
                            for group in kv_cache_config.kv_cache_groups
                        }
                        logger.info(
                            f"dllm_plugin: After unification - block sizes: {final_block_sizes}"
                        )
            
            # Before calling original __init__, ensure block sizes are unified one more time
            # This is a safety check in case the coordinator modified the config
            if kv_cache_config and hasattr(kv_cache_config, 'kv_cache_groups') and len(kv_cache_config.kv_cache_groups) > 0:
                block_sizes_check = {
                    group.kv_cache_spec.block_size 
                    for group in kv_cache_config.kv_cache_groups
                }
                if len(block_sizes_check) > 1:
                    max_block_size = max(block_sizes_check)
                    logger.warning(
                        f"dllm_plugin: Found multiple block sizes {block_sizes_check} right before init, "
                        f"unifying to {max_block_size}"
                    )
                    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
                    unified_groups = []
                    for group in kv_cache_config.kv_cache_groups:
                        if group.kv_cache_spec.block_size != max_block_size:
                            new_spec = replace(group.kv_cache_spec, block_size=max_block_size)
                            unified_groups.append(KVCacheGroupSpec(group.layer_names, new_spec))
                        else:
                            unified_groups.append(group)
                    object.__setattr__(kv_cache_config, 'kv_cache_groups', unified_groups)
            
            # Call the original __init__ - wrap in try/except to catch assertion errors
            try:
                original_manager_init(self, kv_cache_config, *args, **kwargs)
            except AssertionError as e:
                if "Only one block size is supported" in str(e):
                    # If assertion fails, check what block sizes we have now
                    logger.error("dllm_plugin: Assertion caught, checking current block sizes")
                    if kv_cache_config and hasattr(kv_cache_config, 'kv_cache_groups'):
                        current_block_sizes = {
                            group.kv_cache_spec.block_size 
                            for group in kv_cache_config.kv_cache_groups
                        }
                        logger.error(
                            f"dllm_plugin: Current block sizes when assertion failed: {current_block_sizes}, "
                            f"num groups: {len(kv_cache_config.kv_cache_groups) if kv_cache_config.kv_cache_groups else 0}"
                        )
                        
                        # If there are no groups (empty set), the assertion is checking len(set()) == 1
                        # which fails because len(set()) == 0. This happens for attention-free models.
                        # We need to handle this case by either:
                        # 1. Disabling caching if there are no groups
                        # 2. Or ensuring the assertion doesn't run for empty groups
                        
                        if len(current_block_sizes) == 0:
                            # Empty groups - this is an attention-free model
                            # The assertion is failing because it's checking len(set()) == 1
                            # but len(set()) == 0. We need to patch the assertion to handle this.
                            logger.warning(
                                "dllm_plugin: Empty KV cache groups detected (attention-free model). "
                                "The assertion is failing because it expects exactly one block size, "
                                "but there are no groups. This is expected for diffusion models."
                            )
                            # For attention-free models, we should disable caching or handle this differently
                            # But since we can't modify the assertion directly, we'll create a dummy group
                            # with a single block size to satisfy the assertion
                            from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec
                            import torch
                            # Create a dummy group with a standard block size
                            dummy_block_size = kwargs.get('hash_block_size', 16) or 16
                            dummy_spec = FullAttentionSpec(
                                block_size=dummy_block_size,
                                num_kv_heads=1,
                                head_size=64,
                                dtype=torch.float16
                            )
                            dummy_group = KVCacheGroupSpec(layer_names=[], kv_cache_spec=dummy_spec)
                            object.__setattr__(kv_cache_config, 'kv_cache_groups', [dummy_group])
                            logger.info(
                                f"dllm_plugin: Created dummy KV cache group with block_size={dummy_block_size} "
                                "to satisfy assertion for attention-free model"
                            )
                            # Retry with the dummy group
                            original_manager_init(self, kv_cache_config, *args, **kwargs)
                        elif len(current_block_sizes) > 1:
                            # Multiple block sizes - unify them
                            max_block_size = max(current_block_sizes)
                            logger.warning(
                                f"dllm_plugin: Force unifying all block sizes to {max_block_size}"
                            )
                            from vllm.v1.kv_cache_interface import KVCacheGroupSpec
                            unified_groups = []
                            for group in kv_cache_config.kv_cache_groups:
                                if group.kv_cache_spec.block_size != max_block_size:
                                    new_spec = replace(group.kv_cache_spec, block_size=max_block_size)
                                    unified_groups.append(KVCacheGroupSpec(group.layer_names, new_spec))
                                else:
                                    unified_groups.append(group)
                            object.__setattr__(kv_cache_config, 'kv_cache_groups', unified_groups)
                            
                            # Verify again
                            verify_block_sizes = {
                                group.kv_cache_spec.block_size 
                                for group in kv_cache_config.kv_cache_groups
                            }
                            logger.info(
                                f"dllm_plugin: After force unification - block sizes: {verify_block_sizes}"
                            )
                            
                            # Clear any cached coordinator if it exists
                            if hasattr(self, 'coordinator'):
                                delattr(self, 'coordinator')
                            
                            # Retry
                            original_manager_init(self, kv_cache_config, *args, **kwargs)
                        else:
                            logger.error(
                                f"dllm_plugin: Block sizes are already unified ({current_block_sizes}), "
                                "but assertion still failed. This might be a different issue."
                            )
                            raise
                else:
                    raise
        
        kv_cache_manager_module.KVCacheManager.__init__ = patched_manager_init
        
        # Also patch get_kv_cache_coordinator to ensure block sizes are unified
        original_get_coordinator = coordinator_module.get_kv_cache_coordinator
        
        def patched_get_coordinator(*args, **kwargs):
            # Get the kv_cache_config from kwargs or args
            kv_cache_config = kwargs.get('kv_cache_config') or (args[0] if args else None)
            
            if kv_cache_config and hasattr(kv_cache_config, 'kv_cache_groups') and len(kv_cache_config.kv_cache_groups) > 0:
                # Check for multiple block sizes
                block_sizes = {
                    group.kv_cache_spec.block_size 
                    for group in kv_cache_config.kv_cache_groups
                }
                
                # If multiple block sizes exist, unify them to the maximum
                if len(block_sizes) > 1:
                    max_block_size = max(block_sizes)
                    logger.info(
                        f"dllm_plugin: Unifying multiple block sizes {block_sizes} to {max_block_size} in coordinator"
                    )
                    # Create new groups with unified block sizes
                    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
                    unified_groups = []
                    for group in kv_cache_config.kv_cache_groups:
                        if group.kv_cache_spec.block_size != max_block_size:
                            # Create a new spec with unified block size
                            new_spec = replace(
                                group.kv_cache_spec, 
                                block_size=max_block_size
                            )
                            # Create a new group with the unified spec
                            unified_groups.append(
                                KVCacheGroupSpec(group.layer_names, new_spec)
                            )
                        else:
                            unified_groups.append(group)
                    # Replace the groups list - use object.__setattr__ to ensure modification
                    object.__setattr__(kv_cache_config, 'kv_cache_groups', unified_groups)
            
            # Call the original function
            return original_get_coordinator(*args, **kwargs)
        
        coordinator_module.get_kv_cache_coordinator = patched_get_coordinator
        
        # Also patch UnitaryKVCacheCoordinator.__init__ to handle multiple block sizes
        # This is needed because the assertion might be in the coordinator
        original_unitary_init = coordinator_module.UnitaryKVCacheCoordinator.__init__
        
        def patched_unitary_init(self, kv_cache_config, *args, **kwargs):
            # Ensure block sizes are unified before coordinator initialization
            if kv_cache_config and hasattr(kv_cache_config, 'kv_cache_groups') and len(kv_cache_config.kv_cache_groups) > 0:
                block_sizes = {
                    group.kv_cache_spec.block_size 
                    for group in kv_cache_config.kv_cache_groups
                }
                if len(block_sizes) > 1:
                    max_block_size = max(block_sizes)
                    logger.info(
                        f"dllm_plugin: Unifying block sizes {block_sizes} to {max_block_size} in UnitaryKVCacheCoordinator"
                    )
                    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
                    unified_groups = []
                    for group in kv_cache_config.kv_cache_groups:
                        if group.kv_cache_spec.block_size != max_block_size:
                            new_spec = replace(group.kv_cache_spec, block_size=max_block_size)
                            unified_groups.append(KVCacheGroupSpec(group.layer_names, new_spec))
                        else:
                            unified_groups.append(group)
                    # Replace the groups list - use object.__setattr__ to ensure modification
                    object.__setattr__(kv_cache_config, 'kv_cache_groups', unified_groups)
            
            # Call original init
            original_unitary_init(self, kv_cache_config, *args, **kwargs)
        
        coordinator_module.UnitaryKVCacheCoordinator.__init__ = patched_unitary_init
        
        # Also patch the base KVCacheCoordinator.__init__ to ensure block sizes are unified
        # before single_type_managers are created
        original_base_coordinator_init = coordinator_module.KVCacheCoordinator.__init__
        
        def patched_base_coordinator_init(self, kv_cache_config, *args, **kwargs):
            # Ensure block sizes are unified before coordinator initialization
            if kv_cache_config and hasattr(kv_cache_config, 'kv_cache_groups') and len(kv_cache_config.kv_cache_groups) > 0:
                block_sizes = {
                    group.kv_cache_spec.block_size 
                    for group in kv_cache_config.kv_cache_groups
                }
                if len(block_sizes) > 1:
                    max_block_size = max(block_sizes)
                    logger.info(
                        f"dllm_plugin: Unifying block sizes {block_sizes} to {max_block_size} in base KVCacheCoordinator"
                    )
                    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
                    unified_groups = []
                    for group in kv_cache_config.kv_cache_groups:
                        if group.kv_cache_spec.block_size != max_block_size:
                            new_spec = replace(group.kv_cache_spec, block_size=max_block_size)
                            unified_groups.append(KVCacheGroupSpec(group.layer_names, new_spec))
                        else:
                            unified_groups.append(group)
                    # Replace the groups list - use object.__setattr__ to ensure modification
                    object.__setattr__(kv_cache_config, 'kv_cache_groups', unified_groups)
            
            # Call original init
            original_base_coordinator_init(self, kv_cache_config, *args, **kwargs)
        
        coordinator_module.KVCacheCoordinator.__init__ = patched_base_coordinator_init
        
        logger.info("dllm_plugin: Successfully patched KVCacheManager, get_kv_cache_coordinator, KVCacheCoordinator, and UnitaryKVCacheCoordinator")
        
    except Exception as e:
        # If patching fails, log but don't fail registration
        import warnings
        import traceback
        warnings.warn(f"Failed to patch KV cache manager: {e}\n{traceback.format_exc()}", RuntimeWarning)


def _patch_engine_core():
    """
    Patch vLLM's EngineCoreProc to add custom diffusion generation method.

    This must happen at import time, before the worker process is spawned,
    so the patched method is available in the worker process.
    """
    try:
        from .generation import patch_engine_core_for_diffusion

        # Patch EngineCoreProc before any LLM instances are created
        patch_engine_core_for_diffusion()

        import logging
        logger = logging.getLogger(__name__)
        logger.info("dllm_plugin: Patched EngineCoreProc with custom diffusion generation method")

    except Exception as e:
        import warnings
        import traceback
        warnings.warn(f"Failed to patch EngineCoreProc: {e}\n{traceback.format_exc()}", RuntimeWarning)


def _patch_llm_generation():
    """
    Patch vLLM's LLM class to use custom generation for diffusion models.

    This patches the LLM.generate method to detect diffusion models and
    use the custom diffusion generation process instead of the standard
    autoregressive loop.
    """
    try:
        from vllm import LLM
        # Use relative import since we're in the same package
        from .generation import patch_llm_generate
        
        # Store original generate method
        original_generate = LLM.generate
        
        # Create a wrapper that patches instances after initialization
        original_init = LLM.__init__
        
        def patched_init(self, *args, **kwargs):
            model_name = kwargs.get('model', args[0] if args else 'unknown')
            logger.info(f"dllm_plugin: LLM.__init__ called with model={model_name}")
            logger.info("dllm_plugin: This means the patching IS working!")
            original_init(self, *args, **kwargs)
            logger.info("dllm_plugin: LLM.__init__ completed, now patching generate method")
            # Patch the instance's generate method
            try:
                patch_llm_generate(self)
                logger.info("dllm_plugin: ✓ Successfully patched LLM instance generate method")
            except Exception as e:
                logger.error(f"dllm_plugin: ✗ Failed to patch generate method: {e}", exc_info=True)
        
        LLM.__init__ = patched_init
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("dllm_plugin: Patched LLM.__init__ to enable custom diffusion generation")
        
    except Exception as e:
        import warnings
        import traceback
        warnings.warn(f"Failed to patch LLM generation: {e}\n{traceback.format_exc()}", RuntimeWarning)


def register():
    """
    Register diffusion language models with vLLM.

    This function registers the following models:
    - DreamForDiffusionLM: Dream diffusion language model
    - LLaDAForDiffusionLM: LLaDA diffusion language model
    - LLaDAModelLM: Alternative name for LLaDA (used by some checkpoints)
    """
    # Patch KV cache manager to handle multiple block sizes
    _patch_kv_cache_manager()

    # Patch EngineCoreProc with custom diffusion generation method
    # This MUST happen before LLM instances are created so the worker process has the patch
    _patch_engine_core()

    # Patch LLM generation to use custom diffusion generation
    _patch_llm_generation()
    
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
