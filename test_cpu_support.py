#!/usr/bin/env python3
"""
Test script to verify CPU support patch for Diffulex.
"""

import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_cpu_support():
    """Test that the CPU support patch works."""

    print("=" * 80)
    print("Testing CPU Support Patch for Diffulex")
    print("=" * 80)

    # Step 1: Register plugin
    print("\n1. Registering dllm_plugin...")
    try:
        sys.path.insert(0, 'dllm_plugin')
        import dllm_plugin
        dllm_plugin.register()
        print("   ✓ Plugin registered successfully")
    except Exception as e:
        print(f"   ✗ Failed to register plugin: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Import vLLM
    print("\n2. Importing vLLM...")
    try:
        from vllm import LLM, SamplingParams
        print("   ✓ vLLM imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import vLLM: {e}")
        return False

    # Step 3: Test sequence device parameter
    print("\n3. Testing SequenceForDiffusionLM device parameter...")
    try:
        sys.path.insert(0, 'Diffulex')
        from d2f_engine.engine.sequence import SequenceForDiffusionLM
        from d2f_engine.sampling_params import SamplingParams as DiffulexSamplingParams
        from d2f_engine.config import Config as DiffulexConfig

        # Create a minimal config
        config = DiffulexConfig(
            model=".",  # Dummy path
            model_type='diffusion_lm',
            max_model_len=128,
            diffusion_block_size=32,
            kvcache_block_size=16,
            mask_token_id=126336,
            tensor_parallel_size=1,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            num_kvcache_blocks=100,
        )

        # Test CPU device
        seq_cpu = SequenceForDiffusionLM(
            token_ids=[1, 2, 3],
            sampling_params=DiffulexSamplingParams(),
            config=config,
            device='cpu'
        )

        assert seq_cpu.device == 'cpu', f"Expected device='cpu', got {seq_cpu.device}"
        print(f"   ✓ CPU device set correctly: {seq_cpu.device}")

        # Test mask creation with CPU device
        seq_cpu.update_block_mask(is_prefill=True)
        assert seq_cpu.block_mask is not None, "block_mask should be created"
        assert str(seq_cpu.block_mask.device) == 'cpu', f"Expected mask on cpu, got {seq_cpu.block_mask.device}"
        print(f"   ✓ Block mask created on CPU: {seq_cpu.block_mask.device}")

        # Test default device (should be CPU)
        seq_default = SequenceForDiffusionLM(
            token_ids=[1, 2, 3],
            sampling_params=DiffulexSamplingParams(),
            config=config
        )
        assert seq_default.device == 'cpu', f"Expected default device='cpu', got {seq_default.device}"
        print(f"   ✓ Default device is CPU: {seq_default.device}")

        # Test serialization/deserialization
        state = seq_cpu.__getstate__()
        assert 'device' in state, "device should be in serialized state"
        assert state['device'] == 'cpu', f"Expected device='cpu' in state, got {state['device']}"
        print(f"   ✓ Device serialization works")

        seq_restored = SequenceForDiffusionLM(
            token_ids=[1, 2, 3],
            sampling_params=DiffulexSamplingParams(),
            config=config
        )
        seq_restored.__setstate__(state)
        assert seq_restored.device == 'cpu', f"Expected restored device='cpu', got {seq_restored.device}"
        print(f"   ✓ Device deserialization works")

    except Exception as e:
        print(f"   ✗ Sequence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Verify no hardcoded CUDA references
    print("\n4. Verifying CUDA hardcoding removed...")
    try:
        import inspect
        source = inspect.getsource(SequenceForDiffusionLM.update_block_mask)

        if 'torch.cuda.current_device()' in source:
            print("   ✗ Found hardcoded torch.cuda.current_device() in update_block_mask")
            return False
        else:
            print("   ✓ No hardcoded CUDA device in update_block_mask")

        source = inspect.getsource(SequenceForDiffusionLM.__setstate__)
        if 'torch.cuda.current_device()' in source:
            print("   ✗ Found hardcoded torch.cuda.current_device() in __setstate__")
            return False
        else:
            print("   ✓ No hardcoded CUDA device in __setstate__")

    except Exception as e:
        print(f"   ⚠ Could not verify source code: {e}")

    print("\n" + "=" * 80)
    print("✓ All CPU support tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_cpu_support()
    sys.exit(0 if success else 1)
