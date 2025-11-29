#!/usr/bin/env python3
"""
Diagnose generation issues by checking weights and intermediate outputs.
"""

import argparse
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose(model_path: str):
    """Run diagnostics on model loading and generation."""

    print("=" * 80)
    print("GENERATION DIAGNOSTICS")
    print("=" * 80)
    print(f"Model: {model_path}\n")

    # Import and register plugin
    import dllm_plugin
    dllm_plugin.register()
    logger.info("Plugin registered")

    from vllm import LLM
    from dllm_plugin import generate_with_diffusion

    # Load model
    logger.info("Loading model...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    logger.info("Model loaded successfully")

    # Get the actual model for inspection
    try:
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
            actual_model = llm.llm_engine.engine_core.model_executor.driver_worker.model_runner.model
        else:
            actual_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        logger.info(f"Model type: {type(actual_model)}")
    except Exception as e:
        logger.error(f"Could not access model: {e}")
        actual_model = None

    # Check key weights
    if actual_model is not None:
        print("\n" + "=" * 80)
        print("WEIGHT INSPECTION")
        print("=" * 80)

        try:
            # Check embedding weights
            embed_weight = actual_model.model.model.transformer['wte'].weight
            print(f"\nEmbedding weights (wte):")
            print(f"  Shape: {embed_weight.shape}")
            print(f"  Stats: mean={embed_weight.mean():.6f}, std={embed_weight.std():.6f}")
            print(f"  Range: [{embed_weight.min():.6f}, {embed_weight.max():.6f}]")
            print(f"  Has NaN: {torch.isnan(embed_weight).any()}")
            print(f"  Has Inf: {torch.isinf(embed_weight).any()}")

            # Check lm_head weights
            lm_head_weight = actual_model.lm_head.weight
            print(f"\nOutput head weights (lm_head):")
            print(f"  Shape: {lm_head_weight.shape}")
            print(f"  Stats: mean={lm_head_weight.mean():.6f}, std={lm_head_weight.std():.6f}")
            print(f"  Range: [{lm_head_weight.min():.6f}, {lm_head_weight.max():.6f}]")
            print(f"  Has NaN: {torch.isnan(lm_head_weight).any()}")
            print(f"  Has Inf: {torch.isinf(lm_head_weight).any()}")

            # Check if embeddings and lm_head are tied
            if embed_weight.data_ptr() == lm_head_weight.data_ptr():
                print(f"\n✓ Embeddings and lm_head are tied (sharing memory)")
            else:
                print(f"\n✗ Embeddings and lm_head are NOT tied")

            # Check first layer attention weights
            first_layer = actual_model.model.model.transformer['blocks'][0]
            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                proj = getattr(first_layer.self_attn, proj_name)
                if hasattr(proj, 'weight'):
                    weight = proj.weight
                elif hasattr(proj, 'linear') and hasattr(proj.linear, 'weight'):
                    weight = proj.linear.weight
                else:
                    continue

                print(f"\nLayer 0 - {proj_name}:")
                print(f"  Shape: {weight.shape}")
                print(f"  Stats: mean={weight.mean():.6f}, std={weight.std():.6f}")
                print(f"  Has NaN: {torch.isnan(weight).any()}")

        except Exception as e:
            logger.error(f"Error inspecting weights: {e}")
            import traceback
            traceback.print_exc()

    # Test generation with different settings
    print("\n" + "=" * 80)
    print("GENERATION TESTS")
    print("=" * 80)

    test_prompts = [
        "2+2=",
        "The capital of France is",
        "Hello, how are",
    ]

    # Test 1: Default settings
    print("\nTest 1: Default settings")
    print("-" * 80)
    try:
        outputs = generate_with_diffusion(
            llm,
            prompts=test_prompts[:1],
            temperature=0.2,
            max_tokens=16,
        )
        for output in outputs:
            print(f"Prompt: {test_prompts[0]}")
            print(f"Output: {output.outputs[0].text}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Higher temperature
    print("\nTest 2: Higher temperature (0.8)")
    print("-" * 80)
    try:
        outputs = generate_with_diffusion(
            llm,
            prompts=test_prompts[:1],
            temperature=0.8,
            max_tokens=16,
        )
        for output in outputs:
            print(f"Prompt: {test_prompts[0]}")
            print(f"Output: {output.outputs[0].text}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

    # Test 3: Different block sizes
    print("\nTest 3: Different block sizes (block_size=8, diffusion_block_size=16)")
    print("-" * 80)
    try:
        outputs = generate_with_diffusion(
            llm,
            prompts=test_prompts[:1],
            temperature=0.2,
            max_tokens=16,
            block_size=8,
            diffusion_block_size=16,
        )
        for output in outputs:
            print(f"Prompt: {test_prompts[0]}")
            print(f"Output: {output.outputs[0].text}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

    # Test 4: Batch generation
    print("\nTest 4: Batch generation")
    print("-" * 80)
    try:
        outputs = generate_with_diffusion(
            llm,
            prompts=test_prompts,
            temperature=0.2,
            max_tokens=16,
        )
        for prompt, output in zip(test_prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Output: {output.outputs[0].text}")
            print()
    except Exception as e:
        logger.error(f"Generation failed: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print("\nRecommendations:")
    print("1. Check if weights show reasonable statistics (mean near 0, std > 0)")
    print("2. Verify no NaN or Inf values in weights")
    print("3. Compare generation outputs - they should be coherent")
    print("4. If outputs look random, weights may not be loaded correctly")
    print("5. Run verify_weights.py to compare with original HF model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose generation issues")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    diagnose(args.model)
