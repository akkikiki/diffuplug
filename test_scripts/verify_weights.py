#!/usr/bin/env python3
"""
Verify that weights are correctly loaded in the vLLM adapter by comparing
with the original HuggingFace checkpoint.
"""

import argparse
import torch
import sys
from pathlib import Path

def compare_weights(model_path: str, layer_idx: int = 0):
    """
    Compare weights between HuggingFace model and vLLM adapter.

    Args:
        model_path: Path to the model checkpoint
        layer_idx: Which transformer layer to inspect (default: 0)
    """
    print("=" * 80)
    print("Weight Verification Script")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Inspecting layer: {layer_idx}")
    print()

    # =========================================================================
    # Step 1: Load original HuggingFace model
    # =========================================================================
    print("Step 1: Loading original HuggingFace model...")
    try:
        from transformers import AutoModelForCausalLM, AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        hf_model.eval()
        print("✓ HuggingFace model loaded successfully")
        print(f"  Model type: {type(hf_model)}")
        print(f"  Config type: {type(hf_config)}")
        if hasattr(hf_config, 'n_layer'):
            print(f"  Number of layers: {hf_config.n_layer}")
        elif hasattr(hf_config, 'num_hidden_layers'):
            print(f"  Number of layers: {hf_config.num_hidden_layers}")
        print()
    except Exception as e:
        print(f"✗ Failed to load HuggingFace model: {e}")
        print("\nNote: Will proceed with vLLM-only checks")
        import traceback
        traceback.print_exc()
        hf_model = None
        hf_config = None

    # =========================================================================
    # Step 2: Load vLLM adapter model
    # =========================================================================
    print("Step 2: Loading vLLM adapter model...")
    try:
        # Register the plugin first
        import dllm_plugin
        dllm_plugin.register()

        from vllm import LLM

        vllm_model = LLM(
            model=model_path,
            trust_remote_code=True,
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        print("✓ vLLM model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load vLLM model: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Step 3: Extract and compare specific weights
    # =========================================================================
    print("Step 3: Comparing weights...")
    print("-" * 80)

    # Get the actual model from vLLM's LLM wrapper
    # In vLLM v1, the model is in engine_core -> model_executor
    try:
        if hasattr(vllm_model, 'llm_engine') and hasattr(vllm_model.llm_engine, 'engine_core'):
            # vLLM v1 architecture
            vllm_actual_model = vllm_model.llm_engine.engine_core.model_executor.driver_worker.model_runner.model
        else:
            # Try other access patterns
            vllm_actual_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model

        print(f"vLLM model type: {type(vllm_actual_model)}")
        print()
    except Exception as e:
        print(f"✗ Failed to access vLLM model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare embedding weights
    print("\n1. Checking Embedding Weights (wte)")
    print("-" * 80)
    try:
        vllm_embed = vllm_actual_model.model.model.transformer['wte'].weight
        print(f"vLLM embedding shape: {vllm_embed.shape}")
        print(f"vLLM embedding stats: mean={vllm_embed.mean():.6f}, std={vllm_embed.std():.6f}")
        print(f"vLLM embedding range: [{vllm_embed.min():.6f}, {vllm_embed.max():.6f}]")
        print(f"Has NaN: {torch.isnan(vllm_embed).any()}, Has Inf: {torch.isinf(vllm_embed).any()}")

        if hf_model is not None:
            try:
                hf_embed = hf_model.model.transformer['wte'].weight
                print(f"\nHF embedding shape: {hf_embed.shape}")
                print(f"HF embedding stats: mean={hf_embed.mean():.6f}, std={hf_embed.std():.6f}")

                diff = (hf_embed - vllm_embed).abs()
                print(f"\nDifference:")
                print(f"  Max absolute difference: {diff.max():.6e}")
                print(f"  Mean absolute difference: {diff.mean():.6e}")

                if diff.max() < 1e-5:
                    print("✓ Embeddings match!")
                else:
                    print("✗ Embeddings differ significantly!")
            except Exception as e:
                print(f"Could not compare with HF model: {e}")
    except Exception as e:
        print(f"✗ Failed to check embeddings: {e}")
        import traceback
        traceback.print_exc()

    # Check attention weights for a specific layer
    print(f"\n2. Checking Attention Weights (Layer {layer_idx})")
    print("-" * 80)
    try:
        vllm_layer = vllm_actual_model.model.model.transformer['blocks'][layer_idx]

        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            print(f"\n{proj_name}:")
            vllm_proj = getattr(vllm_layer.self_attn, proj_name)

            # Get vLLM weights (handle different wrapper types)
            if hasattr(vllm_proj, 'weight'):
                vllm_weight = vllm_proj.weight
            elif hasattr(vllm_proj, 'linear') and hasattr(vllm_proj.linear, 'weight'):
                vllm_weight = vllm_proj.linear.weight
            else:
                print(f"  Warning: vLLM {proj_name} has no .weight attribute")
                continue

            print(f"  vLLM shape: {vllm_weight.shape}")
            print(f"  vLLM stats: mean={vllm_weight.mean():.6f}, std={vllm_weight.std():.6f}")
            print(f"  vLLM range: [{vllm_weight.min():.6f}, {vllm_weight.max():.6f}]")
            print(f"  Has NaN: {torch.isnan(vllm_weight).any()}, Has Inf: {torch.isinf(vllm_weight).any()}")

            if hf_model is not None:
                try:
                    hf_layer = hf_model.model.transformer['blocks'][layer_idx]
                    hf_proj = getattr(hf_layer.self_attn, proj_name)

                    if hasattr(hf_proj, 'weight'):
                        hf_weight = hf_proj.weight
                    else:
                        continue

                    print(f"\n  HF shape: {hf_weight.shape}")
                    print(f"  HF stats: mean={hf_weight.mean():.6f}, std={hf_weight.std():.6f}")

                    diff = (hf_weight - vllm_weight).abs()
                    print(f"\n  Difference:")
                    print(f"    Max absolute difference: {diff.max():.6e}")
                    print(f"    Mean absolute difference: {diff.mean():.6e}")

                    if diff.max() < 1e-5:
                        print(f"  ✓ {proj_name} weights match!")
                    else:
                        print(f"  ✗ {proj_name} weights differ significantly!")
                except Exception as e:
                    print(f"  Could not compare with HF model: {e}")

    except Exception as e:
        print(f"✗ Failed to check attention weights: {e}")
        import traceback
        traceback.print_exc()

    # Check output head (lm_head)
    print(f"\n3. Checking Output Head (lm_head)")
    print("-" * 80)
    try:
        vllm_lm_head = vllm_actual_model.lm_head.weight
        print(f"vLLM lm_head shape: {vllm_lm_head.shape}")
        print(f"vLLM lm_head stats: mean={vllm_lm_head.mean():.6f}, std={vllm_lm_head.std():.6f}")
        print(f"vLLM lm_head range: [{vllm_lm_head.min():.6f}, {vllm_lm_head.max():.6f}]")
        print(f"Has NaN: {torch.isnan(vllm_lm_head).any()}, Has Inf: {torch.isinf(vllm_lm_head).any()}")

        if hf_model is not None:
            try:
                hf_lm_head = hf_model.lm_head.weight
                print(f"\nHF lm_head shape: {hf_lm_head.shape}")
                print(f"HF lm_head stats: mean={hf_lm_head.mean():.6f}, std={hf_lm_head.std():.6f}")

                diff = (hf_lm_head - vllm_lm_head).abs()
                print(f"\nDifference:")
                print(f"  Max absolute difference: {diff.max():.6e}")
                print(f"  Mean absolute difference: {diff.mean():.6e}")

                if diff.max() < 1e-5:
                    print("✓ lm_head weights match!")
                else:
                    print("✗ lm_head weights differ significantly!")
            except Exception as e:
                print(f"Could not compare with HF model: {e}")
    except Exception as e:
        print(f"✗ Failed to check lm_head: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Step 4: Test forward pass
    # =========================================================================
    if hf_model is not None:
        print("\n" + "=" * 80)
        print("Step 4: Testing Forward Pass")
        print("=" * 80)
        try:
            # Create test input
            batch_size = 2
            seq_len = 10
            vocab_size = hf_config.vocab_size if hf_config else 50257

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            print(f"Test input shape: {input_ids.shape}")

            # HF forward pass
            with torch.no_grad():
                hf_output = hf_model(input_ids)
                if hasattr(hf_output, 'logits'):
                    hf_logits = hf_output.logits
                else:
                    hf_logits = hf_output

            print(f"HF output shape: {hf_logits.shape}")
            print(f"HF output stats: mean={hf_logits.mean():.6f}, std={hf_logits.std():.6f}")
            print(f"HF output range: [{hf_logits.min():.6f}, {hf_logits.max():.6f}]")

            # Note: vLLM forward pass is more complex and may not be directly comparable
            # due to different execution paths
            print("\nNote: vLLM forward pass comparison would require running through")
            print("the full generation pipeline, which is tested separately.")

        except Exception as e:
            print(f"✗ Failed forward pass test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "=" * 80)
        print("Step 4: Forward Pass Test Skipped")
        print("=" * 80)
        print("HuggingFace model not loaded - use diagnose_generation.py for generation tests")

    print("\n" + "=" * 80)
    print("Verification Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify weights are correctly loaded in vLLM adapter"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the LLaDA model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Which transformer layer to inspect (default: 0)"
    )

    args = parser.parse_args()

    compare_weights(args.model, args.layer)
