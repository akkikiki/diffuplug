"""
vLLM-compatible adapter for the LLaDA diffusion language model.
"""

from typing import Iterable, Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

# Patch torch.distributed for CPU single-process compatibility
if not torch.distributed.is_initialized():
    class MockDistributed:
        @staticmethod
        def get_world_size():
            return 1

        @staticmethod
        def get_rank():
            return 0

    torch.distributed.get_world_size = MockDistributed.get_world_size
    torch.distributed.get_rank = MockDistributed.get_rank

# Import the original LLaDA model from Diffulex
from d2f_engine.models.llada import LLaDAForDiffusionLM
from d2f_engine.models.config.llada.configuration_llada import LLaDAConfig


class LLaDAForDiffusionLMVLLM(nn.Module):
    """
    vLLM-compatible wrapper for LLaDA diffusion language model.

    This adapter class wraps the Diffulex LLaDA implementation to make it
    compatible with vLLM's model interface.
    """

    # Packed modules mapping from the original LLaDA model
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Ensure the config is a LLaDAConfig instance
        if not isinstance(config, LLaDAConfig):
            # Convert HF config to LLaDAConfig
            config = LLaDAConfig(**config.to_dict())

        self.config = config

        # Initialize the original LLaDA model
        self.model = LLaDAForDiffusionLM(config)

        # Store logit scale for manual application
        # We don't use LogitsProcessor because we compute logits ourselves
        # and LogitsProcessor expects lm_head with quant_method
        self.logit_scale = getattr(config, "logit_scale", 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass adapter for vLLM.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Not used for LLaDA (for pipeline parallelism)
            inputs_embeds: Not used for LLaDA

        Returns:
            Hidden states from the model
        """
        import logging
        logger = logging.getLogger('dllm_plugin.models.llada')
        logger.debug(
            f"LLaDA forward: input_ids shape={input_ids.shape}, "
            f"positions shape={positions.shape}, device={input_ids.device}"
        )
        
        # LLaDA model doesn't use causal masking (full attention for diffusion)
        # The mask parameter in the original implementation is None for full attention
        mask = None

        # Call the original LLaDA model's forward method
        logger.debug("Calling LLaDA model forward...")
        hidden_states = self.model(input_ids, positions, mask)
        logger.debug(f"LLaDA forward completed: hidden_states shape={hidden_states.shape}")

        # DEBUG: Check hidden states for NaN/Inf right after forward
        logger.debug(f"LLaDA forward output: hidden_states min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
        logger.debug(f"LLaDA forward output: has NaN={torch.isnan(hidden_states).any()}, has Inf={torch.isinf(hidden_states).any()}")

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Args:
            hidden_states: Hidden states from the model

        Returns:
            Logits over the vocabulary
        """
        import logging
        logger = logging.getLogger('dllm_plugin.models.llada')
        logger.debug(f"LLaDA compute_logits: hidden_states shape={hidden_states.shape}")

        # DEBUG: Check hidden states for NaN/Inf
        logger.debug(f"LLaDA compute_logits: hidden_states min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
        logger.debug(f"LLaDA compute_logits: hidden_states has NaN={torch.isnan(hidden_states).any()}, has Inf={torch.isinf(hidden_states).any()}")

        # Use the original model's compute_logits method
        logger.debug("Calling LLaDA model compute_logits...")
        logits = self.model.compute_logits(hidden_states)
        logger.debug(f"LLaDA compute_logits completed: logits shape={logits.shape}")

        # Apply logit scaling if needed (we don't use LogitsProcessor since
        # we compute logits ourselves and LogitsProcessor expects quant_method)
        if self.logit_scale != 1.0:
            logits = logits * self.logit_scale
        logger.debug(f"LLaDA logits after scaling: logits shape={logits.shape}")

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterator of (name, tensor) pairs

        Returns:
            Set of loaded parameter names
        """
        import logging
        logger = logging.getLogger('dllm_plugin.models.llada')

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Debug: Print first few parameter names
        print("\n=== LLaDA Model Parameters (first 10) ===")
        for i, (name, _) in enumerate(self.named_parameters()):
            if i < 10:
                print(f"  Model param: {name}")
            else:
                break

        # Debug: Print all model parameters containing lm_head
        print("\n=== Model Parameters (lm_head related) ===")
        lm_head_params = [name for name in params_dict.keys() if "lm_head" in name]
        if lm_head_params:
            for name in lm_head_params:
                print(f"  Model param: {name}")
        else:
            print("  No lm_head parameters found in model!")

        # Convert iterable to list to allow inspection
        weights_list = list(weights)

        # Debug: Print first few checkpoint weight names
        print("\n=== Checkpoint Weights (first 10) ===")
        for i, (name, _) in enumerate(weights_list[:10]):
            print(f"  Checkpoint: {name}")

        # Debug: Print all checkpoint weights containing ff_out or lm_head
        print("\n=== Checkpoint Weights (ff_out/lm_head related) ===")
        ff_out_weights = [name for name, _ in weights_list if "ff_out" in name or "lm_head" in name]
        if ff_out_weights:
            for name in ff_out_weights:
                print(f"  Checkpoint: {name}")
        else:
            print("  No ff_out or lm_head weights found in checkpoint!")

        # Stacked parameters mapping for LLaDA
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights_list:
            # Debug: Track attention projection weights
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'qkv_proj']):
                logger.debug(f"\n=== Processing attention projection: {name} ===")
                logger.debug(f"  Weight shape: {loaded_weight.shape}")
                logger.debug(f"  Weight stats: mean={loaded_weight.mean():.6f}, std={loaded_weight.std():.6f}")
                logger.debug(f"  Weight range: min={loaded_weight.min():.6f}, max={loaded_weight.max():.6f}")
                logger.debug(f"  Has NaN: {torch.isnan(loaded_weight).any()}, Has Inf: {torch.isinf(loaded_weight).any()}")

            # Debug: Track transformer-level ff_out specifically
            if name == "model.transformer.ff_out.weight":
                logger.debug(f"\n=== Processing transformer-level ff_out ===")
                logger.debug(f"  Checkpoint weight name: {name}")
                logger.debug(f"  Weight shape: {loaded_weight.shape}")

            # Skip position embeddings if present
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Map checkpoint names to model parameter names
            # Checkpoint format: model.transformer.blocks.X...
            # Model format: model.model.transformer.blocks.X...
            # (LLaDA uses 'blocks' not 'layers', unlike Dream)
            mapped_name = name

            # Handle transformer-level ff_out -> lm_head mapping FIRST (before other mappings)
            # This is critical because transformer.ff_out should map to lm_head, not mlp.down_proj
            # Checkpoint might have: transformer.ff_out, model.transformer.ff_out, model.model.transformer.ff_out
            # Model expects: model.lm_head.weight
            is_transformer_ff_out = False
            if "transformer.ff_out.weight" in mapped_name:
                is_transformer_ff_out = True
                # Map various checkpoint patterns to model.lm_head.weight
                if "model.model.transformer.ff_out.weight" in mapped_name:
                    mapped_name = mapped_name.replace("model.model.transformer.ff_out.weight", "model.lm_head.weight")
                elif "model.transformer.ff_out.weight" in mapped_name:
                    mapped_name = mapped_name.replace("model.transformer.ff_out.weight", "model.lm_head.weight")
                elif mapped_name.endswith("transformer.ff_out.weight"):
                    mapped_name = "model.lm_head.weight"
                else:
                    # Fallback: replace transformer.ff_out.weight with model.lm_head.weight
                    mapped_name = mapped_name.replace("transformer.ff_out.weight", "model.lm_head.weight")

            # Handle direct lm_head mapping (checkpoint might already have lm_head)
            # Map checkpoint lm_head to model.lm_head
            if "lm_head.weight" in mapped_name and not is_transformer_ff_out:
                if mapped_name == "lm_head.weight":
                    mapped_name = "model.lm_head.weight"
                elif mapped_name == "model.lm_head.weight":
                    # Already correct, keep as is
                    pass
                elif mapped_name.startswith("model.") and "lm_head.weight" in mapped_name:
                    # Has model prefix but might be model.model.lm_head or similar
                    # Normalize to model.lm_head.weight
                    mapped_name = "model.lm_head.weight"

            # Step 1: Add the extra 'model.' prefix if needed (but skip if already mapped to lm_head)
            # Checkpoint: model.transformer.X -> Model: model.model.transformer.X
            if not mapped_name.startswith("model.lm_head") and mapped_name.startswith("model.transformer.") and not mapped_name.startswith("model.model."):
                mapped_name = "model." + mapped_name

            # Step 2: Use the packed_modules_mapping from D2fEngine LLaDA model
            # This mapping is defined in the original model and tells us how to map
            # checkpoint parameter names to model parameter names
            # NOTE: Skip ff_out mapping if it's transformer-level (already handled above)
            weight_map = {
                "q_proj": "self_attn.q_proj",
                "k_proj": "self_attn.k_proj",
                "v_proj": "self_attn.v_proj",
                "attn_out": "self_attn.o_proj",
                "attn_norm": "input_layernorm",
                "ff_norm": "post_attention_layernorm",
                "ff_proj": "mlp.gate_proj",
                "up_proj": "mlp.up_proj",
                "ff_out": "mlp.down_proj",  # Only for layer-level ff_out, not transformer-level
            }

            # Apply mappings for layer-level parameters (handle both .weight and .bias)
            # Skip if this is transformer-level ff_out (already mapped to lm_head)
            if not is_transformer_ff_out:
                for checkpoint_name, model_name in weight_map.items():
                    for suffix in [".weight", ".bias"]:
                        pattern = f".{checkpoint_name}{suffix}"
                        if pattern in mapped_name:
                            mapped_name = mapped_name.replace(pattern, f".{model_name}{suffix}")
                            break

            # Debug: Print mapping for ff_out/lm_head related weights
            if "ff_out" in name or "lm_head" in name:
                print(f"  Mapping checkpoint weight: '{name}' -> '{mapped_name}'")
                if name == "model.transformer.ff_out.weight":
                    print(f"    is_transformer_ff_out: {is_transformer_ff_out}")
                    print(f"    Target param exists in model: {mapped_name in params_dict}")

            # Try to match stacked parameters
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue

                test_name = mapped_name.replace(weight_name, param_name)

                # Try multiple naming patterns for stacked params
                candidates = [
                    test_name,
                    f"model.{test_name}",
                    test_name.replace("model.", "model.model.", 1) if test_name.startswith("model.") else None,
                ]
                candidates = [c for c in candidates if c is not None]

                for candidate_name in candidates:
                    if candidate_name in params_dict:
                        param = params_dict[candidate_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(candidate_name)
                        matched = True
                        if "ff_out" in name or "lm_head" in name:
                            print(f"    -> Loaded as stacked param: '{candidate_name}'")
                        break

                if matched:
                    break

            if not matched:
                # Handle regular parameters with multiple naming patterns
                candidates = [
                    mapped_name,                    # Exact match
                    f"model.{mapped_name}",        # Add wrapper prefix
                    mapped_name.replace("model.", "model.model.", 1) if mapped_name.startswith("model.") else None,  # Fix double model prefix
                ]
                candidates = [c for c in candidates if c is not None]

                loaded = False
                for candidate_name in candidates:
                    if candidate_name in params_dict:
                        param = params_dict[candidate_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(candidate_name)
                        loaded = True
                        if "ff_out" in name or "lm_head" in name:
                            print(f"    -> Loaded as regular param: '{candidate_name}'")
                            if name == "model.transformer.ff_out.weight":
                                print(f"    SUCCESS: Transformer-level ff_out loaded into {candidate_name}")
                                print(f"    Weight shape: {loaded_weight.shape}")
                        break

                if not loaded:
                    # Only show first 5 unmapped weights to avoid spam
                    unmapped_count = sum(1 for p in loaded_params if "Warning" not in str(p))
                    if unmapped_count < 5 or "ff_out" in name or "lm_head" in name:
                        print(f"Warning: Could not map weight '{name}' (mapped to '{mapped_name}') to any model parameter")
                        if "ff_out" in name or "lm_head" in name:
                            print(f"  Available model params containing 'lm_head': {[p for p in params_dict.keys() if 'lm_head' in p]}")

        # Handle bias parameters that may not be in the checkpoint
        # If the checkpoint doesn't have bias terms, mark them as loaded anyway
        # so they keep their initialized values (typically zeros)
        for param_name in params_dict.keys():
            if param_name not in loaded_params and param_name.endswith('.bias'):
                # Check if this is an attention projection bias (q_proj, k_proj, v_proj)
                if any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']):
                    # Mark as loaded (will keep initialized value)
                    loaded_params.add(param_name)

        # Handle lm_head.weight if it's tied to embed_tokens (tied word embeddings)
        # If embed_tokens.weight was loaded and lm_head.weight is tied, mark lm_head as loaded
        if "model.lm_head.weight" not in loaded_params and "model.lm_head.weight" in params_dict:
            # Check if embed_tokens/wte was loaded (indicating tied embeddings)
            embed_loaded = any(
                "embed_tokens.weight" in p or 
                "model.embed_tokens.weight" in p or
                "wte.weight" in p or
                "model.transformer.wte.weight" in p or
                "model.model.transformer.wte.weight" in p
                for p in loaded_params
            )
            weight_tying = getattr(self.config, 'weight_tying', False)
            
            # Get checkpoint weights containing ff_out or lm_head for debugging
            checkpoint_ff_out_weights = [name for name, _ in weights_list if "ff_out" in name or "lm_head" in name]
            
            print(f"\n=== lm_head.weight Status ===")
            print(f"  In model: {'model.lm_head.weight' in params_dict}")
            print(f"  Loaded: {'model.lm_head.weight' in loaded_params}")
            print(f"  Embed loaded: {embed_loaded}")
            print(f"  Weight tying: {weight_tying}")
            
            if embed_loaded and weight_tying:
                # Mark lm_head as loaded since it's tied to embed_tokens/wte
                # The weight was already set in __init__ when weight_tying is True
                loaded_params.add("model.lm_head.weight")
                print("  -> Marked as loaded (tied to embed_tokens/wte.weight)")
            else:
                # If not tied and still not loaded, this is an error
                # The mapping above should have caught it, but let's provide helpful error info
                print(f"  ERROR: model.lm_head.weight was not loaded from checkpoint!")
                print(f"  Checkpoint weights containing 'ff_out' or 'lm_head': {checkpoint_ff_out_weights}")
                print(f"  Model expects: model.lm_head.weight")
                print(f"  This will cause a ValueError unless weight_tying is enabled and wte was loaded.")

        # Debug summary
        total_model_params = len(params_dict)
        total_loaded = len(loaded_params)
        print(f"\n=== Weight Loading Summary ===")
        print(f"Total model parameters: {total_model_params}")
        print(f"Successfully loaded: {total_loaded}")
        print(f"Not loaded: {total_model_params - total_loaded}")
        if total_loaded < total_model_params:
            print(f"\nNote: {total_model_params - total_loaded} parameters were not in checkpoint and will use initialized values")

        # Verify lm_head weights were loaded correctly
        print(f"\n=== Verifying lm_head Weights ===")
        if hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'weight'):
            lm_head_weight = self.model.lm_head.weight
            print(f"  lm_head.weight shape: {lm_head_weight.shape}")
            print(f"  lm_head.weight mean: {lm_head_weight.mean():.6f}")
            print(f"  lm_head.weight std: {lm_head_weight.std():.6f}")
            print(f"  lm_head.weight min: {lm_head_weight.min():.6f}")
            print(f"  lm_head.weight max: {lm_head_weight.max():.6f}")
            print(f"  lm_head.weight has NaN: {torch.isnan(lm_head_weight).any()}")
            print(f"  lm_head.weight has Inf: {torch.isinf(lm_head_weight).any()}")
        else:
            print(f"  ERROR: model.lm_head.weight not found!")

        # Verify attention projection weights were loaded correctly (check first layer)
        print(f"\n=== Verifying Attention Projection Weights (Layer 0) ===")
        try:
            # Access via self.model.model.transformer["blocks"] (ModuleDict)
            first_block = self.model.model.transformer["blocks"][0]
            if hasattr(first_block, 'self_attn'):
                attn = first_block.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                    if hasattr(attn, proj_name):
                        proj = getattr(attn, proj_name)
                        # Handle ColumnParallelLinear which may have .linear.weight
                        if hasattr(proj, 'weight'):
                            weight = proj.weight
                        elif hasattr(proj, 'linear') and hasattr(proj.linear, 'weight'):
                            weight = proj.linear.weight
                        else:
                            print(f"  {proj_name}: No weight attribute found")
                            continue

                        logger.debug(f"  {proj_name}.weight:")
                        logger.debug(f"    Shape: {weight.shape}")
                        logger.debug(f"    Stats: mean={weight.mean():.6f}, std={weight.std():.6f}")
                        logger.debug(f"    Range: min={weight.min():.6f}, max={weight.max():.6f}")
                        logger.debug(f"    Has NaN: {torch.isnan(weight).any()}, Has Inf: {torch.isinf(weight).any()}")
                    else:
                        print(f"  {proj_name}: Not found in self_attn")
            else:
                print(f"  ERROR: self_attn not found in first block")
        except Exception as e:
            print(f"  ERROR accessing transformer blocks: {e}")

        return loaded_params
