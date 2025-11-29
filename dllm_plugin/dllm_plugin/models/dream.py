"""
vLLM-compatible adapter for the Dream diffusion language model.
"""

from typing import Iterable, Optional
import os

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import IntermediateTensors

# Ensure distributed is initialized for CPU
if not torch.distributed.is_initialized():
    # Initialize distributed for single process if not already initialized
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')

# Import the original Dream model from Diffulex
from d2f_engine.models.dream import DreamForDiffusionLM
from d2f_engine.models.config.dream.configuration_dream import DreamConfig


class DreamForDiffusionLMVLLM(nn.Module):
    """
    vLLM-compatible wrapper for Dream diffusion language model.

    This adapter class wraps the Diffulex Dream implementation to make it
    compatible with vLLM's model interface.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Ensure the config is a DreamConfig instance
        if not isinstance(config, DreamConfig):
            # Convert HF config to DreamConfig
            config = DreamConfig(**config.to_dict())

        self.config = config

        # Initialize the original Dream model
        self.model = DreamForDiffusionLM(config)

        # Initialize logits processor
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, scale=logit_scale
        )

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
            intermediate_tensors: Not used for Dream (for pipeline parallelism)
            inputs_embeds: Not used for Dream

        Returns:
            Hidden states from the model
        """
        import logging
        logger = logging.getLogger('dllm_plugin.models.dream')
        logger.debug(
            f"Dream forward: input_ids shape={input_ids.shape}, "
            f"positions shape={positions.shape}, device={input_ids.device}"
        )
        
        # Dream model doesn't use causal masking (full attention for diffusion)
        # The mask parameter in the original implementation is None for full attention
        mask = None

        # Call the original Dream model's forward method
        logger.debug("Calling Dream model forward...")
        hidden_states = self.model(input_ids, positions, mask)
        logger.debug(f"Dream forward completed: hidden_states shape={hidden_states.shape}")

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
        logger = logging.getLogger('dllm_plugin.models.dream')
        logger.debug(f"Dream compute_logits: hidden_states shape={hidden_states.shape}")
        
        # Use the original model's compute_logits method
        logger.debug("Calling Dream model compute_logits...")
        logits = self.model.compute_logits(hidden_states)
        logger.debug(f"Dream compute_logits completed: logits shape={logits.shape}")

        # Apply logits processor if needed
        logits = self.logits_processor(None, hidden_states, logits)
        logger.debug(f"Dream logits after processing: logits shape={logits.shape}")

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterator of (name, tensor) pairs

        Returns:
            Set of loaded parameter names
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Debug: Print first few parameter names
        print("\n=== Dream Model Parameters (first 10) ===")
        for i, (name, _) in enumerate(self.named_parameters()):
            if i < 10:
                print(f"  Model param: {name}")
            else:
                break

        # Convert iterable to list to allow inspection
        weights_list = list(weights)

        # Debug: Print first few checkpoint weight names
        print("\n=== Checkpoint Weights (first 10) ===")
        for i, (name, _) in enumerate(weights_list[:10]):
            print(f"  Checkpoint: {name}")

        # Now process the weights
        for name, loaded_weight in weights_list:
            # Skip position embeddings if present
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Map checkpoint names to model parameter names
            # Checkpoint format: model.transformer.blocks.X... or transformer.blocks.X...
            # Model format: model.model.layers.X...
            mapped_name = name

            # Handle checkpoint naming variations
            if "transformer.blocks" in mapped_name:
                # Replace transformer.blocks with model.layers
                mapped_name = mapped_name.replace("transformer.blocks", "model.layers")

            # Handle transformer-level ff_out -> lm_head mapping (similar to LLaDA)
            # Checkpoint might have: transformer.ff_out, model.transformer.ff_out, model.model.transformer.ff_out
            # Model expects: model.lm_head.weight
            if "transformer.ff_out.weight" in mapped_name or "ff_out.weight" in mapped_name:
                # Map various checkpoint patterns to model.lm_head.weight
                if "model.model.transformer.ff_out.weight" in mapped_name:
                    mapped_name = mapped_name.replace("model.model.transformer.ff_out.weight", "model.lm_head.weight")
                elif "model.transformer.ff_out.weight" in mapped_name:
                    mapped_name = mapped_name.replace("model.transformer.ff_out.weight", "model.lm_head.weight")
                elif mapped_name.endswith("transformer.ff_out.weight"):
                    mapped_name = "model.lm_head.weight"
                elif mapped_name.endswith("ff_out.weight"):
                    mapped_name = "model.lm_head.weight"

            # Handle direct lm_head mapping (checkpoint might already have lm_head)
            # Map checkpoint lm_head to model.lm_head
            if "lm_head.weight" in mapped_name:
                if mapped_name == "lm_head.weight":
                    mapped_name = "model.lm_head.weight"
                elif mapped_name == "model.lm_head.weight":
                    # Already correct, keep as is
                    pass
                elif mapped_name.startswith("model.") and "lm_head.weight" in mapped_name:
                    # Has model prefix but might be model.model.lm_head or similar
                    # Normalize to model.lm_head.weight
                    mapped_name = "model.lm_head.weight"

            # Try multiple naming patterns
            candidates = [
                mapped_name,                    # Exact match
                f"model.{mapped_name}",        # Add wrapper prefix
                mapped_name.replace("model.", "model.model.", 1) if mapped_name.startswith("model.") else None,  # Fix double model prefix
            ]

            # Remove None values
            candidates = [c for c in candidates if c is not None]

            # Try to load with each candidate name
            loaded = False
            for candidate_name in candidates:
                if candidate_name in params_dict:
                    param = params_dict[candidate_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(candidate_name)
                    loaded = True
                    break

            if not loaded:
                # Log unmapped weights for debugging (only show first 10 to avoid spam)
                if len([p for p in loaded_params if "Warning" not in p]) < 10:
                    print(f"Warning: Could not map weight '{name}' to any model parameter")

        # Handle bias parameters that may not be in the checkpoint
        # If the checkpoint doesn't have bias terms, mark them as loaded anyway
        # so they keep their initialized values (typically zeros)
        for param_name in params_dict.keys():
            if param_name not in loaded_params and param_name.endswith('.bias'):
                # Check if this is an attention projection bias
                if any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']):
                    # Mark as loaded (will keep initialized value)
                    loaded_params.add(param_name)

        # Handle lm_head.weight if it's tied to embed_tokens (tied word embeddings)
        # If embed_tokens.weight was loaded and lm_head.weight is tied, mark lm_head as loaded
        if "model.lm_head.weight" not in loaded_params and "model.lm_head.weight" in params_dict:
            # Check if embed_tokens was loaded (indicating tied embeddings)
            embed_loaded = any("embed_tokens.weight" in p or "model.embed_tokens.weight" in p for p in loaded_params)
            if embed_loaded and getattr(self.config, 'tie_word_embeddings', False):
                # Mark lm_head as loaded since it's tied to embed_tokens
                # The weight was already set in __init__ when tie_word_embeddings is True
                loaded_params.add("model.lm_head.weight")
                print("Note: model.lm_head.weight is tied to embed_tokens.weight and marked as loaded")
            else:
                # If not tied and still not loaded, check if checkpoint has it under a different name
                # This is a fallback - the mapping above should have caught it, but just in case
                print(f"Warning: model.lm_head.weight was not loaded from checkpoint. "
                      f"Checkpoint may use a different name (e.g., transformer.ff_out.weight, ff_out.weight, or lm_head.weight)")

        # Debug summary
        total_model_params = len(params_dict)
        total_loaded = len(loaded_params)
        print(f"\n=== Weight Loading Summary ===")
        print(f"Total model parameters: {total_model_params}")
        print(f"Successfully loaded: {total_loaded}")
        print(f"Not loaded: {total_model_params - total_loaded}")
        if total_loaded < total_model_params:
            print(f"\nNote: {total_model_params - total_loaded} parameters were not in checkpoint and will use initialized values")

        return loaded_params

    @property
    def packed_modules_mapping(self):
        """Return packed modules mapping for the model."""
        return getattr(self.model, 'packed_modules_mapping', {})
