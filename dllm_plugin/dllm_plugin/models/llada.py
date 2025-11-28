"""
vLLM-compatible adapter for the LLaDA diffusion language model.
"""

from typing import Iterable, Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import IntermediateTensors

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
            intermediate_tensors: Not used for LLaDA (for pipeline parallelism)
            inputs_embeds: Not used for LLaDA

        Returns:
            Hidden states from the model
        """
        # LLaDA model doesn't use causal masking (full attention for diffusion)
        # The mask parameter in the original implementation is None for full attention
        mask = None

        # Call the original LLaDA model's forward method
        hidden_states = self.model(input_ids, positions, mask)

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
        # Use the original model's compute_logits method
        logits = self.model.compute_logits(hidden_states)

        # Apply logits processor if needed
        logits = self.logits_processor(None, hidden_states, logits)

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

        # Stacked parameters mapping for LLaDA
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            # Skip position embeddings if present
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Try to match stacked parameters
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                matched = True
                break

            if not matched:
                # Handle regular parameters
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                else:
                    # Try to find the parameter with "model." prefix
                    prefixed_name = f"model.{name}"
                    if prefixed_name in params_dict:
                        param = params_dict[prefixed_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(prefixed_name)

        return loaded_params
