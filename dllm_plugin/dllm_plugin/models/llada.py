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

# Use HuggingFace's AutoModel to load the official LLaDA model
# NOTE: This will use trust_remote_code to load the model code from HuggingFace
# The Diffulex d2f_engine model uses Llama-style architecture which doesn't match HF weights
from transformers import AutoModel


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
        self.config = config

        # Load the HuggingFace model using AutoModel with trust_remote_code
        # This will load the official LLaDA implementation from HuggingFace
        model_name = vllm_config.model_config.model

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading HuggingFace LLaDA model from {model_name}")

        # Load the model - AutoModel will use trust_remote_code to get the official implementation
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
        ).eval()

        logger.info(f"Loaded HF model type: {type(self.model)}")
        logger.info("NOTE: Using HuggingFace's official LLaDA model (NOT Diffulex d2f_engine)")

        # Store logit scale for manual application
        self.logit_scale = getattr(config, "logit_scale", 1.0)

    def load_weights(self, weights: Iterable):
        """
        Override vLLM's weight loading since we already loaded weights
        via AutoModel.from_pretrained() in __init__.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Skipping vLLM weight loading - weights already loaded by HuggingFace AutoModel")
        # No-op: weights are already loaded in self.model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass adapter for vLLM.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Not used for LLaDA (for pipeline parallelism)
            inputs_embeds: Not used for LLaDA
            attention_mask: Attention mask (optional, for diffusion models use all 1s)

        Returns:
            Hidden states from the model
        """
        import logging
        logger = logging.getLogger('dllm_plugin.models.llada')
        logger.debug(
            f"LLaDA forward: input_ids shape={input_ids.shape}, device={input_ids.device}"
        )

        # Track if input was originally 1D so we can squeeze output back
        input_was_1d = input_ids.dim() == 1

        # Ensure input_ids is 2D (batch_size, seq_len)
        # vLLM sometimes passes 1D tensors during warmup
        if input_was_1d:
            input_ids = input_ids.unsqueeze(0)
            logger.debug(f"Unsqueezed input_ids to shape: {input_ids.shape}")

        # Ensure attention_mask is also 2D if provided
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Call the HuggingFace LLaDA model
        # It returns CausalLMOutputWithPast with .logits and .hidden_states
        # Note: HF model doesn't use 'positions', it handles positional encoding internally
        logger.debug("Calling HuggingFace LLaDA model forward...")
        output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Cache the logits so we don't have to recompute them in compute_logits()
        self._cached_logits = output.logits

        # Extract hidden states from the output
        # Return the last hidden state (same as what Diffulex model returned)
        # hidden_states is a tuple of tensors (one per layer), get the last one
        hidden_states = output.hidden_states[-1]

        # If we unsqueezed input, squeeze output back to match original shape
        if input_was_1d:
            hidden_states = hidden_states.squeeze(0)
            self._cached_logits = self._cached_logits.squeeze(0)
            logger.debug(f"Squeezed outputs back - hidden_states shape: {hidden_states.shape}, logits shape: {self._cached_logits.shape}")

        logger.debug(f"LLaDA forward completed: hidden_states shape={hidden_states.shape}")
        logger.debug(f"LLaDA forward logits shape: {output.logits.shape}")

        # DEBUG: Check hidden states and logits for NaN/Inf
        logger.debug(f"LLaDA forward output: hidden_states min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
        logger.debug(f"LLaDA forward output: has NaN={torch.isnan(hidden_states).any()}, has Inf={torch.isinf(hidden_states).any()}")
        logger.debug(f"LLaDA forward logits: min={output.logits.min():.4f}, max={output.logits.max():.4f}, mean={output.logits.mean():.4f}")

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

        # Return the cached logits from the forward pass
        # The HuggingFace model already computed logits, so we just return them
        if hasattr(self, '_cached_logits'):
            logits = self._cached_logits
            logger.debug(f"LLaDA compute_logits: returning cached logits, shape={logits.shape}")
        else:
            logger.warning("No cached logits found! This shouldn't happen.")
            # Fallback: this shouldn't happen, but return zeros to avoid crash
            batch_size, seq_len, hidden_size = hidden_states.shape
            logits = torch.zeros(batch_size, seq_len, self.config.vocab_size, device=hidden_states.device)

        logger.debug(f"LLaDA compute_logits completed: logits shape={logits.shape}")

        # Apply logit scaling if needed (we don't use LogitsProcessor since
        # we compute logits ourselves and LogitsProcessor expects quant_method)
        if self.logit_scale != 1.0:
            logits = logits * self.logit_scale
        logger.debug(f"LLaDA logits after scaling: logits shape={logits.shape}")

        return logits

