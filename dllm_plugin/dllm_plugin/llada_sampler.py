"""
Reference LLaDA generation sampler.

This module implements the official LLaDA generation algorithm as described
in the paper and reference implementation, replacing the Diffulex AutoSampler.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply Gumbel noise for categorical sampling.

    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity but reduces generation quality. Thus, we use float64.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size)
        temperature: Sampling temperature. If 0, returns logits unchanged.

    Returns:
        Logits with Gumbel noise applied
    """
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Precompute the number of tokens to transition at each step.

    LLaDA uses a linear noise schedule (Eq. 8 in paper), so the expected number
    of tokens transitioned at each step should be consistent.

    Args:
        mask_index: Boolean tensor of shape (batch_size, seq_len) indicating masked positions
        steps: Number of diffusion steps

    Returns:
        Tensor of shape (batch_size, steps) with number of tokens to transition at each step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps,
        device=mask_index.device,
        dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


class LLaDASampler:
    """
    Reference implementation of LLaDA generation algorithm.

    This sampler implements the official LLaDA generation algorithm with:
    - Gumbel noise sampling
    - Confidence-based remasking
    - Linear noise schedule
    - Block-based generation
    """

    def __init__(
        self,
        mask_token_id: int = 126336,
        temperature: float = 0.0,
        remasking: str = 'low_confidence',
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
    ):
        """
        Initialize LLaDA sampler.

        Args:
            mask_token_id: Token ID for [MASK] (default: 126336 for LLaDA)
            temperature: Sampling temperature (0 for deterministic)
            remasking: Remasking strategy ('low_confidence' or 'random')
            logits_eos_inf: Whether to set EOS logits to -inf
            confidence_eos_eot_inf: Whether to set EOS/EoT confidence to -inf
        """
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.remasking = remasking
        self.logits_eos_inf = logits_eos_inf
        self.confidence_eos_eot_inf = confidence_eos_eot_inf

        logger.info(
            f"Initialized LLaDASampler with temperature={temperature}, "
            f"remasking={remasking}"
        )

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
    ) -> torch.Tensor:
        """
        Generate text using LLaDA diffusion algorithm.

        Args:
            model: The LLaDA model
            prompt_ids: Input prompt tensor of shape (batch_size, prompt_len)
            attention_mask: Optional attention mask
            steps: Total number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block size for generation

        Returns:
            Generated token IDs of shape (batch_size, prompt_len + gen_length)
        """
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        device = prompt_ids.device

        # Initialize sequence with masks
        x = torch.full(
            (batch_size, prompt_len + gen_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        x[:, :prompt_len] = prompt_ids.clone()

        # Extend attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
            ], dim=-1)

        prompt_index = (x != self.mask_token_id)

        # Block-based generation
        assert gen_length % block_length == 0, \
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, \
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks

        logger.info(
            f"Generating {gen_length} tokens in {num_blocks} blocks, "
            f"{steps_per_block} steps per block"
        )

        # Generate each block
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            # Get mask indices for this block
            block_mask_index = (x[:, block_start:block_end] == self.mask_token_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            logger.debug(f"Processing block {block_idx + 1}/{num_blocks}")

            # Diffusion steps for this block
            for step_idx in range(steps_per_block):
                mask_index = (x == self.mask_token_id)

                # Get model logits
                # Note: vLLM models require positions parameter and return logits directly
                positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
                logits = model(x, positions)

                # Optional: Set EOS logits to -inf
                if self.logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                # Apply Gumbel noise and sample
                logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # (batch_size, seq_len)

                # Compute confidence for remasking
                if self.remasking == 'low_confidence':
                    # Optional: Set EOS/EoT confidence to -inf
                    if self.confidence_eos_eot_inf:
                        logits_with_noise[:, :, 126081] = -torch.inf  # EOS
                        logits_with_noise[:, :, 126348] = -torch.inf  # EoT

                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                        -1
                    )  # (batch_size, seq_len)

                elif self.remasking == 'random':
                    x0_p = torch.rand((batch_size, x.shape[1]), device=device)
                else:
                    raise ValueError(f"Unknown remasking strategy: {self.remasking}")

                # Don't unmask tokens beyond current block
                x0_p[:, block_end:] = -float('inf')

                # Only consider masked positions
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float('inf'))

                # Select tokens to unmask based on confidence
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                for b in range(batch_size):
                    k = num_transfer_tokens[b, step_idx].item()
                    if k > 0:
                        _, select_index = torch.topk(confidence[b], k=k)
                        transfer_index[b, select_index] = True

                # Unmask selected tokens
                x[transfer_index] = x0[transfer_index]

                if step_idx % 10 == 0:
                    logger.debug(
                        f"Block {block_idx + 1}, Step {step_idx + 1}/{steps_per_block}: "
                        f"Unmasked {transfer_index.sum().item()} tokens"
                    )

        logger.info("Generation complete")
        return x

    def __call__(self, *args, **kwargs):
        """Allow calling sampler as a function."""
        return self.generate(*args, **kwargs)
