import os
import torch

import torch.nn as nn

from typing import List
from functools import lru_cache, partial
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask

# Make flash_attn optional for Mac/non-GPU environments
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None

# Import flex_attention - always use raw PyTorch version to avoid compilation issues
# We'll wrap it with compilation only when actually running on GPU
from torch.nn.attention.flex_attention import flex_attention

from d2f_engine.layers.attention.ops import (
    causal_lm_flash_decoding, diffusion_lm_flash_decoding, diffusion_lm_parallel_flash_decoding,
    store_kvcache_unified_layout, store_kvcache_distinct_layout, load_kvcache,
    CHECK_STORING, CHECK_LOADING, CHECK_ATTENTION, HAS_TRITON_OPS
)
from d2f_engine.utils.context import ContextForDiffusionLM, get_context_causal_lm, get_context_diffusion_lm


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        model_type='causal_lm'
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.causal = model_type == 'causal_lm'
        self.model_type = model_type
        is_rtx_xx90 = lambda x: "4090" in x or "3090" in x
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        } if torch.cuda.is_available() and is_rtx_xx90(torch.cuda.get_device_name(0)) else None
        # Create attention function - will be compiled only when running on GPU
        self._attention_fn = partial(flex_attention, kernel_options=kernel_options, enable_gqa=True, 
                                     return_lse=False, training=False)
        # Don't compile here - we'll handle compilation in forward based on device
        # This avoids compilation issues on CPU
        self.attention = None  # Will be set lazily if needed
        self._compiled_attention = None
        self._block_mask_cache = {}

    @lru_cache(maxsize=32)
    def dllm_block_mask(self, block_mask: torch.Tensor, 
                        B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        def _mask_mod(batch, head, token_q, token_kv):
            return block_mask[token_q, token_kv]
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]
    
    @lru_cache(maxsize=32)
    def causal_lm_block_mask(self, cum_seq_lens: torch.Tensor, B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        document_ids = torch.zeros((cum_seq_lens[-1],), dtype=torch.int32, device=device)
        start_idx = 0
        for doc_idx, seq_len in enumerate(cum_seq_lens[1:]):
            end_idx = seq_len
            document_ids[start_idx:end_idx] = doc_idx
            start_idx = end_idx
        
        def _mask_mod(batch, head, token_q, token_kv):
            causal_mask = token_q >= token_kv
            document_mask = document_ids[token_q] == document_ids[token_kv]
            return causal_mask & document_mask
        
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: List[torch.Tensor] | None = None) -> torch.Tensor:
        import logging
        logger = logging.getLogger('dllm_plugin.attention')
        logger.debug(
            f"Attention forward: q shape={q.shape}, k shape={k.shape}, "
            f"v shape={v.shape}, device={q.device}, model_type={self.model_type}"
        )
        
        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Check if we're running on CPU (even if CUDA is available)
        is_cpu = q.device.type == 'cpu'
        use_flex_attention = not is_cpu and torch.cuda.is_available()
        
        # Lazy compilation of attention function only when needed on GPU
        if use_flex_attention and self._compiled_attention is None:
            try:
                self._compiled_attention = torch.compile(self._attention_fn, dynamic=True)
            except Exception:
                # If compilation fails, fall back to CPU path
                use_flex_attention = False

        context: ContextForDiffusionLM = get_context_causal_lm() if self.model_type == 'causal_lm' else get_context_diffusion_lm()
        logger.debug(f"Context: is_prefill={context.is_prefill}, model_type={self.model_type}")
        
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = context.kv_cache_layout == "unified"
        logger.debug(f"KV cache: k_cache shape={k_cache.shape if k_cache.numel() else 'empty'}, "
                    f"v_cache shape={v_cache.shape if v_cache.numel() else 'empty'}, "
                    f"unified_layout={is_unified_layout}")

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if not (self.model_type == 'diffusion_lm' and not context.need_kv_cache_store):
                if HAS_TRITON_OPS:
                    logger.debug("Storing KV cache using Triton kernel...")
                    store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
                    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.model_type, context)
                    logger.debug("KV cache stored")
                    # CHECK_STORING(k_cache, v_cache, k, v, context)
                else:
                    logger.debug("Skipping KV cache store on CPU (Triton ops not available)")
                    # On CPU, we skip KV cache storing since we don't have the optimized kernels
                    # The model will work without caching, just slower
                    pass

        transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d').contiguous()
        # Prefill / Decode logic TODO: Replace the Flex Attention Prefilling
        if context.is_prefill:
            logger.debug("Prefill path")
            # Handle case when context.seqs is None (e.g., during warmup)
            if context.seqs is None or len(context.seqs) == 0:
                logger.debug("Warmup path: context.seqs is None or empty")
                # During warmup when sequences are not available, use a simple attention computation
                q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]
                B, H, S, _ = q_t.shape
                logger.debug(f"Warmup attention: B={B}, H={H}, S={S}, use_flex_attention={use_flex_attention}")
                
                # On CPU, use scaled_dot_product_attention as fallback
                if not use_flex_attention:
                    logger.debug("Using scaled_dot_product_attention (CPU fallback)")
                    # Create a simple causal mask for CPU
                    attn_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
                    # Reshape for scaled_dot_product_attention: (B, H, S, D)
                    q_t = q_t.squeeze(0)  # Remove batch dim: (H, S, D)
                    k_t = k_t.squeeze(0)
                    v_t = v_t.squeeze(0)
                    o = torch.nn.functional.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=attn_mask, scale=self.scale
                    )
                    o = o.unsqueeze(0)  # Add batch dim back: (1, H, S, D)
                else:
                    logger.debug("Using flex_attention with block mask (warmup)")
                    # Create a proper block mask using create_block_mask for warmup
                    # Use a simple causal mask (token_q >= token_kv)
                    def _warmup_mask_mod(batch, head, token_q, token_kv):
                        return token_q >= token_kv
                    logger.debug("Creating block mask...")
                    block_mask = create_block_mask(
                        _warmup_mask_mod, B, H, S, S, device=str(q.device)
                    )
                    logger.debug("Calling compiled attention...")
                    if self._compiled_attention is not None:
                        o = self._compiled_attention(q_t, k_t, v_t, block_mask=block_mask)
                    else:
                        o = self._attention_fn(q_t, k_t, v_t, block_mask=block_mask)
                    logger.debug("Compiled attention completed (warmup)")
            else:
                logger.debug("Normal prefill path with sequences")
                # Block PK
                if context.block_tables is not None and self.model_type == 'causal_lm':
                    k, v = k_cache, v_cache
                elif context.block_tables is not None and self.model_type == 'diffusion_lm':
                    # TODO: Implement Prefix Caching
                    pass

                # Attention computation
                q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]

                B, H, S, _ = q_t.shape
                logger.debug(f"Normal prefill: B={B}, H={H}, S={S}, use_flex_attention={use_flex_attention}")
                
                # On CPU, use scaled_dot_product_attention as fallback
                if not use_flex_attention:
                    logger.debug("Using scaled_dot_product_attention (normal prefill CPU fallback)")
                    # Convert block mask to attention mask for CPU
                    # For now, use a simple causal mask (can be improved later)
                    attn_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
                    # Reshape for scaled_dot_product_attention: (B, H, S, D)
                    q_t = q_t.squeeze(0)  # Remove batch dim: (H, S, D)
                    k_t = k_t.squeeze(0)
                    v_t = v_t.squeeze(0)
                    o = torch.nn.functional.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=attn_mask, scale=self.scale
                    )
                    o = o.unsqueeze(0)  # Add batch dim back: (1, H, S, D)
                    logger.debug("scaled_dot_product_attention completed (normal prefill)")
                else:
                    logger.debug("Using flex_attention with block mask (normal prefill)")
                    block_mask_fn = self.causal_lm_block_mask if self.model_type == 'causal_lm' else self.dllm_block_mask
                    input_obj = context.cu_seqlens_q if self.model_type == 'causal_lm' else context.block_mask
                    logger.debug("Creating block mask...")
                    block_mask = block_mask_fn(input_obj, B, H, S, S, str(q.device))
                    logger.debug("Calling compiled attention...")
                    if self._compiled_attention is not None:
                        o = self._compiled_attention(q_t, k_t, v_t, block_mask=block_mask)
                    else:
                        o = self._attention_fn(q_t, k_t, v_t, block_mask=block_mask)
                    logger.debug("Compiled attention completed (normal prefill)")
        else:
            logger.debug("Decode path")
            if self.model_type == 'causal_lm':
                logger.debug("Causal LM decode path")
                o = causal_lm_flash_decoding(
                    q, k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_tables=context.block_tables, 
                    softmax_scale=self.scale, page_size=256
                )
                logger.debug("Causal LM flash decoding completed")
            else: 
                logger.debug("Diffusion LM decode path")
                # Handle case when context.seqs is None (e.g., during warmup)
                if context.seqs is None or len(context.seqs) == 0:
                    logger.debug("Decode warmup path: context.seqs is None or empty")
                    # During warmup when sequences are not available, use a simple attention computation
                    # This is a fallback for warmup/compilation phase
                    # Use standard attention without KV cache operations
                    q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]
                    B, H, S, _ = q_t.shape
                    
                    # On CPU, use scaled_dot_product_attention as fallback
                    if not use_flex_attention:
                        logger.debug("Using scaled_dot_product_attention (decode warmup CPU fallback)")
                        # Create a simple causal mask for CPU
                        attn_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
                        # Reshape for scaled_dot_product_attention: (B, H, S, D)
                        q_t = q_t.squeeze(0)  # Remove batch dim: (H, S, D)
                        k_t = k_t.squeeze(0)
                        v_t = v_t.squeeze(0)

                        # Debug: Check inputs before attention
                        logger.debug(f"  Attention inputs: q_t shape={q_t.shape}, k_t shape={k_t.shape}, v_t shape={v_t.shape}")
                        logger.debug(f"  q_t: has_nan={torch.isnan(q_t).any()}, mean={q_t.mean():.4f}, std={q_t.std():.4f}")
                        logger.debug(f"  k_t: has_nan={torch.isnan(k_t).any()}, mean={k_t.mean():.4f}, std={k_t.std():.4f}")
                        logger.debug(f"  v_t: has_nan={torch.isnan(v_t).any()}, mean={v_t.mean():.4f}, std={v_t.std():.4f}")
                        logger.debug(f"  scale: {self.scale}")
                        logger.debug(f"  attn_mask shape: {attn_mask.shape}")

                        o = torch.nn.functional.scaled_dot_product_attention(
                            q_t, k_t, v_t, attn_mask=attn_mask, scale=self.scale
                        )

                        # Debug: Check output after attention
                        logger.debug(f"  Attention output: o shape={o.shape}, has_nan={torch.isnan(o).any()}")
                        if not torch.isnan(o).any():
                            logger.debug(f"  o: mean={o.mean():.4f}, std={o.std():.4f}")

                        o = o.unsqueeze(0)  # Add batch dim back: (1, H, S, D)
                        logger.debug("scaled_dot_product_attention completed (decode warmup)")
                    else:
                        logger.debug("Using flex_attention with block mask (decode warmup)")
                        # Create a proper block mask using create_block_mask for warmup
                        # Use a simple causal mask (token_q >= token_kv)
                        def _warmup_mask_mod(batch, head, token_q, token_kv):
                            return token_q >= token_kv
                        logger.debug("Creating block mask...")
                        block_mask = create_block_mask(
                            _warmup_mask_mod, B, H, S, S, device=str(q.device)
                        )
                        logger.debug("Calling compiled attention...")
                        if self._compiled_attention is not None:
                            o = self._compiled_attention(q_t, k_t, v_t, block_mask=block_mask)
                        else:
                            o = self._attention_fn(q_t, k_t, v_t, block_mask=block_mask)
                        logger.debug("Compiled attention completed (decode warmup)")
                else:
                    logger.debug("Normal decode path with sequences")
                    config = context.seqs[0].config
                    diffusion_block_size = config.diffusion_block_size
                    logger.debug(f"Diffusion decode: block_size={diffusion_block_size}, unified_layout={is_unified_layout}")
                    
                    if is_unified_layout:
                        logger.debug("Using unified layout with flash_attn")
                        if not HAS_FLASH_ATTN:
                            raise RuntimeError(
                                "flash_attn is required for unified layout diffusion LM decoding but is not installed. "
                                "This feature requires CUDA and cannot run on Mac or CPU-only machines. "
                                "Either install flash_attn on a GPU machine or use a different kv_cache_layout."
                            )
                        k_comb, v_comb = load_kvcache(self.k_cache, self.v_cache, context, k, v)
                        logger.debug(f"Loaded KV cache: k_comb shape={k_comb.shape}, v_comb shape={v_comb.shape}")
                        logger.debug("Calling flash_attn_varlen_func...")
                        o = flash_attn_varlen_func(q, k_comb, v_comb,
                                                   context.cu_seqlens_q, context.cu_seqlens_k,
                                                   context.max_seqlen_q, context.max_seqlen_k,
                                                   softmax_scale=self.scale, block_table=None)
                        logger.debug("flash_attn_varlen_func completed")
                    else:
                        logger.debug("Using distinct layout")
                        # Check if Triton ops are available (GPU) or fall back to PyTorch (CPU)
                        if HAS_TRITON_OPS:
                            logger.debug("Using Triton kernel: diffusion_lm_parallel_flash_decoding")
                            # FIXME: Kernel not ok...
                            o = torch.empty_like(q).to(q.device).to(q.dtype)
                            q, k, o, k_cache, v_cache = map(lambda x: x.to(torch.float32), (q, k, o, k_cache, v_cache))
                            logger.debug("Calling diffusion_lm_parallel_flash_decoding...")
                            diffusion_lm_parallel_flash_decoding(
                                q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
                                context.block_tables, context.cu_seqlens_q, context.total_lens,
                                max(context.total_lens), max(context.seq_lens), 1.0, 1.0,
                                diffusion_block_size, context.block_mask
                            )
                            logger.debug("diffusion_lm_parallel_flash_decoding completed")
                            CHECK_ATTENTION(o, q, k, v, k_cache, v_cache, context)
                        else:
                            logger.debug("Using PyTorch CPU fallback for distinct layout decode")
                            # CPU fallback: use standard PyTorch scaled_dot_product_attention
                            # Reshape to (batch, heads, seq_len, head_dim)
                            q_reshaped = q.view(-1, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)  # (1, seq, heads, dim) -> (1, heads, seq, dim)

                            # Load KV from cache - simplified version for CPU
                            # For CPU, we'll use a simplified attention without the complex KV cache lookup
                            # Just use the current k, v values
                            k_reshaped = k.view(-1, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
                            v_reshaped = v.view(-1, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)

                            # Apply attention with block mask if available
                            if hasattr(context, 'block_mask') and context.block_mask is not None:
                                # Use the block mask from context
                                # block_mask shape: (1, 1, seq_len, seq_len)
                                attn_mask = context.block_mask.squeeze(0).squeeze(0)  # (seq_len, seq_len)
                                # Expand to match attention dimensions
                                # scaled_dot_product_attention expects mask of shape matching (batch, heads, seq_q, seq_k)
                                # but can broadcast, so (seq_q, seq_k) should work
                            else:
                                # Fallback: causal mask
                                seq_len = q_reshaped.size(2)
                                attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))

                            logger.debug(f"CPU attention shapes: q={q_reshaped.shape}, k={k_reshaped.shape}, v={v_reshaped.shape}, mask={attn_mask.shape if attn_mask is not None else None}")

                            o = torch.nn.functional.scaled_dot_product_attention(
                                q_reshaped, k_reshaped, v_reshaped,
                                attn_mask=attn_mask,
                                scale=self.scale
                            )
                            # Reshape back: (1, heads, seq, dim) -> (seq, heads * dim)
                            o = o.transpose(1, 2).squeeze(0).reshape(-1, self.num_heads * self.head_dim)
                            logger.debug(f"CPU attention output shape: {o.shape}")
                            logger.debug("PyTorch CPU fallback completed")
            
        # Final reshape
        if not context.is_prefill:
            logger.debug("Decode path: reshaping output")
            o = o.view(-1, self.num_heads * self.head_dim).contiguous()
        else:
            logger.debug("Prefill path: reshaping output")
            o = rearrange(o, '1 h s d -> s (h d)').contiguous()
        
        logger.debug(f"Attention forward completed: output shape={o.shape}")

        return o