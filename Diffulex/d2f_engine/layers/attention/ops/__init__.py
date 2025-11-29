# Make triton-based operations optional for Mac/non-GPU environments
try:
    from d2f_engine.layers.attention.ops.triton_decode_attn_clm import causal_lm_decode_attention_fwd as causal_lm_flash_decoding
    from d2f_engine.layers.attention.ops.triton_decode_attn_dlm import diffusion_lm_flash_decoding, CHECK_ATTENTION
    from d2f_engine.layers.attention.ops.chunked_prefill_decoding_unified_kernel import chunked_prefill_paged_decode as diffusion_lm_parallel_flash_decoding
    from d2f_engine.layers.attention.ops.kv_cache_kernels import (
        store_kvcache_distinct_layout, store_kvcache_unified_layout, load_kvcache,
        CHECK_STORING, CHECK_LOADING
    )
    HAS_TRITON_OPS = True
except (ImportError, ModuleNotFoundError):
    # Provide dummy functions that raise helpful errors
    HAS_TRITON_OPS = False

    def _raise_triton_error(*args, **kwargs):
        raise RuntimeError(
            "Triton operations are not available. This requires CUDA and cannot run on Mac or CPU-only machines. "
            "Please run this code on a GPU-enabled machine."
        )

    causal_lm_flash_decoding = _raise_triton_error
    diffusion_lm_flash_decoding = _raise_triton_error
    diffusion_lm_parallel_flash_decoding = _raise_triton_error
    store_kvcache_distinct_layout = _raise_triton_error
    store_kvcache_unified_layout = _raise_triton_error
    load_kvcache = _raise_triton_error
    CHECK_ATTENTION = lambda *args, **kwargs: None
    CHECK_STORING = lambda *args, **kwargs: None
    CHECK_LOADING = lambda *args, **kwargs: None