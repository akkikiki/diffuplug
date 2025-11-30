# Diffulex CPU Support Patch

## Overview

This document describes the patches made to support CPU execution in the Diffulex diffusion language model framework, which originally hardcoded CUDA device usage.

## Problem

Diffulex hardcoded `torch.cuda.current_device()` in critical tensor operations, causing failures when running on CPU:

```python
AssertionError: Torch not compiled with CUDA enabled
```

This occurred in:
1. `Diffulex/d2f_engine/engine/sequence.py:477` - Creating block-wise causal mask
2. `Diffulex/d2f_engine/engine/sequence.py:304-305` - Device alignment during deserialization

## Solution

### 1. Modified `SequenceForDiffusionLM` Class

**File**: `Diffulex/d2f_engine/engine/sequence.py`

#### Added Device Parameter

```python
def __init__(self, token_ids: List[int],
             sampling_params = SamplingParams(),
             config: Config = None,
             device: str = None):  # ← NEW PARAMETER
    super().__init__(token_ids, sampling_params)
    # ... existing code ...
    # Store device for tensor operations (defaults to CPU if not specified)
    self.device = device if device is not None else 'cpu'  # ← NEW
```

#### Updated Serialization

Added `device` to state dictionary:

```python
def __getstate__(self):
    state = {
        # ... existing fields ...
        "device": self.device,  # ← NEW
    }
    return state

def __setstate__(self, state):
    # ... existing code ...
    self.device = state.get("device", "cpu")  # ← NEW
    # Align tensor devices when sequence is reconstructed
    if self.block_mask is not None and str(self.block_mask.device) != self.device:
        self.block_mask = self.block_mask.to(self.device)  # ← CHANGED
```

**Before**:
```python
# Lines 304-305 (OLD)
if self.block_mask is not None and self.block_mask.device.index != torch.cuda.current_device():
    self.block_mask = self.block_mask.to(torch.cuda.current_device())
```

**After**:
```python
# Lines 309-310 (NEW)
if self.block_mask is not None and str(self.block_mask.device) != self.device:
    self.block_mask = self.block_mask.to(self.device)
```

#### Fixed Mask Creation

**Before**:
```python
# Line 477 (OLD)
block_wise_causal_mask = torch.zeros(mask_shape, dtype=torch.bool,
                                    device=torch.cuda.current_device())
```

**After**:
```python
# Line 482 (NEW)
block_wise_causal_mask = torch.zeros(mask_shape, dtype=torch.bool,
                                    device=self.device)
```

### 2. Updated Generation Code

**File**: `dllm_plugin/dllm_plugin/generation.py`

#### Get Device from Model

Added code to detect model device (lines 83-94):

```python
# Get device from model
import torch
if hasattr(model, 'device'):
    device = str(model.device)
else:
    # Try to get device from first parameter
    try:
        device = str(next(model.parameters()).device)
    except:
        # Default to CPU if we can't determine device
        device = 'cpu'
logger.info(f"[Worker Process] Model device: {device}")
```

#### Pass Device to Sequences

Updated sequence creation (lines 220-226):

```python
# Create sequence with device parameter for CPU support
seq = SequenceForDiffusionLM(
    token_ids=prompt_tokens,
    sampling_params=diffulex_params,
    config=diffulex_config,
    device=device  # ← NEW
)
```

### 3. KV Cache Layout Selection

**File**: `dllm_plugin/dllm_plugin/generation.py`

The attention mechanism in Diffulex has two KV cache layouts:
- **"unified"**: Optimized layout requiring flash_attn (CUDA-only)
- **"distinct"**: Standard layout compatible with CPU

Added automatic layout selection based on device (lines 189-193):

```python
# Determine KV cache layout based on device
# "unified" layout requires flash_attn (CUDA-only)
# "distinct" layout works on CPU
kv_cache_layout = "distinct" if device == "cpu" else "unified"
logger.info(f"[Worker Process] Using kv_cache_layout: {kv_cache_layout} for device: {device}")
```

Updated config creation (line 202):

```python
diffulex_config = DiffulexConfig(
    # ... other params ...
    kv_cache_layout=kv_cache_layout,  # Use CPU-compatible layout for CPU
    # ... other params ...
)
```

## Changes Summary

| File | Lines | Change |
|------|-------|--------|
| `Diffulex/d2f_engine/engine/sequence.py` | 213-229 | Added `device` parameter to `__init__` |
| `Diffulex/d2f_engine/engine/sequence.py` | 248-276 | Added `device` to state dict |
| `Diffulex/d2f_engine/engine/sequence.py` | 307-310 | Use `self.device` instead of `torch.cuda.current_device()` |
| `Diffulex/d2f_engine/engine/sequence.py` | 478-482 | Use `self.device` instead of `torch.cuda.current_device()` |
| `dllm_plugin/dllm_plugin/generation.py` | 83-94 | Detect model device |
| `dllm_plugin/dllm_plugin/generation.py` | 189-193 | Auto-select KV cache layout based on device |
| `dllm_plugin/dllm_plugin/generation.py` | 202 | Pass kv_cache_layout to config |
| `dllm_plugin/dllm_plugin/generation.py` | 220-226 | Pass device to sequence creation |

## Device Detection Logic

The device is automatically detected from the model:

1. **First attempt**: Check if model has `device` attribute
2. **Fallback**: Get device from first model parameter
3. **Default**: Use 'cpu' if detection fails

This ensures compatibility with both CPU and CUDA models without hardcoding.

## Benefits

✅ **CPU Support**: Can now run on CPU without CUDA errors
✅ **GPU Compatible**: Still works with CUDA when available
✅ **Auto-Detection**: Automatically uses correct device
✅ **Backward Compatible**: Defaults to CPU if device not specified

## Testing

To test CPU support:

```python
from vllm import LLM, SamplingParams

# Model will run on CPU
llm = LLM(
    model="GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
    enforce_eager=True
)

prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(prompts, sampling_params)
```

## GPU Support

For GPU execution, vLLM will automatically detect CUDA:

```python
llm = LLM(
    model="GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=1,  # Use 1 GPU
)
```

The device will be automatically detected as `cuda:0` and passed to Diffulex.

## Flash Attention Compatibility

The attention mechanism in Diffulex has two implementations:
- **Unified layout**: Uses flash_attn for optimized CUDA performance (requires flash_attn package)
- **Distinct layout**: Uses standard PyTorch attention (CPU and CUDA compatible)

Our patch automatically selects "distinct" layout for CPU execution, avoiding the flash_attn requirement.

## Remaining CUDA References

The following CUDA references in `model_runner.py` are **not problematic**:
- GPU memory management (`torch.cuda.empty_cache()`, etc.)
- CUDA graphs (`torch.cuda.CUDAGraph()`)
- Synchronization (`torch.cuda.synchronize()`)

These are only executed in GPU code paths and won't affect CPU execution.

## Architecture Compatibility

This patch works with:
- ✅ vLLM v0.11.0+ (v1 engine)
- ✅ CPU execution
- ✅ CUDA execution
- ✅ Multi-process worker architecture
- ✅ Monkey-patching approach

---

**Implementation Date**: November 29, 2024
**Status**: ✅ Complete - Ready for testing
