# Custom Diffusion Generation Implementation

## Overview

This document describes the implementation of custom diffusion generation for vLLM v1 engine using worker utilities and monkey-patching.

## The Challenge

**Problem**: vLLM 0.11.0+ uses a v1 engine architecture where the model runs in a separate worker process, making direct model access impossible for custom generation loops.

**Solution**: Implement custom diffusion generation as a worker utility that runs in the worker process where the model lives, then call it via RPC from the main process.

## Architecture

```
Main Process                          Worker Process
     │                                      │
     │  1. Call utility via RPC             │
     ├──────────────────────────────────────>│
     │                                      │
     │                                  2. Access model
     │                                  3. Run diffusion
     │                                  4. Return results
     │                                      │
     │<──────────────────────────────────────┤
     │  5. Convert to RequestOutput          │
     │                                      │
```

## Implementation Details

### 1. Monkey-Patching at Import Time

**File**: `dllm_plugin/__init__.py`

The `EngineCoreProc` class must be patched **before** the worker process is spawned:

```python
def _patch_engine_core():
    """Patch EngineCoreProc before worker process starts."""
    from .generation import patch_engine_core_for_diffusion
    patch_engine_core_for_diffusion()

def register():
    _patch_kv_cache_manager()
    _patch_engine_core()  # ← Must happen before LLM creation!
    _patch_llm_generation()
```

**Why at import time?**
- Worker process is spawned when `LLM()` is initialized
- The worker process has its own Python interpreter
- Monkey-patches applied after worker starts won't be visible in the worker

### 2. Custom Worker Utility Method

**File**: `dllm_plugin/generation.py`

Added `run_diffusion_generation` method to `EngineCoreProc`:

```python
def run_diffusion_generation(self, prompts: List[str],
                            sampling_params_dict: dict) -> List[dict]:
    """
    Runs in worker process with access to:
    - self.model_executor (the executor/worker)
    - self.vllm_config (configuration)
    - The actual model via worker.get_model()
    """
    # 1. Access model through executor
    executor = self.model_executor
    worker = executor.driver_worker or executor
    model = worker.get_model()

    # 2. Load tokenizer
    # 3. Create Diffulex components
    # 4. Run diffusion generation loop
    # 5. Return results as serializable dicts
```

### 3. Calling from Main Process

When diffusion model is detected:

```python
if hasattr(llm.llm_engine, 'engine_core'):
    # Call utility in worker process
    results = llm.llm_engine.engine_core.call_utility(
        'run_diffusion_generation',
        prompts,
        sampling_params_dict
    )
    # Convert results to RequestOutput format
```

### 4. Key Implementation Steps

#### A. Accessing the Model

```python
# Navigate executor → worker → model
executor = self.model_executor  # UniProcExecutor
worker = executor.driver_worker  # CPUWorker
model = worker.get_model()       # LLaDAForDiffusionLMVLLM
```

#### B. Loading Tokenizer

```python
from transformers import AutoTokenizer
model_path = self.vllm_config.model_config.model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

#### C. Getting Local Model Path

Diffulex requires a local directory, not a HuggingFace model name:

```python
from huggingface_hub import snapshot_download
model_path = snapshot_download(model_name)  # Returns local cache path
```

#### D. Configuring Diffulex

```python
diffulex_config = DiffulexConfig(
    model=model_path,  # Local directory path
    kvcache_block_size=16,  # Must be divisible by 16
    num_kvcache_blocks=512,  # Must be > 0
    # ... other params
)
```

## Technical Discoveries

### vLLM v1 Architecture

1. **Engine Structure**:
   - `LLM.llm_engine` → `LLMEngine` (main process)
   - `LLMEngine.engine_core` → `SyncMPClient` (IPC client)
   - Worker process runs `EngineCoreProc`

2. **Executor Types**:
   - `UniProcExecutor` - single process (but still uses IPC!)
   - `MultiprocExecutor` - multiple processes
   - All use ZeroMQ for inter-process communication

3. **Model Access Path**:
   ```
   EngineCoreProc
     └─> self.model_executor (Executor)
          └─> driver_worker (Worker)
               └─> get_model() → Model
   ```

### Diffulex Requirements

1. **Config Parameters**:
   - `model` - Must be local directory path (not HF model name)
   - `kvcache_block_size` - Must be divisible by 16
   - `num_kvcache_blocks` - Must be > 0 (no auto-calculation)

2. **Device Handling**:
   - Original code hardcodes `torch.cuda.current_device()`
   - Needs patching for CPU support

## Challenges Overcome

### 1. Worker Process Isolation
**Problem**: Monkey-patches in main process don't affect worker process.
**Solution**: Patch at import time before worker spawns.

### 2. Model Access
**Problem**: Model is in separate process, not directly accessible.
**Solution**: Use `call_utility()` to execute code in worker process.

### 3. Tokenizer Loading
**Problem**: Worker doesn't have tokenizer attribute.
**Solution**: Load from model path using transformers.

### 4. Local Model Path
**Problem**: Diffulex expects local directory, not HF model name.
**Solution**: Use `snapshot_download()` to get cached path.

### 5. Config Validation
**Problem**: Diffulex has strict assertions on config parameters.
**Solution**: Calculate proper values (kvcache_block_size, num_kvcache_blocks).

## Results

✅ **Successfully implemented**:
- Monkey-patching at import time
- Worker utility method
- Model access in worker process
- Tokenizer loading
- Diffulex config creation
- Sequence and scheduler creation
- Generation loop startup

⏸️ **Remaining issue**:
- Diffulex hardcodes CUDA device in sequence.py
- Needs CPU support patch (see next section)

## Performance Notes

- **RPC Overhead**: Minimal - only for initial call and final results
- **Generation**: Runs entirely in worker process (no IPC during generation)
- **Memory**: Model stays in worker process (no copying)

## Future Improvements

1. **Batch Processing**: Currently processes prompts sequentially
2. **Error Handling**: Better error messages across process boundary
3. **Async Support**: Could implement async version of utility
4. **GPU Support**: Test and optimize for CUDA

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Import-time patching | `__init__.py` | 351-432 |
| Worker utility method | `generation.py` | 28-407 |
| Main process caller | `generation.py` | 509-588 |
| Model detection | `generation.py` | 95-168 |

## References

- vLLM v1 Engine: [vllm/v1/engine/](https://github.com/vllm-project/vllm/tree/main/vllm/v1/engine)
- EngineCoreProc: `vllm/v1/engine/core.py`
- Diffulex: `Diffulex/d2f_engine/`

---

**Implementation Date**: November 29, 2024
**vLLM Version**: 0.11.0
**Status**: Working with CPU patch needed
