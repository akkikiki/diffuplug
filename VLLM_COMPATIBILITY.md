# vLLM Compatibility Notes

## Current Status

The custom diffusion generation plugin currently **does not work with vLLM v1.x** due to architectural differences.

## Issue

vLLM v1.x uses a **multiprocess architecture** where:
- The model runs in a separate worker process
- The `LLM` object in your Python code is just a client/proxy
- Direct model access (required for custom forward passes) is not possible

## Evidence

```
engine_core type: <class 'vllm.v1.engine.core_client.EngineCoreClient'>
```

The model is not accessible via:
- `llm.llm_engine.model_executor` (doesn't exist)
- `llm.llm_engine.engine_core.decoder` (this is a MsgpackDecoder, not the model)
- `llm.llm_engine.engine_core.core_engines` (contains strings, not objects)

## Fallback Behavior

When custom diffusion generation fails, the code automatically falls back to **standard vLLM generation**, so your code will still run - it just won't use the custom diffusion path.

## Potential Solutions

### Option 1: Use vLLM v0.x
Downgrade to vLLM v0.x where the model is directly accessible:
```bash
pip install vllm<1.0.0
```

### Option 2: Check for Single-Process Mode
vLLM v1 might have a single-process mode. Check if there's a configuration option like:
```python
from vllm import LLM

llm = LLM(
    model="your-model",
    # Try adding options like:
    # distributed_executor_backend=None,
    # tensor_parallel_size=1,
    # etc.
)
```

### Option 3: Modify Plugin for vLLM v1 Architecture
This would require a significant rewrite to:
1. Use vLLM's `execute_model` or similar API methods instead of direct model access
2. Work within vLLM's multiprocess framework
3. Possibly contribute changes to vLLM to support custom generation loops

### Option 4: Use vLLM's Custom Model Integration
Register your diffusion model as a custom vLLM model type, which might give you hooks into the generation process.

## Recommended Next Steps

1. **Check your vLLM version**:
   ```python
   import vllm
   print(vllm.__version__)
   ```

2. **Try vLLM v0.x** if you need custom diffusion generation

3. **Use standard vLLM generation** if v1.x is required (current fallback behavior)

4. **Monitor vLLM development** for APIs that support custom generation loops in v1.x
