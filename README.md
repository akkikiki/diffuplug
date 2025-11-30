<p align="center">
  <img src="assets/diffuplug_logo_v4.png" alt="Plugin Icon" width="500"/>
</p>

# DiffuPlug

## Overview

This repository is a vLLM plugin for Diffusion Language Models (LLaDA and Dream [WIP]). This plugin allows the diffusion models from the Diffulex/D2fEngine project to run on vLLM's high-performance inference engine.

## Project Structure

```
dllm_vllm_plugin/
├── dllm_plugin/                    # Main plugin package
│   ├── dllm_plugin/
│   │   ├── __init__.py            # Plugin registration entry point
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── dream.py           # Dream model adapter for vLLM
│   │       └── llada.py           # LLaDA model adapter for vLLM
│   ├── setup.py                   # Package setup with entry points
│   ├── pyproject.toml             # Modern Python packaging config
│   ├── README.md                  # Comprehensive documentation
│   ├── INSTALL.md                 # Installation guide
│   ├── example_usage.py           # Usage examples
│   └── .gitignore                 # Git ignore rules
├── Diffulex/                      # Original D2fEngine implementation (MIT License)
├── vllm/                          # vLLM source code (Apache 2.0 License)
└── LICENSE                        # License file
```

## Key Components

### 1. Plugin Registration (`dllm_plugin/__init__.py`)

Registers Dream and LLaDA models with vLLM's ModelRegistry using the entry points mechanism:

```python
def register():
    ModelRegistry.register_model(
        "DreamForDiffusionLM",
        "dllm_plugin.models.dream:DreamForDiffusionLMVLLM"
    )
    ModelRegistry.register_model(
        "LLaDAForDiffusionLM",
        "dllm_plugin.models.llada:LLaDAForDiffusionLMVLLM"
    )
```

### 2. Model Adapters

#### Dream Adapter (`models/dream.py`)
- Wraps `DreamForDiffusionLM` from D2fEngine
- Implements vLLM-compatible interface:
  - `forward(input_ids, positions, intermediate_tensors, inputs_embeds)`
  - `compute_logits(hidden_states)`
  - `load_weights(weights)`
- Handles configuration conversion from HF config to DreamConfig
- Integrates LogitsProcessor for proper vLLM operation

#### LLaDA Adapter (`models/llada.py`)
- Wraps `LLaDAForDiffusionLM` from D2fEngine
- Same vLLM interface as Dream
- Includes packed modules mapping for efficient weight loading
- Handles stacked parameter loading (qkv_proj, gate_up_proj)

### 3. Entry Points Configuration

The plugin registers via standard Python entry points in `setup.py`:

```python
entry_points={
    "vllm.general_plugins": [
        "register_dllm_models = dllm_plugin:register"
    ]
}
```

## Features

### ✅ Implemented

1. **Full vLLM Integration**
   - Models automatically detected by vLLM
   - Compatible with vLLM's batching and inference pipeline
   - Works with OpenAI-compatible API server

2. **Model Support**
   - Dream (DreamForDiffusionLM) - **[WIP - Work In Progress]**
   - LLaDA (LLaDAForDiffusionLM)

3. **Key Capabilities**
   - Weight loading from checkpoints
   - Logits computation
   - Integration with vLLM's LogitsProcessor
   - Proper configuration handling

4. **Documentation**
   - Comprehensive README with usage examples
   - Installation guide
   - Example scripts
   - Troubleshooting section

### 📋 Architecture Details

**Diffusion Model Characteristics:**
- Full (bidirectional) attention, not causal
- No traditional KV caching (not applicable for diffusion)
- Special generation paradigm optimized for diffusion
- Support for D2F-specific decoding

**vLLM Compatibility:**
- Follows vLLM's model interface requirements
- Integrates with vLLM's tensor parallelism
- Compatible with vLLM's batching mechanism
- Works with vLLM's serving infrastructure

## Installation

### With UV (Recommended)

```bash
# 1. Install D2fEngine
cd Diffulex
uv sync
source .venv/bin/activate
uv pip install -e .

# 2. Install vLLM
uv pip install vllm

# 3. Install the plugin
cd ../dllm_plugin
uv pip install -e .
```

### With pip

```bash
# 1. Install D2fEngine
cd Diffulex
pip install -e .

# 2. Install the plugin
cd ../dllm_plugin
pip install -e .
```

## Usage Examples

### Python API
```python
from vllm import LLM, SamplingParams

llm = LLM(model="path/to/dream/model", trust_remote_code=True)
sampling_params = SamplingParams(temperature=1.0, max_tokens=100)
outputs = llm.generate(["Your prompt here"], sampling_params)
```

### Command Line
```bash
python example_usage.py --model /path/to/model --prompt "Your prompt"
```

### API Server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --trust-remote-code
```

## Technical Highlights

1. **Lazy Model Registration**: Uses vLLM's lazy loading mechanism with string-based registration to avoid unnecessary imports

2. **Weight Adapter**: Handles both direct and stacked parameter loading for efficient checkpoint loading

3. **Configuration Bridge**: Automatically converts between HuggingFace configs and model-specific configs (DreamConfig, LLaDAConfig)

4. **Error Handling**: Robust weight loading with fallback mechanisms for different naming conventions

## Compatibility

- **Python**: >= 3.9
- **vLLM**: >= 0.6.0
- **PyTorch**: >= 2.0.0
- **Transformers**: >= 4.30.0

## Future Enhancements

Potential improvements for future versions:

1. **Optimized Kernels**: Custom CUDA kernels for diffusion-specific operations
2. **Advanced Batching**: Diffusion-aware batching strategies
3. **Quantization Support**: FP8/INT8 quantization for diffusion models
4. **Pipeline Parallelism**: Full PP support for very large models
5. **Streaming Generation**: Support for streaming diffusion generation
6. **Additional Models**: Support for more diffusion LM variants

## Testing

To test the plugin:

```python
# Verify registration
from vllm import ModelRegistry
assert "DreamForDiffusionLM" in ModelRegistry.get_supported_archs()
assert "LLaDAForDiffusionLM" in ModelRegistry.get_supported_archs()

# Test loading
from vllm import LLM
llm = LLM(model="path/to/model")
```

## Contributing

The plugin is structured to be easily extensible:
- Add new models by creating adapters in `models/`
- Register new models in `__init__.py`
- Follow existing patterns for vLLM compatibility

## License

This project is licensed under the Apache License 2.0.

Note: This plugin integrates with Diffulex (MIT License) and vLLM (Apache License 2.0).

## Summary

This plugin successfully bridges Diffulex's diffusion language models with vLLM's inference engine, enabling:
- High-performance inference for diffusion LMs
- Easy integration with existing vLLM workflows
- Support for both Dream and LLaDA models
- Standard vLLM serving capabilities

The implementation follows vLLM's plugin architecture best practices and provides comprehensive documentation for users and developers.

## References

- [LLaDA](https://github.com/ML-GSAI/LLaDA/tree/main) - Original LLaDA implemenation
- [DiffuLex](https://github.com/zhijie-group/Diffulex/tree/main) - nano-vllm extension for diffusion language models (LLaDA, Dream)
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference engine
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) - Minimal implementation reference for vLLM model integration
