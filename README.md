<p align="center">
  <img src="assets/diffuplug_logo_v4.png" alt="Plugin Icon" width="500"/>
</p>





# A vLLM Plugin for Diffusion Language Models

A vLLM plugin that enables inference for diffusion language models:
- **LLaDA**: Latent Diffusion Adapted language model ✅ 
- **Dream**: Diffusion-based language model ⚠️ **TODO**

## Overview

This plugin provides vLLM-compatible adapters for diffusion language models:

https://github.com/user-attachments/assets/a0ddc086-2471-4d2b-8868-8c36f4ab12fa

### **LLaDA**
- Uses **HuggingFace's official LLaDA model** via `AutoModel` with `trust_remote_code=True`
- Custom **LLaDASampler** implementing the reference generation code at [LLaDA](https://github.com/ML-GSAI/LLaDA/tree/main) - Original LLaDA implementation.
- **Prefix caching** for generation
- Works with Python API (offline inference)
- **No Diffulex dependency** for LLaDA

### **Dream (TODO)**
- Currently uses [Diffulex/D2fEngine](../Diffulex) implementation
- Needs update to match LLaDA architecture (HuggingFace model + custom sampler)
- Future work: Apply same fixes and optimizations as LLaDA

## Installation

### Prerequisites

- Python 3.9 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Quick Install with UV

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install vLLM
uv pip install vllm

# Install the plugin
uv pip install -e .
```

The plugin automatically registers with vLLM through the entry points mechanism.

**Note**: Diffulex is no longer required for LLaDA models.

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Example Usage with vLLM
```
python example_usage.py --model GSAI-ML/LLaDA-8B-Instruct --max-tokens 20 --diffusion-steps 32
```

### OpenAI-Compatible API Server

⚠️ **TODO**: OpenAI API server support for LLaDA diffusion models is currently in development.

The custom diffusion generation logic (using `LLaDASampler`) bypasses vLLM's standard generation pipeline, which is required for the OpenAI API server integration.

**Current workaround**: Use the Python API directly (see Basic Usage above).

## Supported Models

### LLaDA (LLaDAModelLM)

**Implementation**:
- Uses LLaDA's HF implemenation via `AutoModel.from_pretrained()`
- Custom `LLaDASampler` implementing diffusion algorithm from [LLaDA](https://github.com/ML-GSAI/LLaDA/tree/main) including block-based generation
- Prefix caching added

**Features**:
- CPU support

**Example Model**: `GSAI-ML/LLaDA-8B-Instruct`

**Configuration**: Uses `LLaDAConfig` from the model's config.json

### Dream (DreamForDiffusionLM) ⚠️

**Status**: TODO - Needs update to new architecture

**Current Implementation**:
- Uses Diffulex/D2fEngine library with older generation logic
- Requires update to match LLaDA architecture (HuggingFace model + custom sampler)

**Configuration**: Uses `DreamConfig` from the model's config.json

## Architecture

The plugin consists of:

1. **Registration Module** (`__init__.py`): Registers models with vLLM's ModelRegistry
2. **Model Adapters** (`models/`):
   - `llada.py`: vLLM adapter for LLaDA (HuggingFace model wrapper)
   - `dream.py`: vLLM adapter for Dream (Diffulex wrapper - TODO)
3. **Generation Logic**:
   - `llada_sampler.py`: Custom diffusion sampler for LLaDA
   - `generation_new.py`: Worker-based generation using LLaDASampler
   - `generation.py`: Older Diffulex-based generation (used by Dream)

### LLaDA Architecture:
- **Model**: HuggingFace `AutoModel` with `trust_remote_code=True`
- **Sampler**: Custom `LLaDASampler` implementing reference algorithm
- **Features**:
  - No Diffulex dependency

## Plugin Mechanism

This plugin uses vLLM's standard plugin system:

1. Entry point registration in `setup.py`:
   ```python
   entry_points={
       "vllm.general_plugins": [
           "register_dllm_models = dllm_plugin:register"
       ]
   }
   ```

2. The `register()` function registers models with vLLM's ModelRegistry:
   ```python
   ModelRegistry.register_model(
       "DreamForDiffusionLM",
       "dllm_plugin.models.dream:DreamForDiffusionLMVLLM"
   )
   ```

## Environment Variables

Control plugin loading using the `VLLM_PLUGINS` environment variable:

```bash
# Load only specific plugins
export VLLM_PLUGINS=register_dllm_models

# Load all plugins (default)
export VLLM_PLUGINS=
```

## Troubleshooting

### Model Not Recognized

When vLLM does not recognize a diffusion model:

1. Verify plugin installation: `uv pip list | grep dllm-plugin`
2. Check the model's `config.json` for correct `architectures` field:
   - For Dream: `["DreamForDiffusionLM"]`
   - For LLaDA: `["LLaDAModelLM"]`

### Import Errors

Ensure all dependencies are installed:
```bash
uv pip install vllm>=0.6.0
uv pip install -e .
```

## Development

### Running Tests

```bash
pytest
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## References

- [LLaDA](https://github.com/ML-GSAI/LLaDA/tree/main) - Original LLaDA implementation
- [DiffuLex](https://github.com/zhijie-group/Diffulex/tree/main) - nano-vllm extension for diffusion language models (LLaDA, Dream)
- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) - Minimal implementation reference for vLLM model integration
