# dllm_plugin: vLLM Plugin for Diffusion Language Models

A vLLM plugin that enables inference for diffusion language models:
- **Dream**: Diffusion-based language model
- **LLaDA**: Latent Diffusion Adapted language model

## Overview

This plugin integrates diffusion language model implementations from [Diffulex/D2fEngine](../Diffulex) with vLLM's high-performance inference engine. It provides vLLM-compatible adapters for Dream and LLaDA models, enabling them to leverage vLLM's optimized inference, batching, and serving capabilities.

## Installation

### Prerequisites

- Python 3.9 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Quick Install with UV

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install D2fEngine from Diffulex
cd ../Diffulex
uv sync
source .venv/bin/activate
uv pip install -e .

# Install vLLM
uv pip install vllm

# Install the plugin
cd ../dllm_plugin
uv pip install -e .
```

### Alternative Install with pip

```bash
# Install D2fEngine package from Diffulex
cd ../Diffulex
pip install -e .

# Install the plugin
cd ../dllm_plugin
pip install -e .
```

The plugin automatically registers with vLLM through the entry points mechanism.

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Basic Usage with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize the LLM with a Dream or LLaDA model
llm = LLM(
    model="path/to/dream/model",
    trust_remote_code=True
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=100
)

# Generate text
prompts = ["Tell me about diffusion language models."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated text: {output.outputs[0].text}")
```

### OpenAI-Compatible API Server

Start the vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model path/to/dream/model \
    --trust-remote-code
```

Use the API with the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.completions.create(
    model="path/to/dream/model",
    prompt="Tell me about diffusion language models.",
    max_tokens=100
)

print(response.choices[0].text)
```

## Supported Models

### Dream (DreamForDiffusionLM)

Dream is a diffusion-based language model that uses full attention (not causal) for generation. Architecture features:
- RMSNorm layer normalization
- Rotary position embeddings (RoPE)
- SiLU activation with gating
- Grouped query attention support

**Configuration**: Uses `DreamConfig` from the model's config.json

### LLaDA (LLaDAForDiffusionLM)

LLaDA is a latent diffusion adapted language model with architecture similar to Dream with implementation variations.

**Configuration**: Uses `LLaDAConfig` from the model's config.json

## Architecture

The plugin consists of:

1. **Registration Module** (`__init__.py`): Registers models with vLLM's ModelRegistry
2. **Model Adapters** (`models/`):
   - `dream.py`: vLLM adapter for Dream
   - `llada.py`: vLLM adapter for LLaDA

Each adapter wraps the original Diffulex implementation and provides:
- vLLM-compatible `forward()` method
- `compute_logits()` for logit computation
- `load_weights()` for checkpoint loading
- Integration with vLLM's batching and inference pipeline

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

## Differences from Standard vLLM Models

Diffusion language models have unique characteristics:

1. **Full Attention**: Unlike causal LMs, diffusion models use full (bidirectional) attention
2. **No KV Caching**: Standard KV caching does not apply to diffusion models
3. **Special Generation**: Different sampling strategies optimized for diffusion

## Troubleshooting

### Model Not Recognized

When vLLM does not recognize a diffusion model:

1. Verify plugin installation: `uv pip list | grep dllm-plugin`
2. Check the model's `config.json` for correct `architectures` field:
   - For Dream: `["DreamForDiffusionLM"]`
   - For LLaDA: `["LLaDAForDiffusionLM"]`

### Import Errors

Ensure all dependencies are installed:
```bash
uv pip install -e ../Diffulex  # D2fEngine package
uv pip install vllm>=0.6.0
uv pip install -e .
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

Contributions are welcome. Please ensure:
1. Code follows existing style conventions
2. All tests pass
3. Documentation is updated accordingly

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [Diffulex/D2fEngine](../Diffulex)
