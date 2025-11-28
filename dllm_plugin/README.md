# dllm_plugin: vLLM Plugin for Diffusion Language Models

This plugin enables vLLM to run diffusion language models, specifically:
- **Dream**: Diffusion-based language model
- **LLaDA**: Latent Diffusion Adapted language model

## Overview

This plugin integrates the diffusion language model implementations from [Diffulex/D2fEngine](../Diffulex) with vLLM's high-performance inference engine. It provides vLLM-compatible adapters for Dream and LLaDA models, allowing them to benefit from vLLM's optimized inference, batching, and serving capabilities.

## Installation

### Prerequisites

1. Install [UV](https://github.com/astral-sh/uv) package manager (recommended) or pip
2. Python 3.9 or higher

### Quick Install with UV

```bash
# Install UV if you haven't already
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

The plugin will automatically register with vLLM through the entry points mechanism.

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Basic Usage with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize the LLM with a Dream or LLaDA model
llm = LLM(
    model="path/to/dream/model",  # or path to LLaDA model
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

### Using the OpenAI-Compatible API Server

```bash
# Start the vLLM server with a diffusion model
python -m vllm.entrypoints.openai.api_server \
    --model path/to/dream/model \
    --trust-remote-code
```

Then use it with the OpenAI client:

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

Dream is a diffusion-based language model that uses full attention (not causal) for generation. The model architecture follows:
- RMSNorm layer normalization
- Rotary position embeddings (RoPE)
- SiLU activation with gating
- Grouped query attention support

**Configuration**: Uses `DreamConfig` from the model's config.json

### LLaDA (LLaDAForDiffusionLM)

LLaDA is a latent diffusion adapted language model with similar architecture to Dream but with some differences in implementation.

**Configuration**: Uses `LLaDAConfig` from the model's config.json

## Architecture

The plugin consists of:

1. **Registration Module** (`__init__.py`): Registers the models with vLLM's ModelRegistry
2. **Model Adapters** (`models/`):
   - `dream.py`: vLLM adapter for Dream
   - `llada.py`: vLLM adapter for LLaDA

Each adapter wraps the original Diffulex implementation and provides:
- vLLM-compatible `forward()` method
- `compute_logits()` for logit computation
- `load_weights()` for checkpoint loading
- Proper integration with vLLM's batching and inference pipeline

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

You can control which plugins are loaded using the `VLLM_PLUGINS` environment variable:

```bash
# Load only specific plugins
export VLLM_PLUGINS=register_dllm_models

# Load all plugins (default)
export VLLM_PLUGINS=
```

## Differences from Standard vLLM Models

Diffusion language models have some unique characteristics:

1. **Full Attention**: Unlike causal LMs, diffusion models use full (bidirectional) attention
2. **No KV Caching**: Standard KV caching doesn't apply to diffusion models
3. **Special Generation**: May require different sampling strategies optimized for diffusion

## Troubleshooting

### Model not recognized

If vLLM doesn't recognize your diffusion model:

1. Verify the plugin is installed: `pip list | grep dllm-plugin`
2. Check that the model's `config.json` has the correct `architectures` field:
   - For Dream: `["DreamForDiffusionLM"]`
   - For LLaDA: `["LLaDAForDiffusionLM"]`

### Import errors

Ensure all dependencies are installed:
```bash
pip install -e ../Diffulex  # D2fEngine package
pip install vllm>=0.6.0
pip install -e .
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing style
2. Tests pass
3. Documentation is updated

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

Note: This plugin integrates with Diffulex (MIT License) and vLLM (Apache License 2.0).

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [Diffulex/D2fEngine](../Diffulex)
