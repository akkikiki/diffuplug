# Installation Guide for dllm_plugin

This guide provides step-by-step instructions for installing and using the dllm_plugin with vLLM.

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (for optimal performance)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation Steps

### 0. Install UV (if not already installed)

UV is a fast Python package manager. Install it following the [official guide](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

For easy activation, add an alias to your shell configuration:

```bash
echo "alias uvon='source .venv/bin/activate'" >> ~/.zshrc  # or ~/.bashrc for bash
source ~/.zshrc
```

### Install Everything in One Environment

If you prefer to set up everything from the plugin directory:

```bash
cd /path/to/diffuplug

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install Diffulex
cd Diffulex
uv pip install -e Diffulex

# Install vLLM
uv pip install vllm

# Install the plugin
uv pip install -e .
```

## Verification

To verify the plugin is installed correctly:

```python
from vllm import ModelRegistry

# Check if the diffusion models are registered
supported_archs = ModelRegistry.get_supported_archs()
print("DreamForDiffusionLM" in supported_archs)  # Should print True
print("LLaDAForDiffusionLM" in supported_archs)  # Should print True
```

## Quick Start

### Option 1: Using the Example Script

```bash
python example_usage.py \
    --model /path/to/dream/or/llada/model \
    --prompt "The future of AI is" \
    --max-tokens 100
```

### Option 2: Python API

```python
from vllm import LLM, SamplingParams

# Load a Dream or LLaDA model
llm = LLM(
    model="/path/to/model",
    trust_remote_code=True
)

# Generate text
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Tell me about diffusion models"], sampling_params)

print(outputs[0].outputs[0].text)
```

### Option 3: OpenAI-Compatible API Server

```bash
# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --trust-remote-code

# In another terminal, use the API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/model",
        "prompt": "Tell me about diffusion models",
        "max_tokens": 100
    }'
```

## Troubleshooting

### Plugin Not Recognized

If vLLM doesn't recognize the plugin:

1. **Check installation:**
   ```bash
   uv pip list | grep dllm-plugin
   ```

2. **Verify entry points:**
   ```bash
   python -c "import pkg_resources; print([ep for ep in pkg_resources.iter_entry_points('vllm.general_plugins')])"
   ```

3. **Force plugin loading:**
   ```bash
   export VLLM_PLUGINS=register_dllm_models
   ```

### Import Errors

If you get import errors related to `d2f_engine`:

1. Make sure Diffulex is installed:
   ```bash
   uv pip list | grep d2f
   ```

2. Install it if missing:
   ```bash
   cd /path/to/Diffulex
   uv sync
   source .venv/bin/activate
   uv pip install -e .
   ```

### Model Architecture Not Found

Ensure your model's `config.json` contains:

```json
{
    "architectures": ["DreamForDiffusionLM"]
}
```

or

```json
{
    "architectures": ["LLaDAForDiffusionLM"]
}
```

## Uninstallation

To uninstall the plugin:

```bash
uv pip uninstall dllm-plugin
```

## Development Mode

For development, install with additional dependencies:

```bash
uv pip install -e ".[dev]"
```

Or if you want to use UV's project management:

```bash
# From the plugin directory
uv sync --all-extras
```

## Next Steps

- Check out the [README.md](README.md) for detailed usage examples
- See [example_usage.py](example_usage.py) for more code examples
- Read the [vLLM documentation](https://docs.vllm.ai/) for advanced configurations
