# Dream DVLLM GSM8K Test Script Usage

This script has been modified to support flexible configuration through command-line arguments.

## Requirements

- **GPU**: This script requires a CUDA-enabled NVIDIA GPU
- **Dependencies**: flash-attn, triton (CUDA-only libraries)
- **Models**: Dream-v0-Base-7B and optionally D2F_Dream_Base_7B_Lora

## Basic Usage

### Minimal Example (with local model)
```bash
python examples/test_dream_dvllm_gsm8k.py \
  --model /path/to/Dream-org/Dream-v0-Base-7B \
  --skip-sleep \
  --num-samples 5
```

### With LoRA Weights
```bash
python examples/test_dream_dvllm_gsm8k.py \
  --model /path/to/Dream-org/Dream-v0-Base-7B \
  --lora-path /path/to/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora \
  --use-lora \
  --num-samples 10 \
  --skip-sleep
```

### Using HuggingFace Model Names
```bash
python examples/test_dream_dvllm_gsm8k.py \
  --model Dream-org/Dream-v0-Base-7B \
  --lora-path SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora \
  --use-lora \
  --num-samples 10
```

### Full Configuration (Original Settings)
```bash
python examples/test_dream_dvllm_gsm8k.py \
  --model /root/autodl-fs/models/Dream-org/Dream-v0-Base-7B \
  --lora-path /root/autodl-fs/models/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora \
  --use-lora \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.25 \
  --max-num-seqs 20 \
  --max-model-len 2048 \
  --kv-cache-layout unified
```

## Command-Line Arguments

### Required
- `--model`: Path to the base model (local path or HuggingFace name)

### Optional
- `--lora-path`: Path to LoRA weights (default: None)
- `--use-lora`: Enable LoRA (flag)
- `--dataset`: Path to GSM8K dataset (default: openai/gsm8k)
- `--num-samples`: Number of samples to test (default: all ~1300)
- `--data-parallel-size`: Data parallel size (default: 1)
- `--tensor-parallel-size`: Tensor parallel size (default: 1)
- `--gpu-memory-utilization`: GPU memory utilization (default: 0.9)
- `--max-num-seqs`: Max number of sequences (default: 20)
- `--max-model-len`: Max model length (default: 2048)
- `--kv-cache-layout`: KV cache layout - unified or distinct (default: unified)
- `--output-file`: Output profiling file (default: log/profiles/perf_dvllm_dream_7B.json)
- `--skip-sleep`: Skip the 60 second sleep before generation (flag)
- `--temperature`: Sampling temperature (default: 0.0)
- `--max-tokens`: Max tokens to generate (default: 256)

## Quick Testing

For quick testing with a small number of samples:

```bash
python examples/test_dream_dvllm_gsm8k.py \
  --model <your-model-path> \
  --num-samples 5 \
  --skip-sleep
```

## Notes for Mac Users

⚠️ **This script cannot run on Mac** because it requires:
- CUDA (NVIDIA GPU)
- flash-attn (CUDA-only)
- triton (CUDA-only)

However, the import errors have been fixed to allow code editing and development on Mac. To actually run the script, you'll need to:
1. Transfer to a Linux machine with NVIDIA GPU
2. Install CUDA toolkit
3. Install flash-attn: `pip install flash-attn --no-build-isolation`
4. Install triton: `pip install triton`

## Troubleshooting

### Model Path Error
```
Error: Model path 'xxx' does not exist.
```
**Solution**: Ensure the model path exists locally or use a valid HuggingFace model name (org/model format)

### CUDA Not Available
```
Error initializing LLM: ...
Note: This script requires a CUDA-enabled GPU to run.
```
**Solution**: Run on a machine with NVIDIA GPU and CUDA installed

### Flash Attention Error
```
RuntimeError: flash_attn is required for unified layout...
```
**Solution**: Either:
- Install flash-attn on a GPU machine
- Use `--kv-cache-layout distinct` (if supported)
