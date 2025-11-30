# Dependencies

## External Libraries

### vLLM
- **Version**: 0.11.0+
- **Purpose**: LLM inference engine
- **Architecture**: v1 engine with multiprocess worker support
- **Repository**: https://github.com/vllm-project/vllm
- **Installation**: `pip install vllm>=0.11.0`

### Diffulex
- **Base Commit**: `9d461707f1bb01c7907a4a8d1d8cff289c087c90`
- **Branch**: `main` (HEAD -> main, origin/main, origin/HEAD)
- **Author**: drewjin
- **Date**: Fri Oct 10 07:29:34 2025 +0000
- **Commit Message**: "doc: rename the site url"
- **Purpose**: Diffusion language model framework
- **Status**: **Modified version included in this repository**

#### Diffulex Modifications

The included Diffulex library has been modified to support CPU execution:

**File**: `Diffulex/d2f_engine/layers/attention/attention_v5.py`

**Changes**:
1. **Line 26**: Added `HAS_TRITON_OPS` to imports
2. **Lines 134-147**: Added CPU fallback for KV cache store operations
3. **Lines 299-351**: Added CPU fallback for decode attention using PyTorch `scaled_dot_product_attention`

**Rationale**: Original Diffulex only supports CUDA via Triton kernels and flash_attn. These modifications add PyTorch fallbacks for CPU compatibility while preserving GPU optimizations.

**Compatibility**: Modifications are backward compatible - GPU code paths unchanged.

### PyTorch
- **Version**: Compatible with vLLM requirements
- **Purpose**: Deep learning framework
- **Required**: Yes
- **Features Used**:
  - `torch.nn.functional.scaled_dot_product_attention` (CPU fallback)
  - Standard tensor operations

### Transformers (HuggingFace)
- **Version**: Compatible with vLLM requirements
- **Purpose**: Model loading and tokenization
- **Required**: Yes
- **Features Used**: `AutoTokenizer`, model loading

### Other Dependencies
- **einops**: Tensor reshaping operations
- **huggingface_hub**: Model downloading and caching

## Python Version
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested**: Python 3.12

## Platform Support

### Operating Systems
- ✅ Linux (CPU and CUDA)
- ✅ macOS (CPU only)
- ✅ Windows (CPU and CUDA)

### Hardware
- ✅ CPU (x86_64, ARM64)
- ✅ NVIDIA GPU (CUDA 11.8+)
- ❌ AMD GPU (not tested)
- ❌ Intel GPU (not tested)

## Installation

### From Source
```bash
# Clone repository
git clone <repository-url>
cd dllm_vllm_plugin

# Install dependencies
pip install -e dllm_plugin/

# Diffulex is included (modified version)
# No separate installation needed
```

### Dependencies Install
```bash
# Core dependencies
pip install vllm>=0.11.0
pip install transformers
pip install torch
pip install einops

# Optional: For GPU support
pip install flash-attn  # CUDA only
```

## Dependency Graph

```
dllm_plugin
├── vLLM (0.11.0+)
│   ├── PyTorch
│   ├── Transformers
│   └── CUDA Toolkit (optional, for GPU)
├── Diffulex (modified, included)
│   ├── PyTorch
│   ├── einops
│   ├── Triton (optional, for GPU)
│   └── flash-attn (optional, for GPU)
└── HuggingFace Hub
    └── Transformers
```

## Version Compatibility Matrix

| Component | CPU | CUDA |
|-----------|-----|------|
| vLLM 0.11.0 | ✅ | ✅ |
| Diffulex (modified) | ✅ | ✅ |
| PyTorch 2.0+ | ✅ | ✅ |
| flash-attn | ❌ | ✅ |
| Triton | ❌ | ✅ |

## License Notes

- **vLLM**: Apache 2.0
- **Diffulex**: Check original repository license
- **dllm_plugin**: (Your license here)
- **PyTorch**: BSD-style license
- **Transformers**: Apache 2.0

## Acknowledgments

- **Diffulex Authors**: drewjin and contributors
- **vLLM Team**: For the excellent inference engine
- **PyTorch Team**: For the deep learning framework

---

**Last Updated**: November 29, 2024
