"""
Example usage of the dllm_plugin with vLLM.

This script demonstrates how to use diffusion language models (Dream or LLaDA)
with vLLM after installing the dllm_plugin.
"""

import argparse
import logging
import sys

# IMPORTANT: Register the plugin BEFORE importing LLM
# This ensures the patching happens before vLLM's LLM class is used
try:
    import dllm_plugin
    dllm_plugin.register()
    print("✓ dllm_plugin registered successfully")
except Exception as e:
    print(f"⚠ Warning: Failed to register dllm_plugin: {e}")
    import traceback
    traceback.print_exc()

from vllm import LLM, SamplingParams

# Set up verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Enable logging for our plugin
logger = logging.getLogger('dllm_plugin')
logger.setLevel(logging.DEBUG)

# Enable logging for vLLM components
logging.getLogger('vllm').setLevel(logging.INFO)
logging.getLogger('vllm.v1').setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with diffusion language models using vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the Dream or LLaDA model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('dllm_plugin').setLevel(logging.DEBUG)
        logging.getLogger('vllm').setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('dllm_plugin').setLevel(logging.INFO)

    print(f"Loading model from: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    
    # Verify plugin is registered
    logger.info("Checking if plugin is registered...")
    try:
        from vllm import ModelRegistry
        supported_archs = ModelRegistry.get_supported_archs()
        logger.info(f"Supported architectures: {supported_archs}")
        if "LLaDAForDiffusionLM" in supported_archs or "DreamForDiffusionLM" in supported_archs:
            logger.info("✓ Diffusion models are registered")
        else:
            logger.warning("⚠ Diffusion models are NOT registered - patching may not work")
    except Exception as e:
        logger.warning(f"Could not verify registration: {e}")

    # Initialize the LLM
    # The plugin will automatically detect and handle Dream/LLaDA models
    logger.info("Creating LLM instance (patching should happen in __init__)...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graphs for CPU compatibility
        # Additional configurations for diffusion models
        # Note: You may need to adjust these based on your model's requirements
        # Try these:
        distributed_executor_backend=None,  # or try 'single', 'ray', etc.
        # Or disable multiprocessing entirely:
        # disable_custom_all_reduce=True,  # you already have this
    )
    
    # Verify patching worked
    logger.info("Verifying generate method was patched...")
    if hasattr(llm.generate, '__name__'):
        logger.info(f"LLM.generate method name: {llm.generate.__name__}")
        if 'patched' in llm.generate.__name__ or 'patch' in llm.generate.__name__:
            logger.info("✓ Generate method appears to be patched")
        else:
            logger.warning("⚠ Generate method does not appear to be patched!")
    else:
        logger.warning("⚠ Could not verify if generate method was patched")

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Single prompt example
    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)
    
    logger.info("Starting generation...")
    logger.debug(f"Sampling params: {sampling_params}")
    
    try:
        logger.info("Calling llm.generate()...")
        
        outputs = llm.generate([args.prompt], sampling_params)
        logger.info("Generation completed successfully")
    except KeyboardInterrupt:
        logger.error("Generation interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Generation failed with error: {e}", exc_info=True)
        raise

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text:\n{generated_text}")
        print("-" * 80)

    # Batch generation example
    print("\nBatch generation example:")
    print("-" * 80)

    prompts = [
        "The future of artificial intelligence is",
        "In the world of quantum computing,",
        "The impact of climate change on",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Output: {generated_text}\n")

    print("-" * 80)
    print("Generation complete!")


if __name__ == "__main__":
    main()
