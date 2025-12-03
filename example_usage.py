"""
Example usage of the dllm_plugin with vLLM.

This script demonstrates how to use diffusion language models (Dream or LLaDA)
with vLLM after installing the dllm_plugin.

Configurable Parameters:
  --temperature: Controls sampling randomness (default: 1.0)
  --max-tokens: Maximum number of tokens to generate (default: 2)
  --top-p: Top-p (nucleus) sampling parameter (default: 0.9)
  --diffusion-steps: Number of diffusion denoising steps (default: model config)
  --block-size: KV cache block size (default: 4, rounded to multiple of 16)
  --diffusion-block-size: Diffusion generation block size (default: 32)
  --use-chat-template: Apply chat template to prompts (for Instruct models)

Example usage:
  # Basic generation
  python example_usage.py --model /path/to/model

  # With custom temperature and max tokens
  python example_usage.py --model /path/to/model --temperature 0.5 --max-tokens 64

  # With custom block sizes (affects generation quality/speed)
  python example_usage.py --model /path/to/model --block-size 8 --diffusion-block-size 16

  # With chat template (for Instruct models)
  python example_usage.py --model GSAI-ML/LLaDA-8B-Instruct --use-chat-template --max-tokens 128
"""

import argparse
import logging
import sys
import os

# Parse arguments FIRST to set environment variables before spawning workers
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
    default="What is 2+2? ",
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
    "--diffusion-steps",
    type=int,
    default=None,
    help="Number of diffusion steps (default: use model config value)",
)
parser.add_argument(
    "--block-size",
    type=int,
    default=None,
    help="KV cache block size (default: 4, will be rounded up to nearest multiple of 16)",
)
parser.add_argument(
    "--diffusion-block-size",
    type=int,
    default=None,
    help="Diffusion generation block size (default: 32)",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Enable verbose logging",
)
parser.add_argument(
    "--use-chat-template",
    action="store_true",
    help="Apply chat template to prompts (for Instruct models)",
)

args = parser.parse_args()

# Set diffusion steps via environment variable BEFORE registering plugin
# This ensures worker processes inherit the environment variable
if args.diffusion_steps is not None:
    os.environ['DLLM_DIFFUSION_STEPS'] = str(args.diffusion_steps)
    print(f"Setting diffusion steps to: {args.diffusion_steps}")

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

# Get logger instance (will be configured in main())
logger = logging.getLogger('dllm_plugin')


def main():
    # Set up logging based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

    # Configure plugin logging
    logger.setLevel(log_level)

    # Configure vLLM logging
    if args.verbose:
        logging.getLogger('vllm').setLevel(logging.DEBUG)
        logging.getLogger('vllm.v1').setLevel(logging.DEBUG)
    else:
        logging.getLogger('vllm').setLevel(logging.WARNING)
        logging.getLogger('vllm.v1').setLevel(logging.WARNING)

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

    # Get tokenizer for chat template
    tokenizer = llm.get_tokenizer()

    # Helper function to apply chat template if requested
    def format_prompt(prompt):
        if args.use_chat_template:
            message = {"role": "user", "content": prompt}
            return tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        return prompt

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Single prompt example
    formatted_prompt = format_prompt(args.prompt)
    print(f"\nPrompt: {args.prompt}")
    if args.use_chat_template:
        print(f"Formatted prompt: {formatted_prompt[:200]}...")
    print("-" * 80)

    logger.info("Starting generation...")
    logger.debug(f"Sampling params: {sampling_params}")

    try:
        logger.info("Calling llm.generate()...")

        outputs = llm.generate([formatted_prompt], sampling_params)
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
        "Hello! How are you?",
        "What is 2+2? Think setp by step",
        #"The impact of climate change on",
    ]

    # Apply chat template to batch prompts if requested
    formatted_prompts = [format_prompt(p) for p in prompts]

    outputs = llm.generate(formatted_prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Output: {generated_text}\n")

    # Example using generate_with_diffusion directly with custom block sizes
    if args.block_size is not None or args.diffusion_block_size is not None:
        print("\nCustom block size generation example:")
        print("-" * 80)
        print(f"Using block_size={args.block_size}, diffusion_block_size={args.diffusion_block_size}")

        from dllm_plugin import generate_with_diffusion

        custom_outputs = generate_with_diffusion(
            llm,
            prompts=["What is the capital of France?"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            block_size=args.block_size,
            diffusion_block_size=args.diffusion_block_size,
        )

        for output in custom_outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text}")
        print("-" * 80)

    print("-" * 80)
    print("Generation complete!")


if __name__ == "__main__":
    main()
