"""
Example usage of the dllm_plugin with vLLM.

This script demonstrates how to use diffusion language models (Dream or LLaDA)
with vLLM after installing the dllm_plugin.
"""

import argparse
from vllm import LLM, SamplingParams


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
        default=100,
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

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")

    # Initialize the LLM
    # The plugin will automatically detect and handle Dream/LLaDA models
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        # Additional configurations for diffusion models
        # Note: You may need to adjust these based on your model's requirements
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Single prompt example
    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)

    outputs = llm.generate([args.prompt], sampling_params)

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
