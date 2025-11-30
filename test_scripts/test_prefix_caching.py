"""
Test prefix caching optimization for LLaDA block generation.
"""
import os
import time
os.environ['DLLM_DIFFUSION_STEPS'] = '64'  # Use more steps to see caching benefit

import logging
logging.basicConfig(level=logging.INFO)

# Register plugin
import dllm_plugin
dllm_plugin.register()

from vllm import LLM, SamplingParams

def main():
    print("=" * 80)
    print("Testing LLaDA with Prefix Caching Optimization")
    print("=" * 80)

    # Initialize model
    print("\nInitializing LLM...")
    llm = LLM(
        model="GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
    )

    # Get tokenizer
    tokenizer = llm.get_tokenizer()

    # Test prompt
    prompt = "Write a short story about a robot learning to paint."

    # Apply chat template
    message = {"role": "user", "content": prompt}
    formatted_prompt = tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)

    print(f"\nPrompt: {prompt}")
    print(f"Formatted (first 100 chars): {formatted_prompt[:100]}...")

    # Sampling parameters - generate longer text to benefit from caching
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,  # Generate 4 blocks of 32 tokens each
    )

    print("\n" + "=" * 80)
    print("Generating with Prefix Caching...")
    print("=" * 80)
    print("Watch for log messages:")
    print("  - 'Caching prefix KV' indicates prefix is being cached")
    print("  - 'Using prefix cache' shows tokens saved per diffusion step")
    print("=" * 80)

    # Generate
    start_time = time.time()
    outputs = llm.generate([formatted_prompt], sampling_params)
    end_time = time.time()

    # Print result
    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    generated_text = outputs[0].outputs[0].text
    print(f"Generated: {generated_text}")
    print("=" * 80)
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    print("=" * 80)

    # Check if output is good
    if len(generated_text) > 50 and not all(c in ': Spencer.\n., "e:, until' for c in generated_text[:20]):
        print("\n✓ Output looks reasonable!")
        print(f"✓ Generated {len(generated_text)} characters")
    else:
        print("\n⚠ Output might be garbage")

if __name__ == '__main__':
    main()
