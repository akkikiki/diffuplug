"""
Test the fixed LLaDA model that uses HuggingFace implementation.
"""
import os
os.environ['DLLM_DIFFUSION_STEPS'] = '32'  # Use fewer steps for testing

import logging
logging.basicConfig(level=logging.INFO)

# Register plugin
import dllm_plugin
dllm_plugin.register()

from vllm import LLM, SamplingParams

def main():
    print("=" * 80)
    print("Testing LLaDA with HuggingFace model implementation")
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
    prompt = "What is 2+2? Think step by step"

    # Apply chat template
    message = {"role": "user", "content": prompt}
    formatted_prompt = tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)

    print(f"\nPrompt: {prompt}")
    print(f"Formatted (first 100 chars): {formatted_prompt[:100]}...")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,  # Short generation for testing
    )

    print("\n" + "=" * 80)
    print("Generating...")
    print("=" * 80)

    # Generate
    outputs = llm.generate([formatted_prompt], sampling_params)

    # Print result
    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    generated_text = outputs[0].outputs[0].text
    print(f"Generated: {generated_text}")
    print("=" * 80)

    # Check if output is good
    if "2" in generated_text and "4" in generated_text:
        print("\n✓ Output looks reasonable (contains '2' and '4')")
    else:
        print("\n⚠ Output might be garbage")

if __name__ == '__main__':
    main()
