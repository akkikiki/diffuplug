"""
Test generation with reference implementation parameters.
"""
import os
os.environ['DLLM_DIFFUSION_STEPS'] = '128'  # Use 128 steps like reference

from vllm import LLM, SamplingParams
import dllm_plugin  # This registers the model

def main():
    # Initialize model
    print("Initializing LLM...")
    llm = LLM(
        model="GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
    )

    # Get tokenizer
    tokenizer = llm.get_tokenizer()

    # Ensure left padding (like reference)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # Test prompts (same as reference)
    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
    ]

    # Apply chat template (like reference)
    print("\nApplying chat template...")
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    formatted_prompts = [
        tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        for message in messages
    ]

    print("\nFormatted prompts:")
    for i, fp in enumerate(formatted_prompts):
        print(f"{i+1}. {fp[:100]}...")

    # Sampling parameters
    # Reference uses: steps=128, gen_length=128, block_length=32, temperature=0
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,  # gen_length=128 like reference
    )

    print("\n" + "="*80)
    print("GENERATION PARAMETERS (matching reference):")
    print(f"  steps: 128 (from DLLM_DIFFUSION_STEPS env var)")
    print(f"  gen_length (max_tokens): {sampling_params.max_tokens}")
    print(f"  block_length (diffusion_block_size): 32 (default)")
    print(f"  temperature: {sampling_params.temperature}")
    print("="*80)

    # Generate
    print("\nGenerating...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Print results
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Output: {output.outputs[0].text}")
        print("-" * 80)

if __name__ == '__main__':
    main()
