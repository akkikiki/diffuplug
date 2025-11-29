#!/usr/bin/env python3
"""
Test the new LLaDASampler implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dllm_plugin'))

import torch
from transformers import AutoTokenizer, AutoModel
from dllm_plugin.llada_sampler import LLaDASampler

def main():
    print("Testing new LLaDASampler...")
    print("=" * 80)

    device = 'cpu'

    print("Loading model...")
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'

    print("✓ Model loaded\n")

    # Create sampler
    sampler = LLaDASampler(
        mask_token_id=126336,
        temperature=0.0,
        remasking='low_confidence'
    )
    print("✓ Sampler created\n")

    # Test prompt
    prompt = "What is 2+2?"
    print(f"Test prompt: '{prompt}'")

    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    encoded = tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"\nGenerating with new LLaDASampler...")
    print(f"  steps=32, gen_length=32, block_length=32\n")

    # Generate
    output_ids = sampler.generate(
        model=model,
        prompt_ids=input_ids,
        attention_mask=attention_mask,
        steps=32,
        gen_length=32,
        block_length=32
    )

    output_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print("\n" + "=" * 80)
    print("RESULT:")
    print("-" * 80)
    print(output_text)
    print("-" * 80)

    # Check quality
    if len(output_text.strip()) > 0:
        print("\n✓ New sampler produces output")

        # Check if it's reasonable (not gibberish)
        if '4' in output_text or 'four' in output_text.lower():
            print("✓ Output appears correct (contains '4')")
            print("✅ NEW SAMPLER WORKS CORRECTLY!")
        else:
            print("⚠ Output doesn't contain expected answer")
    else:
        print("\n✗ No output generated")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
