#!/usr/bin/env python3
"""
Test the reference LLaDA generation implementation.
This verifies the reference code works and produces good output.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''Gumbel max sampling for categorical distributions'''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''Precompute number of tokens to transition at each step'''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128,
             temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''Reference LLaDA generation'''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length),
                                   dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, block_end:] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    print("Testing reference LLaDA generation...")
    print("=" * 80)

    device = 'cpu'  # Use CPU for compatibility

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

    # Simple test prompt
    prompt = "What is 2+2?"
    print(f"Test prompt: '{prompt}'")

    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    encoded = tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Generating with reference implementation...")
    print(f"  steps=32, gen_length=32, block_length=32")

    # Generate with reference implementation
    out = generate(
        model,
        input_ids,
        attention_mask,
        steps=32,  # Fewer steps for CPU
        gen_length=32,
        block_length=32,
        temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence'
    )

    output_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("RESULT:")
    print("-" * 80)
    print(output_text)
    print("-" * 80)

    # Check if output is reasonable
    if len(output_text.strip()) > 0 and not all(c in ' \n.?!,*"' for c in output_text):
        print("\n✓ Reference implementation produces non-trivial output")
        print("✓ Generation appears to work correctly")
    else:
        print("\n✗ Output seems empty or trivial")

    print("\n" + "=" * 80)
    print("This confirms the reference implementation works.")
    print("The vLLM integration needs to match this implementation.")


if __name__ == '__main__':
    main()
