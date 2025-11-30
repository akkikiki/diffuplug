"""
Run the reference implementation directly (not through vLLM).
This will tell us if the model can produce good outputs.
"""
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Reference implementation (from the user's shared code)
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=32, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    print(f"Generating {gen_length} tokens in {num_blocks} blocks, {steps_per_block} steps per block")

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        print(f"\nBlock {num_block + 1}/{num_blocks}")

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/{steps_per_block}: {mask_index.sum().item()} masks remaining")

    return x

def main():
    device = 'cpu'
    print("Loading model...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.float32).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    tokenizer.padding_side = 'left'

    # Simple test prompt
    prompt = "What is 2+2? Think step by step"

    # Apply chat template
    message = {"role": "user", "content": prompt}
    formatted_prompt = tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)

    print(f"\nPrompt: {prompt}")
    print(f"Formatted: {formatted_prompt[:100]}...")

    # Tokenize
    encoded = tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate with reference implementation
    print("\n" + "="*80)
    print("GENERATING (32 tokens, 32 steps)...")
    print("="*80)

    out = generate(model, input_ids, attention_mask, steps=32, gen_length=32, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')

    # Decode
    output_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(f"Generated: {output_text}")
    print("="*80)

if __name__ == '__main__':
    main()
