"""
Verify that vLLM loaded weights match HuggingFace reference model.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test prompt
prompt = "What is 2+2? Think setp by step"

# Load HuggingFace model directly
print("Loading HuggingFace model...")
model_name = "GSAI-ML/LLaDA-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu"
)
hf_model.eval()

print(f"Model loaded: {type(hf_model)}")
print(f"Model class: {hf_model.__class__.__name__}")

# Tokenize
encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
input_ids = encoded['input_ids']
print(f"\nInput tokens: {input_ids[0].tolist()}")
print(f"Input length: {input_ids.shape[1]}")

# Add mask tokens (same as in diffusion generation)
mask_token_id = 126336
gen_length = 32
mask_tokens = torch.full((1, gen_length), mask_token_id, dtype=torch.long)
full_input = torch.cat([input_ids, mask_tokens], dim=1)

print(f"\nFull input shape: {full_input.shape}")
print(f"Full input (first 20): {full_input[0, :20].tolist()}")
print(f"Number of mask tokens: {(full_input == mask_token_id).sum().item()}")

# Get logits from HF model
print("\nRunning HF model forward pass...")
with torch.no_grad():
    # Forward pass using input_ids parameter
    outputs = hf_model(input_ids=full_input)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

print(f"Logits shape: {logits.shape}")

# Analyze logits at first masked position
first_masked_pos = (full_input[0] == mask_token_id).nonzero(as_tuple=True)[0][0].item()
logits_at_pos = logits[0, first_masked_pos]

print(f"\nLogits at first masked position (pos {first_masked_pos}):")
print(f"  Min: {logits_at_pos.min().item():.2f}")
print(f"  Max: {logits_at_pos.max().item():.2f}")
print(f"  Mean: {logits_at_pos.mean().item():.2f}")
print(f"  Std: {logits_at_pos.std().item():.2f}")

# Top 5 logits
top_logits, top_indices = torch.topk(logits_at_pos, k=5)
print(f"\nTop 5 logits at pos {first_masked_pos}:")
for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
    token_text = tokenizer.decode([idx.item()])
    print(f"  {i+1}. token_id={idx.item()}, logit={logit.item():.2f}, text='{token_text}'")

print("\n" + "="*80)
print("Compare these values with the vLLM diagnostic output!")
print("If they match, the weights are loaded correctly.")
print("If they differ significantly, there's a weight loading issue.")
print("="*80)
