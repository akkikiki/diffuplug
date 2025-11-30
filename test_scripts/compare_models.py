"""
Compare Diffulex model vs HuggingFace model on the same input.
"""
import torch

# Patch torch.distributed for CPU single-process compatibility
if not torch.distributed.is_initialized():
    class MockDistributed:
        @staticmethod
        def get_world_size():
            return 1
        @staticmethod
        def get_rank():
            return 0
    torch.distributed.get_world_size = MockDistributed.get_world_size
    torch.distributed.get_rank = MockDistributed.get_rank

from transformers import AutoModel, AutoTokenizer
from d2f_engine.models.llada import LLaDAForDiffusionLM
from d2f_engine.models.config.llada.configuration_llada import LLaDAConfig

def main():
    device = 'cpu'

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )

    # Load Diffulex model with same config
    print("\nLoading Diffulex model...")
    config = LLaDAConfig.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')
    diffulex_model = LLaDAForDiffusionLM(config).to(device).eval()

    # Load the same weights into Diffulex model
    print("Loading weights into Diffulex model...")
    state_dict = hf_model.state_dict()

    # Check if weight keys match
    hf_keys = set(state_dict.keys())
    diffulex_keys = set(diffulex_model.state_dict().keys())

    print(f"\nHF model keys (first 10): {sorted(list(hf_keys))[:10]}")
    print(f"\nDiffulex model keys (first 10): {sorted(list(diffulex_keys))[:10]}")

    # Check for key differences
    only_in_hf = hf_keys - diffulex_keys
    only_in_diffulex = diffulex_keys - hf_keys

    if only_in_hf:
        print(f"\nKeys only in HF model (first 10): {sorted(list(only_in_hf))[:10]}")
    if only_in_diffulex:
        print(f"\nKeys only in Diffulex model (first 10): {sorted(list(only_in_diffulex))[:10]}")

    # Create simple test input
    prompt = "Hello"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print(f"\nTest input: {prompt}")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")

    # Test HF model
    print("\n" + "="*80)
    print("Testing HuggingFace model...")
    with torch.no_grad():
        hf_output = hf_model(input_ids, attention_mask=attention_mask)
        hf_logits = hf_output.logits

    print(f"HF output logits shape: {hf_logits.shape}")
    print(f"HF logits[0, -1, :10]: {hf_logits[0, -1, :10]}")
    print(f"HF logits max: {hf_logits.max():.4f}, min: {hf_logits.min():.4f}, mean: {hf_logits.mean():.4f}")

    # Get top 5 predictions
    hf_probs = torch.softmax(hf_logits[0, -1], dim=-1)
    hf_top_probs, hf_top_indices = torch.topk(hf_probs, k=5)
    print("\nHF top 5 predictions:")
    for prob, idx in zip(hf_top_probs, hf_top_indices):
        token = tokenizer.decode([idx])
        print(f"  {idx}: '{token}' (prob={prob:.4f})")

    # Test Diffulex model
    print("\n" + "="*80)
    print("Testing Diffulex model...")

    # Diffulex needs positions
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    try:
        with torch.no_grad():
            diffulex_hidden = diffulex_model(input_ids, positions, mask=attention_mask)
            diffulex_logits = diffulex_model.compute_logits(diffulex_hidden)

        print(f"Diffulex output logits shape: {diffulex_logits.shape}")
        print(f"Diffulex logits[0, -1, :10]: {diffulex_logits[0, -1, :10]}")
        print(f"Diffulex logits max: {diffulex_logits.max():.4f}, min: {diffulex_logits.min():.4f}, mean: {diffulex_logits.mean():.4f}")

        # Get top 5 predictions
        diffulex_probs = torch.softmax(diffulex_logits[0, -1], dim=-1)
        diffulex_top_probs, diffulex_top_indices = torch.topk(diffulex_probs, k=5)
        print("\nDiffulex top 5 predictions:")
        for prob, idx in zip(diffulex_top_probs, diffulex_top_indices):
            token = tokenizer.decode([idx])
            print(f"  {idx}: '{token}' (prob={prob:.4f})")

        # Compare outputs
        print("\n" + "="*80)
        print("COMPARISON:")
        logits_diff = (hf_logits - diffulex_logits).abs()
        print(f"Max absolute difference in logits: {logits_diff.max():.4f}")
        print(f"Mean absolute difference in logits: {logits_diff.mean():.4f}")

        if logits_diff.max() > 1e-3:
            print("\n⚠️  MODELS PRODUCE DIFFERENT OUTPUTS!")
        else:
            print("\n✓ Models produce similar outputs")

    except Exception as e:
        print(f"\n❌ Error with Diffulex model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
