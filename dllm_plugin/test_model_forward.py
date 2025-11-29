"""
Test script to verify the model forward pass works independently of vLLM.
"""
import torch
import sys
import os

# Add the plugin to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dllm_plugin.models.llada import LLaDAForDiffusionLMVLLM
from vllm.config import VllmConfig
from transformers import AutoConfig

def test_forward():
    print("Testing LLaDA model forward pass...")
    
    # Create a minimal config
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    
    try:
        # Load config
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loaded config: {type(hf_config)}")
        
        # Create vLLM config (minimal)
        # We need to create a mock VllmConfig object
        from types import SimpleNamespace
        
        model_config = SimpleNamespace()
        model_config.hf_config = hf_config
        
        vllm_config = SimpleNamespace()
        vllm_config.model_config = model_config
        
        # Initialize model
        print("Initializing model...")
        model = LLaDAForDiffusionLMVLLM(vllm_config=vllm_config)
        print("Model initialized successfully")
        
        # Test forward pass with dummy inputs
        print("Testing forward pass...")
        batch_size = 1
        seq_len = 10
        vocab_size = model.config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Positions shape: {positions.shape}")
        
        with torch.no_grad():
            hidden_states = model.forward(input_ids, positions)
            print(f"Forward pass completed. Hidden states shape: {hidden_states.shape}")
            
            logits = model.compute_logits(hidden_states)
            print(f"Logits computed. Logits shape: {logits.shape}")
        
        print("✓ Model forward pass test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forward()
    sys.exit(0 if success else 1)

