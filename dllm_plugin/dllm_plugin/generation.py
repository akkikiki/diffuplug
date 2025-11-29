"""
Custom generation methods for diffusion language models.

This module provides custom generation logic that runs the full diffusion process
instead of vLLM's standard autoregressive generation loop.

It integrates with the Diffulex engine components to run the complete diffusion generation.
"""

import logging
import sys
import os
from typing import List, Optional, Union
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)

# Track if we've already patched EngineCoreProc
_ENGINE_CORE_PATCHED = False

# Add Diffulex to path if needed
_diffulex_path = os.path.join(os.path.dirname(__file__), '../../Diffulex')
if os.path.exists(_diffulex_path) and _diffulex_path not in sys.path:
    sys.path.insert(0, _diffulex_path)


def patch_engine_core_for_diffusion():
    """
    Monkey-patch EngineCoreProc to add custom diffusion generation method.
    This allows us to run diffusion generation in the worker process where the model lives.
    """
    global _ENGINE_CORE_PATCHED
    if _ENGINE_CORE_PATCHED:
        logger.debug("EngineCoreProc already patched, skipping")
        return

    try:
        from vllm.v1.engine.core import EngineCoreProc

        def run_diffusion_generation(self, prompts: List[str], sampling_params_dict: dict) -> List[dict]:
            """
            Custom method to run diffusion generation in the worker process.
            This runs in the EngineCoreProc where self.model_executor has access to the model.

            Args:
                prompts: List of input prompts
                sampling_params_dict: Dict with sampling parameters (temperature, max_tokens, etc.)

            Returns:
                List of dicts with generated text and metadata
            """
            logger.info(f"[Worker Process] Running diffusion generation for {len(prompts)} prompts")

            try:
                # Access the model through model_executor
                # self.model_executor is the executor (UniProcExecutor, MultiprocExecutor, etc.)
                # We need to get the worker from the executor
                executor = self.model_executor

                logger.info(f"[Worker Process] Executor type: {type(executor)}")
                logger.info(f"[Worker Process] Executor class: {executor.__class__.__name__}")

                # Get the worker from the executor
                # Different executors have different ways to access the worker
                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker
                    logger.info(f"[Worker Process] Got worker via driver_worker: {type(worker)}")
                elif hasattr(executor, 'worker'):
                    worker = executor.worker
                    logger.info(f"[Worker Process] Got worker via worker: {type(worker)}")
                else:
                    # For UniProcExecutor, it might be the worker itself
                    worker = executor
                    logger.info(f"[Worker Process] Using executor as worker: {type(worker)}")

                # Get the model from the worker
                model = worker.get_model()

                logger.info(f"[Worker Process] Got model: {type(model)}")
                logger.info(f"[Worker Process] Model class: {model.__class__.__name__}")

                # Get device from model
                import torch
                if hasattr(model, 'device'):
                    device = str(model.device)
                else:
                    # Try to get device from first parameter
                    try:
                        device = str(next(model.parameters()).device)
                    except:
                        # Default to CPU if we can't determine device
                        device = 'cpu'
                logger.info(f"[Worker Process] Model device: {device}")

                # Get tokenizer from EngineCoreProc
                # In vLLM v1, the tokenizer is initialized separately
                # We need to create it from the model config
                from transformers import AutoTokenizer
                model_config = self.vllm_config.model_config
                model_path = getattr(model_config, 'model', None) or getattr(model_config.hf_config, '_name_or_path', None)

                logger.info(f"[Worker Process] Loading tokenizer from: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False
                )

                # Get model config
                model_config = self.vllm_config.model_config
                hf_config = model_config.hf_config
                max_model_len = getattr(model_config, 'max_model_len', 4096)

                # Get diffusion-specific parameters from model config
                # Check for environment variable override first
                import os
                diffusion_steps_env = os.environ.get('DLLM_DIFFUSION_STEPS')
                if diffusion_steps_env is not None:
                    diffusion_steps = int(diffusion_steps_env)
                else:
                    diffusion_steps = getattr(hf_config, 'diffusion_steps', 128)
                block_size = getattr(hf_config, 'block_size', 4)
                diffusion_block_size = getattr(hf_config, 'diffusion_block_size', 32)
                mask_token_id = getattr(hf_config, 'mask_token_id', None)

                # Diffulex requires kvcache_block_size to be divisible by 16
                # Round up to nearest multiple of 16
                kvcache_block_size = ((block_size + 15) // 16) * 16
                if kvcache_block_size < 16:
                    kvcache_block_size = 16

                logger.info(f"[Worker Process] Adjusted kvcache_block_size from {block_size} to {kvcache_block_size}")

                if mask_token_id is None:
                    try:
                        mask_token_id = tokenizer.mask_token_id
                    except:
                        mask_token_id = 126336  # LLaDA default

                logger.info(
                    f"[Worker Process] Diffusion config: steps={diffusion_steps}, "
                    f"block_size={block_size}, diffusion_block_size={diffusion_block_size}, "
                    f"mask_token_id={mask_token_id}"
                )

                # Import Diffulex components
                try:
                    from d2f_engine.sampling_params import SamplingParams as DiffulexSamplingParams
                    from d2f_engine.config import Config as DiffulexConfig
                    from d2f_engine.engine.sequence import SequenceForDiffusionLM
                    from d2f_engine.engine.scheduler import SchedulerForDiffusionLM
                    from d2f_engine.layers.sampler import AutoSampler
                    from d2f_engine.utils.context import set_context_diffusion_lm, reset_context_diffusion_lm

                    logger.info("[Worker Process] Successfully imported Diffulex components")
                except ImportError as e:
                    logger.error(f"[Worker Process] Failed to import Diffulex: {e}")
                    raise RuntimeError(f"Diffulex not available: {e}")

                # Get model path for config
                # Diffulex needs a local directory path, not a HF model name
                # Try to get the actual downloaded model path from HF cache
                model_name = getattr(model_config, 'model', None) or getattr(hf_config, '_name_or_path', None)
                if model_name is None:
                    model_name = "GSAI-ML/LLaDA-8B-Instruct"

                # Get the local path from HuggingFace cache
                import os
                from huggingface_hub import snapshot_download

                try:
                    # This will either download or find the cached model
                    model_path = snapshot_download(
                        model_name,
                        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
                        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                    )
                    logger.info(f"[Worker Process] Using local model path: {model_path}")
                except Exception as e:
                    logger.warning(f"[Worker Process] Could not get local model path: {e}")
                    # Fallback: try using model name directly (might fail)
                    model_path = model_name

                # Calculate number of KV cache blocks
                # For CPU, use a modest number of blocks
                # Each block stores kvcache_block_size tokens
                # We need enough blocks for max_model_len tokens per sequence
                tokens_per_block = kvcache_block_size
                blocks_per_seq = (max_model_len + tokens_per_block - 1) // tokens_per_block
                num_kvcache_blocks = blocks_per_seq * len(prompts) * 2  # 2x for safety margin

                logger.info(f"[Worker Process] Calculated num_kvcache_blocks: {num_kvcache_blocks}")

                # Determine KV cache layout based on device
                # "unified" layout requires flash_attn (CUDA-only)
                # "distinct" layout works on CPU
                # Check if device is CPU (handle both "cpu" and "cpu:0" formats)
                is_cpu = "cpu" in device.lower()
                kv_cache_layout = "distinct" if is_cpu else "unified"
                logger.info(f"[Worker Process] Device: {device}, is_cpu: {is_cpu}, kv_cache_layout: {kv_cache_layout}")

                # Detect model name for sampler selection
                # LLaDA models contain "llada" in the model path/name
                # Dream models contain "dream" in the model path/name
                model_name_lower = model_path.lower()
                if 'llada' in model_name_lower:
                    model_name = 'llada'
                elif 'dream' in model_name_lower:
                    model_name = 'dream'
                else:
                    # Default to llada if can't determine
                    model_name = 'llada'
                    logger.warning(f"[Worker Process] Could not determine model type from path '{model_path}', defaulting to 'llada'")

                logger.info(f"[Worker Process] Detected model_name: {model_name}")

                # Create Diffulex config
                diffulex_config = DiffulexConfig(
                    model=model_path,
                    model_name=model_name,  # Required for AutoSampler to select correct sampler
                    model_type='diffusion_lm',
                    max_model_len=max_model_len,
                    diffusion_block_size=diffusion_block_size,
                    kvcache_block_size=kvcache_block_size,  # Use adjusted value divisible by 16
                    kv_cache_layout=kv_cache_layout,  # Use CPU-compatible layout for CPU
                    mask_token_id=mask_token_id,
                    tensor_parallel_size=1,
                    max_num_seqs=len(prompts),
                    max_num_batched_tokens=max_model_len * len(prompts),
                    num_kvcache_blocks=num_kvcache_blocks,  # Set positive value
                    enforce_eager=True,  # Match vLLM setting
                )

                # Create sequences
                sequences = []
                diffulex_sampling_params_list = []

                for prompt in prompts:
                    # Tokenize prompt
                    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

                    # Create Diffulex SamplingParams
                    diffulex_params = DiffulexSamplingParams(
                        temperature=sampling_params_dict['temperature'],
                        max_tokens=sampling_params_dict['max_tokens'] or 128,
                        ignore_eos=False,
                    )
                    diffulex_sampling_params_list.append(diffulex_params)

                    # Create sequence with device parameter for CPU support
                    seq = SequenceForDiffusionLM(
                        token_ids=prompt_tokens,
                        sampling_params=diffulex_params,
                        config=diffulex_config,
                        device=device
                    )
                    sequences.append(seq)

                logger.info(f"[Worker Process] Created {len(sequences)} sequences")

                # Create scheduler
                scheduler = SchedulerForDiffusionLM(diffulex_config)
                for seq in sequences:
                    scheduler.add(seq)

                # Create sampler
                sampler = AutoSampler.from_config(diffulex_config)

                logger.info("[Worker Process] Starting diffusion generation loop...")

                # Helper functions for preparing inputs
                def prepare_prefill(seqs, block_size, device):
                    """Prepare prefill inputs for diffusion generation."""
                    import torch

                    input_ids = []
                    positions = []
                    cu_seqlens_q = [0]
                    cu_seqlens_k = [0]
                    max_seqlen_q = 0
                    max_seqlen_k = 0
                    slot_mapping = []
                    context_lens = []
                    seq_lens = []

                    for seq in seqs:
                        seq.next_diffusion_step(is_prefill=True)

                        total_seqlen = len(seq)
                        input_ids.extend(seq[seq.cached_num_tokens:])
                        positions.extend(list(range(seq.cached_num_tokens, total_seqlen)))
                        seq_lens.append(total_seqlen)
                        context_lens.append(0)

                        seqlen_q = total_seqlen - seq.cached_num_tokens
                        seqlen_k = total_seqlen
                        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

                        max_seqlen_q = max(seqlen_q, max_seqlen_q)
                        max_seqlen_k = max(seqlen_k, max_seqlen_k)

                        if not seq.block_table:
                            continue
                        for i in range(0, seq.num_prompt_blocks):
                            if seq.block_cache_missed[i]:
                                start = seq.block_table[i] * block_size
                                if i != seq.num_prompt_blocks - 1:
                                    end = start + block_size
                                else:
                                    end = start + seq.last_block_prompt_num_tokens
                                slot_mapping.extend(list(range(start, end)))
                            else:
                                slot_mapping.extend([-1] * block_size)
                        slot_mapping.extend([-1] * seq.diffusion_block_size)

                    # Convert to tensors
                    input_ids = torch.tensor(input_ids, dtype=torch.int64).to(device)
                    positions = torch.tensor(positions, dtype=torch.int64).to(device)
                    seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32).to(device)
                    context_lens = torch.tensor(context_lens, dtype=torch.int32).to(device)
                    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(device)
                    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(device)
                    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(device) if slot_mapping else torch.tensor([], dtype=torch.int32).to(device)

                    # Set up context
                    set_context_diffusion_lm(
                        is_prefill=True,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        slot_mapping=slot_mapping,
                        context_lens=context_lens,
                        block_tables=None,
                        seqs=seqs,
                        kv_cache_layout=kv_cache_layout,  # Use the calculated layout
                        seq_lens=seq_lens,
                        seq_lens_ts=seq_lens_ts,
                    )
                    return input_ids, positions

                def prepare_decode(seqs, device):
                    """Prepare decode inputs for diffusion generation."""
                    import torch

                    input_ids = []
                    positions = []
                    cu_seqlens_q = [0]
                    cu_seqlens_k = [0]
                    slot_mapping = []
                    context_lens = []
                    seq_lens = []
                    max_seqlen_q = 0
                    max_seqlen_k = 0

                    for seq in seqs:
                        seq.next_diffusion_step()
                        cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()

                        seq_lens.append(len(cur_input_ids))
                        input_ids.extend(cur_input_ids)
                        positions.extend(cur_positions)
                        context_lens.append(cur_context_len)

                        total_seqlen = len(seq)
                        seqlen_q = total_seqlen - seq.cached_num_tokens
                        seqlen_k = total_seqlen
                        max_seqlen_q = max(seqlen_q, max_seqlen_q)
                        max_seqlen_k = max(seqlen_k, max_seqlen_k)
                        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

                        slot_mapping.extend([-1] * len(cur_input_ids))

                    # Convert to tensors
                    input_ids = torch.tensor(input_ids, dtype=torch.int64).to(device)
                    positions = torch.tensor(positions, dtype=torch.int64).to(device)
                    seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32).to(device)
                    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(device)
                    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(device)
                    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(device)
                    context_lens = torch.tensor(context_lens, dtype=torch.int32).to(device)

                    # Set up context
                    set_context_diffusion_lm(
                        is_prefill=False,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        slot_mapping=slot_mapping,
                        context_lens=context_lens,
                        block_tables=None,
                        seqs=seqs,
                        seq_lens=seq_lens,
                        seq_lens_ts=seq_lens_ts,
                        kv_cache_layout=kv_cache_layout,  # Use the calculated layout
                        need_kv_cache_store=False,
                        d2f_pp=True
                    )
                    return input_ids, positions

                # Run generation loop
                import torch
                device = next(model.parameters()).device
                step_count = 0
                max_steps = 1000

                while not scheduler.is_finished() and step_count < max_steps:
                    step_count += 1
                    if step_count % 10 == 0:
                        logger.debug(f"[Worker Process] Generation step {step_count}")

                    # Schedule sequences
                    seqs, is_prefill = scheduler.schedule()
                    if not seqs:
                        break

                    # Prepare inputs
                    try:
                        if is_prefill:
                            input_ids, positions = prepare_prefill(seqs, block_size, device)
                        else:
                            input_ids, positions = prepare_decode(seqs, device)

                        if input_ids.numel() == 0:
                            break

                        # Run model forward pass
                        with torch.no_grad():
                            hidden_states = model(input_ids, positions)
                            logits = model.compute_logits(hidden_states)

                        # Debug: Log logits statistics
                        if step_count <= 3:
                            logger.debug(f"[Worker Process] Step {step_count}: logits shape={logits.shape}")
                            logger.debug(f"[Worker Process] Step {step_count}: logits min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                            logger.debug(f"[Worker Process] Step {step_count}: logits std={logits.std():.4f}")
                            # Check for top-5 predicted tokens
                            top5_logits, top5_indices = torch.topk(logits[0], k=5)
                            logger.debug(f"[Worker Process] Step {step_count}: top5 token indices={top5_indices.tolist()}")
                            logger.debug(f"[Worker Process] Step {step_count}: top5 logit values={top5_logits.tolist()}")

                        # Prepare temperatures
                        temperatures = torch.tensor(
                            [params.temperature for params in diffulex_sampling_params_list],
                            device=logits.device
                        )
                        logger.debug(f"[Worker Process] Step {step_count}: temperatures={temperatures.tolist()}")

                        # Sample tokens
                        logger.debug(f"[Worker Process] Step {step_count}: Calling sampler with logits shape={logits.shape}")
                        sample_output = sampler(logits, temperatures)
                        logger.debug(f"[Worker Process] Step {step_count}: Sampler returned type={type(sample_output).__name__}")

                        # Debug: Log sampling output structure
                        if step_count <= 5:  # Only log first 5 steps to avoid spam
                            if hasattr(sample_output, 'sampled_tokens_map'):
                                logger.debug(f"[Worker Process] Step {step_count}: sampled_tokens_map keys={list(sample_output.sampled_tokens_map.keys())}")
                                for seq_id, tokens_map in sample_output.sampled_tokens_map.items():
                                    logger.debug(f"[Worker Process] Step {step_count}: seq {seq_id} tokens_map keys={list(tokens_map.keys())}")
                                    for block_id, tokens in tokens_map.items():
                                        logger.debug(f"[Worker Process] Step {step_count}: seq {seq_id} block {block_id} tokens={tokens[:10] if hasattr(tokens, '__len__') and len(tokens) > 10 else tokens}")
                            if hasattr(sample_output, 'accepted_ids_map'):
                                logger.debug(f"[Worker Process] Step {step_count}: accepted_ids_map={sample_output.accepted_ids_map}")

                        # Update sequences
                        logger.debug(f"[Worker Process] Step {step_count}: Calling scheduler.postprocess")
                        scheduler.postprocess(seqs, sample_output)
                        logger.debug(f"[Worker Process] Step {step_count}: Postprocess completed")

                        # Debug: Check sequence state after postprocess
                        if step_count <= 3:
                            for i, seq in enumerate(seqs):
                                logger.debug(f"[Worker Process] Step {step_count}: Sequence {i} token_ids length={len(seq.token_ids)}")
                                logger.debug(f"[Worker Process] Step {step_count}: Sequence {i} last 10 tokens={seq.token_ids[-10:]}")
                                logger.debug(f"[Worker Process] Step {step_count}: Sequence {i} num_tokens={seq.num_tokens}, input_num_tokens={getattr(seq, 'input_num_tokens', 'N/A')}")

                        # Reset context
                        reset_context_diffusion_lm()

                    except Exception as e:
                        logger.error(f"[Worker Process] Error in generation step {step_count}: {e}", exc_info=True)
                        try:
                            reset_context_diffusion_lm()
                        except:
                            pass
                        raise

                logger.info(f"[Worker Process] Completed diffusion generation in {step_count} steps")

                # Extract results
                results = []
                for i, (prompt, seq) in enumerate(zip(prompts, sequences)):
                    if seq.is_finished:
                        generated_token_ids = seq.completion_token_ids
                        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    else:
                        generated_token_ids = seq.completion_token_ids if hasattr(seq, 'completion_token_ids') else []
                        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True) if generated_token_ids else ""

                    # Debug: Log completion tokens
                    logger.debug(f"[Worker Process] Sequence {i}: completion_token_ids={generated_token_ids[:20] if len(generated_token_ids) > 20 else generated_token_ids}")
                    logger.debug(f"[Worker Process] Sequence {i}: generated_text='{generated_text}'")

                    results.append({
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'model_class': model.__class__.__name__,
                        'success': True,
                        'steps': step_count,
                    })

                return results

            except Exception as e:
                logger.error(f"[Worker Process] Error in diffusion generation: {e}", exc_info=True)
                return [{
                    'prompt': p,
                    'generated_text': '',
                    'error': str(e),
                    'success': False
                } for p in prompts]

        # Monkey-patch the method onto EngineCoreProc
        EngineCoreProc.run_diffusion_generation = run_diffusion_generation
        _ENGINE_CORE_PATCHED = True
        logger.info("✓ Successfully patched EngineCoreProc with run_diffusion_generation method")

    except Exception as e:
        logger.warning(f"Failed to patch EngineCoreProc: {e}")


def is_diffusion_model(llm: LLM) -> bool:
    """Check if the LLM instance is using a diffusion model."""
    try:
        diffusion_archs = [
            "DreamForDiffusionLM",
            "LLaDAForDiffusionLM",
            "LLaDAModelLM"
        ]

        # Try to get model config from various possible locations in vLLM v1
        model_config = None

        # Method 1: Direct attribute (vLLM v0.x)
        if hasattr(llm, 'model_config'):
            model_config = llm.model_config
        # Method 2: Through engine (vLLM v1.x)
        elif hasattr(llm, 'engine') and hasattr(llm.engine, 'model_config'):
            model_config = llm.engine.model_config
        # Method 3: Through llm_engine
        elif hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_config'):
            model_config = llm.llm_engine.model_config

        # Check architecture from model config
        if model_config:
            # Try hf_config first (vLLM v1)
            if hasattr(model_config, 'hf_config'):
                hf_config = model_config.hf_config
                model_arch = getattr(hf_config, 'architectures', [])
                if isinstance(model_arch, list):
                    model_arch = model_arch[0] if model_arch else None
                if model_arch in diffusion_archs:
                    logger.info(f"✓ Detected diffusion model from hf_config.architectures: {model_arch}")
                    return True

            # Try architectures directly
            model_arch = getattr(model_config, 'architectures', [])
            if isinstance(model_arch, list):
                model_arch = model_arch[0] if model_arch else None
            if model_arch in diffusion_archs:
                logger.info(f"✓ Detected diffusion model from model_config.architectures: {model_arch}")
                return True

        # Method 4: Check the actual loaded model class name
        # Try different paths to get to the model
        model = None

        # Path 1: engine -> model_executor -> loaded_model
        if hasattr(llm, 'engine'):
            if hasattr(llm.engine, 'model_executor') and hasattr(llm.engine.model_executor, 'loaded_model'):
                model = llm.engine.model_executor.loaded_model
            elif hasattr(llm.engine, 'model'):
                model = llm.engine.model

        # Path 2: llm_engine path (v0.x)
        if model is None and hasattr(llm, 'llm_engine'):
            if hasattr(llm.llm_engine, 'model_executor'):
                model_executor = llm.llm_engine.model_executor
                if hasattr(model_executor, 'driver_worker'):
                    driver_worker = model_executor.driver_worker
                    if hasattr(driver_worker, 'model_runner') and hasattr(driver_worker.model_runner, 'model'):
                        model = driver_worker.model_runner.model

        # Check model class name
        if model is not None:
            model_class_name = model.__class__.__name__
            if any(arch in model_class_name for arch in diffusion_archs):
                logger.info(f"✓ Detected diffusion model from model class name: {model_class_name}")
                return True

        logger.debug("Not a diffusion model - none of the detection methods matched")
        return False
    except Exception as e:
        logger.debug(f"Error checking if model is diffusion model: {e}", exc_info=True)
        return False


def generate_with_diffusion(
    llm: LLM,
    prompts: Union[str, List[str]],
    sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    **kwargs
) -> List[RequestOutput]:
    """
    Generate text using diffusion model's full generation process.

    This function uses Diffulex components to run the complete diffusion
    generation process instead of vLLM's standard autoregressive loop.

    Args:
        llm: The LLM instance (must be a diffusion model)
        prompts: Input prompt(s)
        sampling_params: Sampling parameters (optional, will use defaults if not provided)
        max_tokens: Maximum tokens to generate (defaults to sampling_params.max_tokens)
        temperature: Sampling temperature (defaults to 0.2 for diffusion models)
        top_p: Top-p sampling parameter
        **kwargs: Additional generation parameters

    Returns:
        List of RequestOutput objects containing generated text
    """
    logger.info("=" * 80)
    logger.info("CUSTOM DIFFUSION GENERATION CALLED")
    logger.info("=" * 80)
    logger.info("Using custom diffusion generation method with Diffulex components")
    
    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    logger.info(f"Processing {len(prompts)} prompt(s)")

    # Check if we're using vLLM v1 engine with engine_core (multiprocess architecture)
    if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
        logger.info("Detected vLLM v1 engine with engine_core - using worker utility method")

        # Note: EngineCoreProc was already patched at import time in __init__.py
        # This ensures the worker process has the custom method

        # Prepare sampling parameters as a dict (for serialization over RPC)
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or 128,
            )

        if isinstance(sampling_params, list):
            # Use first one if list (simplification for now)
            sampling_params = sampling_params[0]

        sampling_params_dict = {
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_tokens,
            'top_p': sampling_params.top_p if hasattr(sampling_params, 'top_p') else None,
        }

        # Call our custom method in the worker process
        try:
            logger.info("Calling run_diffusion_generation in worker process...")
            results = llm.llm_engine.engine_core.call_utility(
                'run_diffusion_generation',
                prompts,
                sampling_params_dict
            )

            logger.info(f"Got results from worker: {results}")

            # Convert results to RequestOutput format
            tokenizer = llm.get_tokenizer()
            from vllm.outputs import CompletionOutput, RequestOutput as VLLMRequestOutput

            request_outputs = []
            for i, result in enumerate(results):
                if result['success']:
                    completion = CompletionOutput(
                        index=0,
                        text=result['generated_text'],
                        token_ids=tokenizer.encode(result['generated_text']),
                        cumulative_logprob=0.0,
                        logprobs=None,
                        finish_reason="length",
                    )

                    request_output = VLLMRequestOutput(
                        request_id=f"diffusion_gen_{i}",
                        prompt=result['prompt'],
                        prompt_token_ids=tokenizer.encode(result['prompt']),
                        prompt_logprobs=None,
                        outputs=[completion],
                        finished=True,
                    )
                    request_outputs.append(request_output)
                else:
                    logger.error(f"Generation failed for prompt {i}: {result.get('error')}")
                    raise RuntimeError(f"Diffusion generation failed: {result.get('error')}")

            logger.info(f"✓ Successfully completed diffusion generation via worker utility")
            return request_outputs

        except Exception as e:
            logger.error(f"Error calling worker utility: {e}", exc_info=True)
            raise RuntimeError(f"Failed to call worker utility: {e}") from e

    # Get the underlying model and config for direct access (v0 or simple v1)
    try:
        # Try vLLM v1.x structure first
        if hasattr(llm, 'engine'):
            engine = llm.engine
            # Get model from engine
            if hasattr(engine, 'model_executor') and hasattr(engine.model_executor, 'loaded_model'):
                model = engine.model_executor.loaded_model
            elif hasattr(engine, 'model'):
                model = engine.model
            else:
                raise RuntimeError("Cannot find model in engine")

            # Get model config
            if hasattr(engine, 'model_config'):
                model_config = engine.model_config
            else:
                raise RuntimeError("Cannot find model_config in engine")
        # Fallback to llm_engine structure (v0.x and newer versions)
        elif hasattr(llm, 'llm_engine'):
            # Try v0.x structure with model_executor FIRST
            # This works for vLLM 0.x versions including 0.11.0
            if hasattr(llm.llm_engine, 'model_executor'):
                model_executor = llm.llm_engine.model_executor
                driver_worker = model_executor.driver_worker
                model_runner = driver_worker.model_runner
                model = model_runner.model
            elif hasattr(llm.llm_engine, 'model'):
                # Direct model access
                model = llm.llm_engine.model
            else:
                raise RuntimeError(
                    f"Cannot find model in llm_engine. "
                    f"Available attributes: {dir(llm.llm_engine)}"
                )

            # Get model config
            if hasattr(llm, 'model_config'):
                model_config = llm.model_config
            elif hasattr(llm.llm_engine, 'model_config'):
                model_config = llm.llm_engine.model_config
            else:
                raise RuntimeError("Cannot find model_config")
        else:
            raise RuntimeError("Cannot find engine in LLM object")

        # Get tokenizer
        tokenizer = llm.get_tokenizer()

        logger.info(f"✓ Successfully accessed model: {model.__class__.__name__}")
        logger.info(f"✓ Successfully accessed model_config")

    except Exception as e:
        logger.error(f"Failed to access model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to access model: {e}") from e
    
    # Get generation parameters
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or 128,  # Default for diffusion models
        )
    
    if isinstance(sampling_params, SamplingParams):
        sampling_params = [sampling_params] * len(prompts)
    
    # Get diffusion-specific parameters from model config
    hf_config = model_config.hf_config
    max_model_len = getattr(model_config, 'max_model_len', 4096)
    
    # Get diffusion config parameters
    # Check for environment variable override first
    import os
    diffusion_steps_env = os.environ.get('DLLM_DIFFUSION_STEPS')
    if diffusion_steps_env is not None:
        diffusion_steps = int(diffusion_steps_env)
    else:
        diffusion_steps = getattr(hf_config, 'diffusion_steps', 128)
    block_size = getattr(hf_config, 'block_size', 4)
    diffusion_block_size = getattr(hf_config, 'diffusion_block_size', 32)
    mask_token_id = getattr(hf_config, 'mask_token_id', None)
    
    if mask_token_id is None:
        # Try to get from tokenizer
        try:
            mask_token_id = tokenizer.mask_token_id
        except:
            # Default mask token IDs for common models
            mask_token_id = 126336  # LLaDA default
    
    logger.info(
        f"Diffusion generation config: steps={diffusion_steps}, "
        f"block_size={block_size}, diffusion_block_size={diffusion_block_size}, "
        f"mask_token_id={mask_token_id}, max_model_len={max_model_len}"
    )
    
    
    try:
        from d2f_engine.sampling_params import SamplingParams as DiffulexSamplingParams
        from d2f_engine.config import Config as DiffulexConfig
        from d2f_engine.engine.sequence import SequenceForDiffusionLM
        from d2f_engine.engine.scheduler import SchedulerForDiffusionLM
        from d2f_engine.engine.model_runner import ModelRunnerForDiffusionLM
        from d2f_engine.utils.context import set_context_diffusion_lm, get_context_diffusion_lm
        
        # Get model path for config
        model_path = getattr(model_config, 'model', None) or getattr(hf_config, '_name_or_path', None)
        if model_path is None:
            model_path = "GSAI-ML/LLaDA-8B-Instruct"  # Default fallback
        
        # Create Diffulex config
        diffulex_config = DiffulexConfig(
            model=model_path,
            model_type='diffusion_lm',
            max_model_len=max_model_len,
            diffusion_block_size=diffusion_block_size,
            block_size=block_size,
            mask_token_id=mask_token_id,
            tensor_parallel_size=1,
            max_num_seqs=len(prompts),
            max_num_batched_tokens=max_model_len * len(prompts),
        )
        
        # Create sequences
        sequences = []
        diffulex_sampling_params_list = []
        for prompt, params in zip(prompts, sampling_params):
            # Tokenize prompt
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            
            # Create Diffulex SamplingParams
            diffulex_params = DiffulexSamplingParams(
                temperature=params.temperature,
                max_tokens=params.max_tokens or 128,
                ignore_eos=params.ignore_eos if hasattr(params, 'ignore_eos') else False,
            )
            diffulex_sampling_params_list.append(diffulex_params)
            
            # Create sequence
            seq = SequenceForDiffusionLM(
                token_ids=prompt_tokens,
                sampling_params=diffulex_params,
                config=diffulex_config
            )
            sequences.append(seq)
        
        # Create scheduler
        scheduler = SchedulerForDiffusionLM(diffulex_config)
        for seq in sequences:
            scheduler.add(seq)
        
        # Create model runner with our existing model
        # Note: This is a simplified approach - in practice, ModelRunner expects
        # to load the model itself, but we'll try to work around this
        logger.info("Running diffusion generation loop...")
        
        # Helper function to prepare prefill inputs
        def prepare_prefill(seqs):
            import torch
            device = next(model.parameters()).device

            input_ids = []
            positions = []
            cu_seqlens_q = [0]
            cu_seqlens_k = [0]
            max_seqlen_q = 0
            max_seqlen_k = 0
            slot_mapping = []
            context_lens = []
            seq_lens = []

            for seq in seqs:
                seq.next_diffusion_step(is_prefill=True)

                total_seqlen = len(seq)
                # tokens and positions to run in this prefill step
                input_ids.extend(seq[seq.cached_num_tokens:])
                positions.extend(list(range(seq.cached_num_tokens, total_seqlen)))
                seq_lens.append(total_seqlen)
                context_lens.append(0)

                seqlen_q = total_seqlen - seq.cached_num_tokens
                seqlen_k = total_seqlen
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                # Simplified slot mapping for non-KV cache case
                if not seq.block_table:
                    continue
                for i in range(0, seq.num_prompt_blocks):
                    if seq.block_cache_missed[i]:
                        start = seq.block_table[i] * block_size
                        if i != seq.num_prompt_blocks - 1:
                            end = start + block_size
                        else:
                            end = start + seq.last_block_prompt_num_tokens
                        slot_mapping.extend(list(range(start, end)))
                    else:
                        slot_mapping.extend([-1] * block_size)
                slot_mapping.extend([-1] * seq.diffusion_block_size)

            # Convert to tensors
            input_ids = torch.tensor(input_ids, dtype=torch.int64).to(device)
            positions = torch.tensor(positions, dtype=torch.int64).to(device)
            seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32).to(device)
            context_lens = torch.tensor(context_lens, dtype=torch.int32).to(device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(device)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(device)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(device) if slot_mapping else torch.tensor([], dtype=torch.int32).to(device)
            block_tables = None  # Simplified - would need proper block table preparation

            # Set up proper context
            set_context_diffusion_lm(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                seqs=seqs,
                kv_cache_layout="unified",
                seq_lens=seq_lens,
                seq_lens_ts=seq_lens_ts,
            )
            return input_ids, positions

        # Helper function to prepare decode inputs
        def prepare_decode(seqs):
            import torch
            device = next(model.parameters()).device

            input_ids = []
            positions = []
            cu_seqlens_q = [0]
            cu_seqlens_k = [0]
            slot_mapping = []
            context_lens = []
            seq_lens = []
            max_seqlen_q = 0
            max_seqlen_k = 0

            for seq in seqs:
                seq.next_diffusion_step()
                cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()

                seq_lens.append(len(cur_input_ids))
                input_ids.extend(cur_input_ids)
                positions.extend(cur_positions)
                context_lens.append(cur_context_len)

                total_seqlen = len(seq)
                seqlen_q = total_seqlen - seq.cached_num_tokens
                seqlen_k = total_seqlen
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

                # Simplified slot mapping
                slot_mapping.extend([-1] * len(cur_input_ids))

            # Convert to tensors
            input_ids = torch.tensor(input_ids, dtype=torch.int64).to(device)
            positions = torch.tensor(positions, dtype=torch.int64).to(device)
            seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32).to(device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(device)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(device)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(device)
            context_lens = torch.tensor(context_lens, dtype=torch.int32).to(device)
            block_tables = None  # Simplified

            # Set up proper context
            set_context_diffusion_lm(
                is_prefill=False,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                seqs=seqs,
                seq_lens=seq_lens,
                seq_lens_ts=seq_lens_ts,
                kv_cache_layout="unified",
                need_kv_cache_store=False,
                d2f_pp=True
            )
            return input_ids, positions

        # Create sampler once
        from d2f_engine.layers.sampler import AutoSampler
        sampler = AutoSampler.from_config(diffulex_config)

        # Run generation loop
        results = []
        step_count = 0
        max_steps = 1000  # Safety limit

        logger.info(f"Starting diffusion generation loop (max {max_steps} steps)")

        while not scheduler.is_finished() and step_count < max_steps:
            step_count += 1
            logger.debug(f"Generation step {step_count}")

            # Schedule sequences
            seqs, is_prefill = scheduler.schedule()
            if not seqs:
                logger.debug("No sequences scheduled, breaking")
                break

            logger.debug(f"Scheduled {len(seqs)} sequences, is_prefill={is_prefill}")

            # Prepare inputs with proper context setup
            try:
                import torch

                if is_prefill:
                    input_ids, positions = prepare_prefill(seqs)
                else:
                    input_ids, positions = prepare_decode(seqs)

                if input_ids.numel() == 0:
                    logger.debug("No input tokens, breaking")
                    break

                logger.debug(f"Prepared {input_ids.numel()} input tokens")

                # Run model forward pass
                logger.debug(f"Running model forward pass...")
                with torch.no_grad():
                    hidden_states = model(input_ids, positions)
                    logits = model.compute_logits(hidden_states)

                logger.debug(f"Model forward pass completed, logits shape: {logits.shape}")

                # Prepare temperatures
                temperatures = torch.tensor([params.temperature for params in diffulex_sampling_params_list],
                                          device=logits.device)

                # Sample tokens using proper diffusion sampler
                logger.debug("Calling diffusion sampler...")
                sample_output = sampler(logits, temperatures)
                logger.debug("Sampler completed, got sample_output")

                # Update sequences using scheduler.postprocess
                logger.debug("Calling scheduler.postprocess...")
                scheduler.postprocess(seqs, sample_output)
                logger.debug("Scheduler.postprocess completed")

                # Reset context after each step
                from d2f_engine.utils.context import reset_context_diffusion_lm
                reset_context_diffusion_lm()

                logger.debug(f"Updated sequences, finished={[s.is_finished for s in seqs]}")

            except Exception as e:
                logger.error(f"Error in generation step {step_count}: {e}", exc_info=True)
                try:
                    from d2f_engine.utils.context import reset_context_diffusion_lm
                    reset_context_diffusion_lm()
                except:
                    pass
                raise

        if step_count >= max_steps:
            logger.warning(f"Generation loop reached max steps ({max_steps}), stopping")
        
        # Extract results
        for prompt, seq in zip(prompts, sequences):
            if seq.is_finished:
                generated_token_ids = seq.completion_token_ids
                generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            else:
                # Sequence didn't finish - return what we have
                generated_token_ids = seq.completion_token_ids if hasattr(seq, 'completion_token_ids') else []
                generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True) if generated_token_ids else ""
            
            from vllm.outputs import CompletionOutput, RequestOutput as VLLMRequestOutput
            
            completion = CompletionOutput(
                index=0,
                text=generated_text,
                token_ids=generated_token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="length" if seq.is_finished else "length",
            )
            
            request_output = VLLMRequestOutput(
                request_id="diffusion_gen",
                prompt=prompt,
                prompt_token_ids=tokenizer.encode(prompt, add_special_tokens=False),
                prompt_logprobs=None,
                outputs=[completion],
                finished=seq.is_finished,
            )
            
            results.append(request_output)
        
        logger.info(f"Completed diffusion generation in {step_count} steps")
        return results
        
    except ImportError as e:
        logger.error(f"Failed to import Diffulex components: {e}")
        logger.error("Make sure Diffulex is installed and in the Python path")
        raise
    except Exception as e:
        logger.error(f"Failed to use Diffulex generation: {e}", exc_info=True)
        raise RuntimeError(
            "Failed to run diffusion generation. "
            "The full Diffulex engine integration requires additional setup. "
            f"Error: {e}"
        ) from e


def patch_llm_generate(llm: LLM):
    """
    Patch the LLM instance's generate method to use custom diffusion generation.
    
    This function monkey-patches the generate method to detect diffusion models
    and use the custom generation path.
    """
    original_generate = llm.generate
    
    def patched_generate(
        prompts,
        sampling_params=None,
        use_tqdm=True,
        lora_request=None,
        priority=None,
    ):
        # IMPORTANT: This function should be called if patching worked
        logger.info("=" * 80)
        logger.info("PATCHED GENERATE METHOD CALLED!")
        logger.info("=" * 80)
        
        # Check if this is a diffusion model
        logger.info("patched_generate called - checking if diffusion model...")
        is_diffusion = is_diffusion_model(llm)
        logger.info(f"is_diffusion_model returned: {is_diffusion}")
        
        if is_diffusion:
            logger.info("✓ Diffusion model detected, using custom generation")
            logger.info("Detected diffusion model, using custom generation")
            try:
                return generate_with_diffusion(
                    llm=llm,
                    prompts=prompts,
                    sampling_params=sampling_params,
                )
            except Exception as e:
                logger.error(f"Custom diffusion generation failed: {e}", exc_info=True)
                logger.warning("Falling back to standard vLLM generation")
                return original_generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                    lora_request=lora_request,
                    priority=priority,
                )
        else:
            # Use standard vLLM generation
            logger.info("Not a diffusion model, using standard vLLM generation")
            logger.info("This means the patching is working, but model detection failed")
            return original_generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
                priority=priority,
            )
    
    # Verify the original generate is not already patched
    if hasattr(llm.generate, '__name__') and llm.generate.__name__ == 'patched_generate':
        logger.warning("LLM.generate() was already patched - skipping")
        return
    
    llm.generate = patched_generate
    logger.info("✓ Patched LLM.generate() to use custom diffusion generation")
    
    # Verify patching worked
    if hasattr(llm.generate, '__name__') and llm.generate.__name__ == 'patched_generate':
        logger.info("✓ Verification: LLM.generate() is now patched")
    else:
        logger.error("✗ Verification FAILED: LLM.generate() was not patched correctly!")
