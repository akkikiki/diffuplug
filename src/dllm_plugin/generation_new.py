"""
Simplified diffusion generation using LLaDASampler.

This module provides a clean interface for diffusion generation using the
reference LLaDA algorithm via LLaDASampler.
"""

import logging
from typing import List, Optional, Union

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from .llada_sampler import LLaDASampler

logger = logging.getLogger(__name__)


def patch_engine_core_for_diffusion_simple():
    """
    Simplified patch for EngineCoreProc using LLaDASampler.

    This replaces the complex Diffulex integration with a clean
    implementation using the reference LLaDA algorithm.
    """
    try:
        from vllm.v1.engine.core import EngineCoreProc

        def run_diffusion_generation_simple(
            self,
            prompts: List[str],
            sampling_params_dict: dict,
            diffusion_config: dict = None
        ) -> List[dict]:
            """
            Simplified diffusion generation using LLaDASampler.

            Args:
                prompts: List of input prompts
                sampling_params_dict: Dict with sampling parameters
                diffusion_config: Dict with diffusion-specific config

            Returns:
                List of dicts with generated text and metadata
            """
            if diffusion_config is None:
                diffusion_config = {}

            # Direct print to stderr to verify function is called
            import sys
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[WORKER DEBUG] run_diffusion_generation_simple CALLED", file=sys.stderr)
            print(f"[WORKER DEBUG] Number of prompts: {len(prompts)}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            sys.stderr.flush()

            # Configure logging for worker process
            import logging
            worker_logger = logging.getLogger('dllm_plugin.generation_new')
            if not worker_logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('(Worker) %(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                worker_logger.addHandler(handler)
                worker_logger.setLevel(logging.INFO)
                worker_logger.propagate = False

            # Also configure logger for llada_sampler
            sampler_logger = logging.getLogger('dllm_plugin.llada_sampler')
            if not sampler_logger.handlers:
                sampler_handler = logging.StreamHandler(sys.stderr)
                sampler_handler.setLevel(logging.INFO)
                sampler_handler.setFormatter(formatter)
                sampler_logger.addHandler(sampler_handler)
                sampler_logger.setLevel(logging.INFO)
                sampler_logger.propagate = False

            logger.info(
                f"[Worker Process] Running simple diffusion generation "
                f"for {len(prompts)} prompts"
            )

            try:
                # Get model and tokenizer
                executor = self.model_executor

                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker
                elif hasattr(executor, 'worker'):
                    worker = executor.worker
                else:
                    worker = executor

                model = worker.get_model()
                logger.info(f"[Worker Process] Got model: {type(model).__name__}")

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

                # Get tokenizer
                # Note: tokenizer might not be directly accessible in worker
                # We'll use the tokenizer from the model config
                try:
                    from transformers import AutoTokenizer
                    # Get model path from config
                    model_config = self.vllm_config.model_config
                    model_path = getattr(model_config, 'model', None) or \
                                 getattr(model_config.hf_config, '_name_or_path', None)
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    tokenizer.padding_side = 'left'
                    logger.info(f"[Worker Process] Loaded tokenizer from: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer: {e}")
                    tokenizer = None

                # Create LLaDASampler
                temperature = sampling_params_dict.get('temperature', 0.0)
                sampler = LLaDASampler(
                    mask_token_id=126336,
                    temperature=temperature,
                    remasking='low_confidence',
                    confidence_eos_eot_inf=True,  # Prevent premature EOS generation
                )
                logger.info(f"[Worker Process] Created LLaDASampler")

                # Process each prompt
                results = []
                for prompt in prompts:
                    try:
                        # Tokenize prompt
                        if tokenizer:
                            encoded = tokenizer(
                                prompt,
                                add_special_tokens=False,
                                return_tensors="pt"
                            )
                            prompt_ids = encoded['input_ids'].to(device)
                            attention_mask = encoded.get('attention_mask')
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(device)
                        else:
                            # Fallback: raise error since we need tokenizer
                            raise RuntimeError("Tokenizer could not be loaded and is required for generation")

                        # Get generation parameters
                        max_tokens = sampling_params_dict.get('max_tokens', 128)
                        gen_length = min(max_tokens, 128)  # Limit for CPU performance
                        block_length = diffusion_config.get('diffusion_block_size', 32)

                        # Adjust block_length and gen_length to be compatible
                        if gen_length < block_length:
                            # If requested gen_length is smaller than block_length, cap block_length to gen_length
                            block_length = gen_length
                            logger.info(
                                f"[Worker Process] Adjusted block_length from {diffusion_config.get('diffusion_block_size', 32)} "
                                f"to {block_length} to match requested gen_length={gen_length}"
                            )
                        elif gen_length % block_length != 0:
                            # Only round up if gen_length is greater than block_length
                            gen_length = ((gen_length // block_length) + 1) * block_length
                            logger.info(
                                f"[Worker Process] Rounded gen_length up to {gen_length} "
                                f"(nearest multiple of block_length={block_length})"
                            )

                        # Get diffusion steps from config
                        num_blocks = gen_length // block_length
                        configured_steps = diffusion_config.get('diffusion_steps', 128)

                        # Calculate steps: use configured steps, but ensure it's divisible by num_blocks
                        if configured_steps % num_blocks == 0:
                            steps = configured_steps
                        else:
                            # Round to nearest multiple of num_blocks
                            steps_per_block = max(1, configured_steps // num_blocks)
                            steps = steps_per_block * num_blocks
                            logger.info(
                                f"[Worker Process] Adjusted steps from {configured_steps} to {steps} "
                                f"(must be divisible by {num_blocks} blocks)"
                            )

                        logger.info(
                            f"[Worker Process] Generating: "
                            f"gen_length={gen_length}, "
                            f"block_length={block_length}, "
                            f"steps={steps}"
                        )

                        # Log initial state
                        logger.info("[Worker Process] ===== INITIAL STATE =====")
                        logger.info(f"[Worker Process] Prompt: '{prompt}'")
                        logger.info(f"[Worker Process] Prompt token IDs (first 20): {prompt_ids[0].tolist()[:20]}")
                        logger.info(f"[Worker Process] Prompt length: {prompt_ids.shape[1]}")
                        logger.info(f"[Worker Process] Generation length: {gen_length}")
                        logger.info(f"[Worker Process] Mask token ID: {sampler.mask_token_id}")

                        # Show what the initial sequence looks like (prompt + masks)
                        import torch
                        mask_tokens = torch.full((1, gen_length), sampler.mask_token_id, dtype=torch.long, device=device)
                        initial_sequence = torch.cat([prompt_ids, mask_tokens], dim=1)
                        logger.info(f"[Worker Process] Initial sequence shape: {initial_sequence.shape}")
                        logger.info(f"[Worker Process] Initial sequence (first 30 tokens): {initial_sequence[0].tolist()[:30]}")
                        logger.info(f"[Worker Process] Initial sequence (last 20 tokens, all masks): {initial_sequence[0].tolist()[-20:]}")
                        logger.info(f"[Worker Process] Device: {device}")
                        logger.info("[Worker Process] ========================")

                        # Generate
                        output_ids = sampler.generate(
                            model=model,
                            prompt_ids=prompt_ids,
                            attention_mask=attention_mask,
                            steps=steps,
                            gen_length=gen_length,
                            block_length=block_length,
                            tokenizer=tokenizer
                        )

                        # Decode output
                        if tokenizer:
                            generated_text = tokenizer.decode(
                                output_ids[0, prompt_ids.shape[1]:],
                                skip_special_tokens=True
                            )
                        else:
                            generated_text = model.tokenizer.decode(
                                output_ids[0, prompt_ids.shape[1]:],
                                skip_special_tokens=True
                            )

                        logger.info(
                            f"[Worker Process] Generated text: '{generated_text[:100]}...'"
                        )

                        results.append({
                            'prompt': prompt,
                            'generated_text': generated_text,
                            'model_class': model.__class__.__name__,
                            'success': True,
                            'steps': steps,
                        })

                    except Exception as e:
                        logger.error(
                            f"[Worker Process] Error generating for prompt: {e}",
                            exc_info=True
                        )
                        results.append({
                            'prompt': prompt,
                            'generated_text': '',
                            'error': str(e),
                            'success': False
                        })

                return results

            except Exception as e:
                logger.error(
                    f"[Worker Process] Error in diffusion generation: {e}",
                    exc_info=True
                )
                return [{
                    'prompt': p,
                    'generated_text': '',
                    'error': str(e),
                    'success': False
                } for p in prompts]

        # Monkey-patch the method
        EngineCoreProc.run_diffusion_generation = run_diffusion_generation_simple

        logger.info(
            "✓ Successfully patched EngineCoreProc with simplified "
            "run_diffusion_generation using LLaDASampler"
        )

    except Exception as e:
        logger.warning(f"Failed to patch EngineCoreProc: {e}", exc_info=True)


# Keep the same interface as the original generation.py
def is_diffusion_model(llm: LLM) -> bool:
    """Check if the LLM instance is using a diffusion model."""
    # Import from original module
    from .generation import is_diffusion_model as _is_diffusion_model
    return _is_diffusion_model(llm)


def patch_llm_generate(llm_instance: LLM):
    """
    Patch an LLM instance's generate method to use diffusion generation.

    This wraps the original generate method and checks if it's a diffusion model.
    If so, it uses the custom diffusion generation. Otherwise, falls back to original.
    """
    original_generate = llm_instance.generate

    def patched_generate(
        prompts: Union[str, List[str]],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        **kwargs
    ) -> List[RequestOutput]:
        """Patched generate that detects and handles diffusion models."""
        logger.info("patched_generate called - checking if diffusion model...")

        # Check if this is a diffusion model
        is_diff = is_diffusion_model(llm_instance)
        logger.info(f"is_diffusion_model returned: {is_diff}")

        if is_diff:
            logger.info("Detected diffusion model, using custom generation")
            # Import from original module to reuse the implementation
            from .generation import generate_with_diffusion
            return generate_with_diffusion(
                llm_instance,
                prompts,
                sampling_params,
                **kwargs
            )
        else:
            logger.info("Not a diffusion model, using original generate")
            return original_generate(prompts, sampling_params, **kwargs)

    # Replace the instance method
    llm_instance.generate = patched_generate

    logger.info("✓ Patched LLM.generate() to use custom diffusion generation")
    logger.info("✓ Verification: LLM.generate() is now patched")
