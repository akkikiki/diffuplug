#!/usr/bin/env python3
"""
Test the vLLM integration with the new LLaDASampler.
"""

import sys
import os
import logging

# Setup logging to see all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Import and register the plugin
    logger.info("=" * 80)
    logger.info("Step 1: Importing and registering plugin...")
    logger.info("=" * 80)

    from dllm_plugin import register
    register()

    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating vLLM instance...")
    logger.info("=" * 80)

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model='GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            dtype='float32',
            enforce_eager=True,
            max_model_len=2048,
        )
        logger.info("✓ vLLM instance created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create vLLM instance: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Testing generation...")
    logger.info("=" * 80)

    test_prompts = ["What is 2+2?"]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    logger.info(f"Test prompt: '{test_prompts[0]}'")
    logger.info(f"Sampling params: temp={sampling_params.temperature}, max_tokens={sampling_params.max_tokens}")

    try:
        logger.info("\nGenerating...")
        outputs = llm.generate(test_prompts, sampling_params)

        logger.info("\n" + "-" * 80)
        logger.info("RESULTS:")
        logger.info("-" * 80)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: '{generated_text}'")

            # Check quality
            if len(generated_text.strip()) > 0:
                logger.info("✓ Generated non-empty text")

                if '4' in generated_text or 'four' in generated_text.lower():
                    logger.info("✅ Output appears correct (contains answer '4')")
                else:
                    logger.info("⚠ Output doesn't contain expected answer")
            else:
                logger.info("✗ Generated empty text")

        logger.info("-" * 80)

    except Exception as e:
        logger.error(f"✗ Generation failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("Test complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
