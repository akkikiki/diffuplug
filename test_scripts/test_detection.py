"""
Simple test to check if diffusion model detection works.
"""
import logging
import sys

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Register plugin
    logger.info("Registering plugin...")
    import dllm_plugin
    dllm_plugin.register()

    # Import vLLM
    from vllm import LLM

    # Create LLM instance
    logger.info("Creating LLM instance...")
    llm = LLM(
        model="GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        enforce_eager=True,
    )
    logger.info("✓ LLM instance created")

    # Import detection function
    from dllm_plugin.generation import is_diffusion_model

    # Test detection
    logger.info("Testing diffusion model detection...")
    is_diff = is_diffusion_model(llm)
    logger.info(f"is_diffusion_model returned: {is_diff}")

    if is_diff:
        logger.info("✓✓✓ SUCCESS! Diffusion model was detected!")
    else:
        logger.error("✗✗✗ FAILED! Diffusion model was NOT detected!")
        # Debug: print LLM attributes
        logger.info(f"LLM attributes: {dir(llm)}")
        if hasattr(llm, 'engine'):
            logger.info(f"LLM.engine attributes: {dir(llm.engine)}")
            if hasattr(llm.engine, 'model_config'):
                logger.info(f"Found engine.model_config")
                config = llm.engine.model_config
                if hasattr(config, 'hf_config'):
                    logger.info(f"hf_config architectures: {getattr(config.hf_config, 'architectures', None)}")
