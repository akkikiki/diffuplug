"""
Test script to verify the patching is working correctly.
"""
import logging
import sys

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Register plugin
    logger.info("Step 1: Registering plugin...")
    import dllm_plugin
    dllm_plugin.register()
    logger.info("✓ Plugin registered")

    # Import vLLM
    logger.info("Step 2: Importing vLLM...")
    from vllm import LLM

    # Check if LLM.__init__ was patched
    logger.info(f"Step 3: LLM.__init__ function: {LLM.__init__}")
    if hasattr(LLM.__init__, '__name__'):
        logger.info(f"LLM.__init__.__name__ = {LLM.__init__.__name__}")

    # Create LLM instance (this should trigger patching)
    logger.info("Step 4: Creating LLM instance (this should trigger instance patching)...")
    try:
        llm = LLM(
            model="GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            enforce_eager=True,
        )
        logger.info("✓ LLM instance created")
    except Exception as e:
        logger.error(f"Failed to create LLM instance: {e}", exc_info=True)
        sys.exit(1)

    # Check if generate was patched
    logger.info("Step 5: Checking if generate method was patched...")
    if hasattr(llm.generate, '__name__'):
        method_name = llm.generate.__name__
        logger.info(f"llm.generate.__name__ = '{method_name}'")
        if 'patched' in method_name or 'patch' in method_name:
            logger.info("✓✓✓ SUCCESS! Generate method was patched!")
        else:
            logger.warning(f"⚠ WARNING: Generate method name '{method_name}' doesn't contain 'patched'")
    else:
        logger.warning("⚠ WARNING: llm.generate doesn't have __name__ attribute")

    # Check generate method type
    logger.info(f"llm.generate type: {type(llm.generate)}")
    logger.info(f"llm.generate: {llm.generate}")

    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*80)
