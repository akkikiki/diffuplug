# LLaDA Generation Implementation Comparison

## Original LLaDA Implementation (from user)

Key features:
1. **Gumbel Noise with Temperature**: Uses float64 precision for Gumbel-Max sampling
2. **Block-based Generation**: Generates in blocks (e.g., 32 tokens at a time)
3. **Remasking Strategy**: 'low_confidence' or 'random' - unmasks tokens based on confidence
4. **num_transfer_tokens**: Precomputes how many tokens to unmask at each step (uniform linear schedule)
5. **Confidence-based Selection**: Uses softmax probabilities to select which tokens to unmask
6. **Special Token Handling**: Optional handling of EOS/EoT tokens

## vLLM/Diffulex Integration

The vLLM integration uses Diffulex's `AutoSampler` which may not implement the exact LLaDA algorithm.

### Potential Issues:

1. **Different Sampling Algorithm**: Diffulex AutoSampler might use a different diffusion sampling approach
2. **Missing Gumbel Noise**: May not apply Gumbel noise correctly or at all
3. **Wrong Remasking Strategy**: May not use confidence-based remasking
4. **Incompatible Block Structure**: Block sizes or scheduling might differ

### Next Steps:

1. Check if Diffulex's LLaDA sampler implements the reference algorithm
2. Verify Gumbel noise is being applied
3. Check if confidence-based token selection is used
4. Compare num_transfer_tokens computation

## Recommendation

The vLLM integration should either:
- Use the reference implementation directly
- Verify Diffulex's sampler matches the reference exactly
- Or implement a custom LLaDA sampler that follows the reference
