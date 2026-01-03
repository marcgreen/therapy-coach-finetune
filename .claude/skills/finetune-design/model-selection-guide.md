# Model Selection Guide

How to choose a base model for fine-tuning multi-turn conversations.

---

## Selection Factors

### 1. Context Window

The context window determines the maximum conversation length you can train on.

| Context Window | Practical Limit | Use Case |
|----------------|-----------------|----------|
| 8K | ~4K tokens/example | Short conversations (10-15 turns) |
| 32K | ~16K tokens/example | Medium conversations (25-40 turns) |
| 128K | ~60K tokens/example | Long conversations (50+ turns) |

**Why "practical limit" is lower than window:**
- Need buffer for model overhead
- Very long examples train inefficiently
- Cost scales with token count

### 2. Quantization Support

For local deployment, you need quantization:

| Format | Platform | Notes |
|--------|----------|-------|
| GGUF | llama.cpp, Ollama | Most portable, wide model support |
| MLX | Apple Silicon | Fast on Mac, limited models |
| QAT (Q4) | Various | Quantization-aware training, best quality |

**Recommendation:** Prioritize models with good GGUF support unless you're Apple-only.

### 3. Base Capability

Evaluate the model on your rubric BEFORE fine-tuning:
- If base model passes >70%, fine-tuning may not be needed
- If base model passes <50%, fine-tuning should help significantly
- This evaluation informs your expected improvement

### 4. Parameter Size vs. Speed

| Size | VRAM (4-bit) | Speed (M3 Max) | Quality |
|------|--------------|----------------|---------|
| 7B | ~4GB | ~50 tok/s | Good for simple tasks |
| 12B | ~8GB | ~30 tok/s | Good balance |
| 27B+ | ~16GB+ | ~15 tok/s | Best quality, slower |

**Trade-off:** Larger models are better but slower. For real-time conversation, 12B is often the sweet spot.

### 5. Training Method Support

| Method | Memory | Quality | Speed |
|--------|--------|---------|-------|
| Full fine-tune | Very high | Best | Slow |
| LoRA | Medium | Good | Fast |
| QLoRA | Low | Good | Fast |

**Recommendation:** QLoRA for most projects — good quality with manageable memory.

---

## Token Economics

**Critical constraint often discovered late in projects.**

### The 16K Rule

Training cost scales with tokens per example:

| Tokens/Example | Relative Cost | Notes |
|----------------|---------------|-------|
| 4K | 1x | Cheap, short conversations |
| 8K | 2x | Reasonable |
| 16K | 4x | **Practical ceiling** |
| 32K | 8x | Expensive |
| 64K+ | 16x+ | Very expensive, may need special handling |

**Why 16K is the practical ceiling:**
- HuggingFace Jobs pricing scales linearly
- Gradient accumulation becomes unwieldy
- Diminishing returns on very long context

### Planning Conversation Length

Work backwards from token budget:

```
Target: 16K tokens/example max

Typical conversation:
- System prompt: ~500 tokens
- Per exchange: ~400-600 tokens (user + assistant)
- Buffer: ~1K tokens

Available for conversation: ~14K tokens
At 500 tokens/exchange: ~28 exchanges max
With safety margin: ~20-25 exchanges recommended
```

**Lesson:** Decide token budget early. It affects conversation design and training requirements.

---

## Model Comparison Framework

When evaluating candidates, score on:

| Factor | Weight | How to Evaluate |
|--------|--------|-----------------|
| Context window | High | Fits your conversation length? |
| Base capability | High | Run 20 test scenarios, check pass rate |
| Quantization | Medium | GGUF available? Quality loss acceptable? |
| Speed | Medium | Acceptable for your use case? |
| Community | Low | Active development, issues resolved? |

---

## Example: Therapy Project

**Selected:** Gemma 3 12B IT

| Factor | Gemma 3 12B | Why It Worked |
|--------|-------------|---------------|
| Context | 128K | Supports very long conversations |
| Base capability | 80% pass rate | Strong baseline for domain |
| Quantization | QAT Q4 (7.5GB GGUF) | Runs well on Mac |
| Speed | 31.5 tok/s on M3 Max | Acceptable for conversation |
| Training | QLoRA support | Trainable on A10G |

**Gotcha discovered:** Gemma 3 has 262K vocabulary (for vision-language support). This causes OOM issues:
- Logits computed in full precision regardless of model quantization
- At 4096 tokens: 4.3GB just for logits
- Solution: Reduce `max_length` to 2048 on A10G

**Lesson:** Large vocabularies have hidden memory costs. Test before committing.

---

## Decision Checklist

Before finalizing model choice:

- [ ] Context window fits your target conversation length
- [ ] Base model evaluated on your rubric (pass rate documented)
- [ ] Quantized version available in your target format
- [ ] Speed acceptable for your use case
- [ ] Training memory fits your GPU (check vocab size!)
- [ ] Token budget defined (recommend ≤16K/example)

---

## Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Choosing by benchmark scores | Benchmarks don't reflect your task | Evaluate on YOUR rubric |
| Ignoring vocab size | Hidden OOM issues | Check vocab, test memory |
| Maximizing context window | Diminishing returns, more memory | Match to actual need |
| Skipping base evaluation | Don't know if fine-tuning helps | Always evaluate first |
