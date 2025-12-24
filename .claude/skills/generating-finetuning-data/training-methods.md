# Training Methods

Choose your training method based on what you're trying to achieve and what data you can generate.

## Decision Guide

```
What are you trying to do?
│
├─▶ Teach new capability/format/style
│   └─▶ SFT (Supervised Fine-Tuning)
│
├─▶ Improve quality on existing capability
│   │
│   ├─▶ Can generate preference pairs?
│   │   └─▶ DPO (Direct Preference Optimization)
│   │
│   └─▶ Only have binary good/bad labels?
│       └─▶ KTO (Kahneman-Tversky Optimization)
│
└─▶ Want online learning with reward signal?
    └─▶ GRPO (Group Relative Policy Optimization)
```

---

## SFT: Supervised Fine-Tuning

**When to use**: Teaching the model new behaviors, formats, or styles it doesn't already exhibit.

**Data format**:
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "User input here"},
  {"role": "assistant", "content": "Ideal response here"}
]}
```

**Generation approach**:
1. Generate diverse inputs from taxonomy
2. Generate one high-quality response per input (strong model, good prompt)
3. Evaluate and filter
4. Format for training

**Quality threshold**: High (0.80+). You're showing the model exactly what to do.

**Typical volume**: 1K-5K examples. Quality matters more than quantity.

---

## DPO: Direct Preference Optimization

**When to use**: The model already does the task but you want to improve subtle quality aspects. Requires paired examples where one response is clearly better.

**Data format**:
```json
{
  "prompt": "User input here",
  "chosen": "Better response",
  "rejected": "Worse response"
}
```

### Generating Preference Pairs

The key insight: **The rejected response should be plausibly wrong, not obviously terrible.** If the difference is too stark, the model learns little. If too subtle, signal is weak.

**Strategy 1: Strong vs. Weak Model**
```
chosen  = strong_model.generate(prompt)  # Claude Opus, GPT-4
rejected = weak_model.generate(prompt)   # Haiku, GPT-3.5
```
Best when: Weak model makes systematic errors you want to eliminate.

**Strategy 2: With vs. Without Domain Reference**
```
chosen  = model.generate(prompt, context=domain_reference)
rejected = model.generate(prompt, context=None)
```
Best when: Domain knowledge significantly improves response quality.

**Strategy 3: Temperature Variation**
```
chosen  = model.generate(prompt, temperature=0.3)  # Focused
rejected = model.generate(prompt, temperature=1.2) # More random
```
Best when: You want to reduce variance and rambling.

**Strategy 4: Correct vs. Subtly Flawed**
```
chosen  = generate_response(prompt)
rejected = introduce_flaw(chosen, flaw_type)

flaw_types = [
    "slightly_off_topic",     # Addresses adjacent issue
    "missing_acknowledgment", # Jumps to advice
    "too_verbose",            # Adds unnecessary content
    "wrong_tone",             # Formal when should be warm
    "prescriptive",           # "You should" instead of "You might"
]
```
Best when: You want to teach specific quality distinctions.

**Strategy 5: Rubric-Based Selection**
```
# Generate multiple responses, score all, pick best and worst
responses = [model.generate(prompt, temp=0.8) for _ in range(5)]
scores = [evaluate(prompt, r) for r in responses]
chosen = responses[argmax(scores)]
rejected = responses[argmin(scores)]
```
Best when: You have a good evaluation rubric and want data-driven pairs.

### DPO Quality Requirements

- **Chosen must be clearly better** — Passes evaluation rubric
- **Rejected must be plausibly wrong** — Not garbage, just worse
- **Margin matters** — Score difference should be meaningful (0.15+ delta)
- **Same prompt** — Both responses to identical input

**Typical volume**: 1K-3K pairs. Preference learning is sample-efficient.

---

## KTO: Kahneman-Tversky Optimization

**When to use**: You have binary labels (good/bad) but can't construct preference pairs. Simpler than DPO, sometimes comparable results.

**Data format**:
```json
{
  "prompt": "User input here",
  "completion": "Response here",
  "label": true  // or false
}
```

**Generation approach**:
1. Generate responses using any method
2. Evaluate with rubric
3. Label: `true` if passes, `false` if fails
4. Include both positive and negative examples

**Balance**: Aim for 60-70% positive examples. Too many negatives hurts learning.

**Typical volume**: 2K-5K examples (more than DPO since signal is weaker per example).

---

## GRPO: Group Relative Policy Optimization

**When to use**: You want online learning where the model generates its own responses during training and learns from reward signals.

**Data format**:
```json
{"prompt": "User input here"}
```

Responses are generated during training by the policy model, then scored by a reward function.

**Requirements**:
- Reward function that can score any response (your evaluation rubric)
- Training infrastructure that supports online generation
- More compute than offline methods

**Reward function design**:
```python
def reward(prompt: str, response: str) -> float:
    """
    Score a response. Higher = better.
    Must be fast enough to run during training.
    """
    # Option 1: Use your evaluation rubric
    answers, _ = evaluate_response(prompt, response)
    score = compute_score(answers)
    return score

    # Option 2: Simpler heuristics for speed
    # (if full rubric is too slow)
```

**Trade-offs**:
- Pro: Model learns from its own distribution
- Pro: Can adapt during training
- Con: More complex setup
- Con: Higher compute cost
- Con: Reward hacking risk if reward function is imperfect

**Typical volume**: 5K-20K prompts. More data helps since responses vary.

---

## Combining Methods

**Common patterns**:

**SFT → DPO** (most common)
1. SFT to teach basic capability
2. DPO to refine quality

**SFT → GRPO**
1. SFT for initial capability
2. GRPO for online refinement with reward

**Why not start with DPO/GRPO?**
- Model needs basic capability first
- Preference methods refine, they don't teach from scratch

---

## Method Comparison

| Aspect | SFT | DPO | KTO | GRPO |
|--------|-----|-----|-----|------|
| **Data needed** | (prompt, response) | (prompt, chosen, rejected) | (prompt, response, label) | (prompt) + reward |
| **Data volume** | 1K-5K | 1K-3K | 2K-5K | 5K-20K |
| **Complexity** | Low | Medium | Low | High |
| **Best for** | New capabilities | Quality refinement | Binary feedback | Online learning |
| **Compute** | Low | Low | Low | High |
| **Risk** | Overfitting | Pair quality | Label noise | Reward hacking |

---

## Data Quality Checklist

Before training with any method:

- [ ] Evaluation rubric validated on sample data
- [ ] Input taxonomy covers production distribution
- [ ] Pass rate stable (not still iterating on prompts)
- [ ] Format matches trainer expectations (validate with chat template)
- [ ] Held-out eval set created (10% split)
- [ ] For DPO: Verified margin between chosen/rejected
- [ ] For GRPO: Reward function tested on edge cases
