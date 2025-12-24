---
name: generating-finetuning-data
description: Use when creating synthetic training data for LLM fine-tuning. Covers SFT, DPO, GRPO, and reinforcement approaches. Requires evaluation rubric and input taxonomy.
---

# Generating Fine-tuning Data

## Core Principle

**Generate → Evaluate → Analyze → Improve → Repeat.**

This is an iterative loop, not a linear pipeline. Expect 3-5 iterations before generation prompts produce acceptable pass rates.

## The Loop

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Generate │───▶│ Evaluate │───▶│ Analyze  │      │
│  └──────────┘    └──────────┘    └────┬─────┘      │
│       ▲                               │            │
│       │         ┌──────────┐          │            │
│       └─────────│ Improve  │◀─────────┘            │
│                 │ Prompts  │                       │
│                 └──────────┘                       │
│                                                    │
│  Exit when: pass rate stabilizes at target          │
└─────────────────────────────────────────────────────┘
```

**Pass rate expectations by rubric stringency:**
- Lenient rubric: 70-90% (may be too easy)
- Moderate rubric: 50-70% (typical target)
- Strict rubric: 30-50% (high quality, lower volume)

## Prerequisites

| Artifact | Purpose |
|----------|---------|
| **Evaluation rubric** | Quality criteria with binary (YES/NO/NA) questions |
| **Input taxonomy** | Distribution of inputs to generate (see [input-taxonomy.md](input-taxonomy.md)) |
| **Domain reference** | Ground truth knowledge for the domain (optional) |

## Training Method Selection

| Method | When to Use | Data Format |
|--------|-------------|-------------|
| **SFT** | Teaching new behaviors, style, format | `(prompt, response)` |
| **DPO** | Preference alignment, subtle quality improvements | `(prompt, chosen, rejected)` |
| **GRPO** | Online learning, reward optimization | `(prompt)` + reward function |
| **KTO** | Binary feedback, simpler than DPO | `(prompt, response, label)` |

See [training-methods.md](training-methods.md) for detailed guidance on each method.

## Phase 1: Generate Diverse Inputs

**Goal**: Cover the input distribution the model will face in production.

Systematically vary across your taxonomy:
- **Topics/scenarios** — What the user asks about
- **Communication styles** — Terse, verbose, emotional, analytical
- **Difficulty levels** — 30% easy, 50% medium, 20% hard
- **Edge cases** — 10-15% of total (including out-of-scope)

**Diversity check**: After generation, compute pairwise similarity. Flag if >5% of inputs have similarity >0.8.

## Phase 2: Generate Responses

**For SFT**: Generate one high-quality response per input using a strong model.

**For DPO/Preference**: Generate paired responses. See [training-methods.md](training-methods.md) for strategies:
- Strong model vs. weak model
- High temperature vs. low temperature
- With vs. without domain reference
- Correct vs. subtly flawed

**For GRPO**: Skip this phase — responses generated during training by the policy model.

## Phase 3: Evaluate

Run every generated example through your evaluation rubric.

**Rubric requirements**:
- Binary questions (YES/NO/NA) for reliability
- Safety gate (any safety failure = automatic discard)
- Category scores for diagnostics
- Overall pass/fail with configurable threshold

**Threshold selection**:
- 0.70 — Minimum viable
- 0.80 — Recommended for training data
- 0.85+ — Premium quality, lower volume

## Phase 4: Analyze Failures

**This is where most people skip ahead. Don't.**

When pass rate is below target, diagnose before regenerating:

| Symptom | Likely Cause | Investigation |
|---------|--------------|---------------|
| Many failures in one category | Generation prompt missing that aspect | Review rubric criteria for that category |
| Failures across all categories | Fundamental prompt issue | Compare failed vs. passed examples |
| High variance in scores | Inconsistent generation | Check temperature, add constraints |
| Safety failures | Missing guardrails | Add explicit safety instructions |

**Diagnostic questions**:
1. Which criteria fail most often?
2. What do failed examples have in common?
3. What do passed examples do differently?
4. Is the rubric too strict for this domain?

## Phase 5: Improve Generation Prompts

Two approaches: **manual iteration** or **automated optimization** with DSPy.

### Option A: Manual Iteration

Based on failure analysis, revise prompts:

| Failure Pattern | Prompt Fix |
|-----------------|------------|
| Missing context acknowledgment | Add: "First acknowledge what the user said before responding..." |
| Too verbose/too terse | Add length guidance: "Respond in 2-3 paragraphs..." |
| Wrong tone | Add: "Match the tone to the user's message..." |
| Missing structure | Add template: "Structure your response as: 1) ... 2) ... 3) ..." |
| Boundary violations | Add constraints specific to your domain's safety requirements |

**Then return to Phase 2.** Regenerate a sample (100-200 examples), evaluate, check if pass rate improved.

### Option B: Automated Optimization with DSPy

Use [DSPy](https://dspy.ai) to automatically optimize generation prompts using your rubric as the objective function.

**Recommended optimizer: GEPA** (Genetic-Pareto) — uses textual feedback from your rubric, not just scores.

| Optimizer | Best For | Signal Used |
|-----------|----------|-------------|
| **GEPA** | Rich rubrics with per-criterion feedback | Score + textual reasoning |
| MIPROv2 | Simple pass/fail metrics, few-shot optimization | Score only |

**Why GEPA for evaluation rubrics:**
- Your rubric returns *why* each criterion failed — GEPA exploits this
- Pareto frontier maintains diverse solutions (one per failure mode)
- 35x more sample-efficient than alternatives

```python
import dspy

class GenerateResponse(dspy.Signature):
    """Generate a response following domain guidelines."""
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

def rubric_metric(example, pred, trace=None):
    """Metric returning score + textual feedback for GEPA."""
    answers, reasonings = evaluate(example.user_input, pred.response)
    result = score(answers)

    # Build feedback from failed criteria
    feedback_parts = []
    for criterion_id in result.get("failed_checks", []):
        feedback_parts.append(f"{criterion_id}: {reasonings[criterion_id]}")

    return {
        "score": result["score"],
        "feedback": "\n".join(feedback_parts) or "All criteria passed",
    }

# Optimize
optimizer = dspy.GEPA(
    metric=rubric_metric,
    reflection_lm=dspy.LM("claude-sonnet-4-20250514", temperature=1.0),
    auto="medium",  # or "light" for quick iteration
)

optimized = optimizer.compile(
    GenerateResponse(),
    trainset=sample_inputs,  # 50-200 examples
    valset=validation_inputs,
)

# Use optimized program for generation at scale
for input_text in all_inputs:
    response = optimized(user_input=input_text).response
```

**When to use DSPy:**
- Rubric has 5+ criteria with textual reasoning
- Manual iteration isn't converging
- You have 50+ labeled examples for optimization

**When to skip DSPy:**
- Simple rubrics (2-3 criteria)
- Already achieving 60%+ pass rate manually
- Limited compute budget for optimization

## Phase 6: Scale and Format

Once pass rate is stable at target:

1. **Generate at scale** — 2-3x your target volume
2. **Filter** — Keep only examples above threshold
3. **Format** — Convert to training method format
4. **Split** — 90% train, 10% held-out eval
5. **Validate** — Apply chat template to verify compatibility

## Output Formats

**SFT** (messages format):
```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**DPO** (preference pairs):
```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

**KTO** (binary labels):
```json
{
  "prompt": "...",
  "completion": "...",
  "label": true
}
```

**GRPO** (prompts only — responses generated during training):
```json
{"prompt": "..."}
```

## Cost Estimation

| Stage | Relative Cost | Notes |
|-------|---------------|-------|
| Input generation | Low | Small outputs |
| Response generation | Medium-High | Main driver for SFT/DPO |
| Evaluation | Medium | N API calls per example (batch pricing helps) |
| Iteration overhead | 2-3x base | Expect 3-5 prompt revision cycles |

**Use batch API** for generation and evaluation — 50% cost reduction, latency irrelevant.

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Skipping failure analysis | Thrashing on prompt changes | Diagnose before changing |
| Threshold too low | Training on mediocre data | Use 0.80+ for training data |
| No diversity check | Model overfits to narrow patterns | Validate input diversity |
| Ignoring edge cases | Model fails on boundaries | 10-15% edge cases in taxonomy |
| Linear thinking | Frustration when first pass fails | Expect iteration |

## Outputs

```
output/
├── training_data.jsonl    # Filtered, formatted for training
├── eval_holdout.jsonl     # 10% held out
├── generation_report.json # Pass rates, iteration history, costs
├── failed_examples.jsonl  # For ongoing prompt debugging
└── rubric_analysis.json   # Which criteria failed most
```
