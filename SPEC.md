# Therapeutic Coaching Fine-tuning Project Specification

## Project Goal

Build a **privacy-first, locally-runnable therapeutic coaching model** that:
- Runs offline on consumer hardware (Mac, Ollama, GGUF)
- Applies eclectic therapeutic approaches adaptively
- Feels genuinely helpful for self-care
- Is open-source for community benefit

**Primary motivations:**
1. Privacy — personal conversations stay local
2. Offline capability — no internet required
3. Cost savings — no ongoing API costs
4. Customization — behaviors prompting can't achieve

**Success criteria:**
1. You actually use it for self-care
2. Measurable improvement on evaluation rubric (p < 0.05)
3. Conversations feel genuinely helpful, not robotic

---

## Target Model

**Size:** 7B parameters (portable, runs on consumer hardware)

**Deployment targets:**
- Apple Silicon Macs (MLX, llama.cpp)
- Ollama
- GGUF quantized formats
- Any local inference setup

**Note:** The pipeline supports fine-tuning multiple model sizes for different hardware. 7B is the initial target.

---

## Therapeutic Approach

### Philosophy

- **Adaptive tone** — Matches user's energy (casual when casual, serious when serious)
- **Eclectic/integrative** — Uses whichever approach fits the situation
- **Coaching, not therapy** — Supportive self-care tool for mature adults
- **Safety stance: Acknowledge and hold space** — Validates severity, doesn't overstep, gently suggests resources when appropriate

### Therapeutic Frameworks (9 Styles)

| Framework | Focus |
|-----------|-------|
| **CBT** | Cognitive distortions, thought reframing |
| **DBT** | Distress tolerance, emotional regulation |
| **ACT** | Acceptance, values-based action |
| **Motivational Interviewing** | Exploring ambivalence, change talk |
| **Solution-Focused (SFBT)** | Future-oriented, "what's working" |
| **Person-Centered/Rogerian** | Unconditional positive regard, reflection |
| **Positive Psychology** | Strengths, gratitude, meaning |
| **Compassion-Focused (CFT)** | Self-criticism, developing self-compassion |
| **Behavioral Activation** | Activity scheduling, breaking avoidance cycles |

### Target Audience

- Mature adults doing self-care
- Not in active crisis
- Not a replacement for clinical therapy
- Self-selected users who want a private coaching tool

---

## 5-SKILL Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  SKILL 1: Domain Knowledge Extractor                                 │
│  ─────────────────────────────────────                              │
│  • therapeutic-frameworks.md (expand to 9 approaches)               │
│  • input-taxonomy.yaml (topics, styles, edge cases)                 │
│  • evaluation-rubric.md (update for long conversations)             │
│                                                                      │
│  Outputs: Domain reference, taxonomy, quality criteria              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SKILL 2: Synthetic Data Generator                                   │
│  ─────────────────────────────────                                  │
│  • Generate multi-turn conversations from taxonomy                  │
│  • Use DSPy/GEPA for automated prompt optimization                  │
│  • Evaluate with rubric, filter at 0.80 threshold                   │
│  • Target: 1-2K filtered conversations                              │
│                                                                      │
│  Outputs: training_data.jsonl, eval_holdout.jsonl                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SKILL 3: Base Model Selector                                        │
│  ────────────────────────────                                       │
│  • Research current SOTA for target size                            │
│  • Consider: license, quality, context, community, quantization     │
│  • Discuss trade-offs with user                                     │
│  • Pilot evaluation on sample inputs                                │
│                                                                      │
│  Outputs: Selected model + rationale                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SKILL 4: Training (HuggingFace)                                     │
│  ───────────────────────────────                                    │
│  • QLoRA fine-tuning via HuggingFace infrastructure                 │
│  • SFT approach (revisit DPO if needed later)                       │
│  • Merge adapter weights                                            │
│  • Export to GGUF for local inference                               │
│                                                                      │
│  Outputs: Fine-tuned model, GGUF export                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SKILL 5: Model Evaluator                                            │
│  ────────────────────────                                           │
│  • Run baseline vs fine-tuned on held-out set                       │
│  • Statistical comparison (t-test)                                  │
│  • Qualitative spot-check                                           │
│  • Decision: deploy / iterate / try DPO                             │
│                                                                      │
│  Outputs: Comparison report, deployment decision                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Data Specification

### Conversation Length

| Type | Percentage | Turns |
|------|------------|-------|
| Medium | 50% | ≤15 turns |
| Extended | 50% | ≤30 turns |

**Definition:** A "turn" = one user message + one assistant response (a pair).
- 15 turns = 30 messages total (15 user + 15 assistant)
- 30 turns = 60 messages total (30 user + 30 assistant)

**No short/single-turn conversations** — realistic coaching sessions are the goal.

### Data Volume

- Generate: ~3-4K raw conversations
- Filter at 0.80 threshold: ~1.5-2K conversations
- **Holdout split: 10%** (stratified by topic/difficulty)
  - Training: 90% → ~1.35-1.8K conversations
  - Evaluation: 10% → ~150-200 conversations
- Turn count: 22K-60K turns of training signal
  - Lower bound: 1.5K convos × 15 turns = 22.5K turns
  - Upper bound: 2K convos × 30 turns = 60K turns
- Token estimate: ~3-8M tokens total
  - ~200 tokens per message average × 2 messages per turn × 22-60K turns

### Pilot Calibration (Before Scaling)

**Run a pilot of 100 conversations before scaling to full volume.**

The pilot serves to:
1. Calibrate pass rate expectations (50% assumed, may be lower)
2. Identify systematic failures in generation prompts
3. Tune rubric thresholds if needed
4. Estimate actual generation costs

**Decision criteria:**
- Pass rate ≥40%: Proceed to scale, iterate prompts if below 50%
- Pass rate 25-40%: Major prompt revision needed before scaling
- Pass rate <25%: Fundamental issue — revisit taxonomy or rubric

### Data Format (SFT)

```json
{"messages": [
  {"role": "system", "content": "You are a supportive therapeutic coach..."},
  {"role": "user", "content": "I've been feeling really overwhelmed..."},
  {"role": "assistant", "content": "That sounds exhausting..."},
  {"role": "user", "content": "Yeah, my boss keeps piling on..."},
  {"role": "assistant", "content": "The pressure to keep up..."},
  ...
]}
```

### Input Taxonomy

```yaml
taxonomy:
  topics:
    - name: anxiety
      weight: 0.20
      subtopics: [work_stress, social, health, general_worry, panic]
    - name: relationships
      weight: 0.20
      subtopics: [romantic, family, friendship, coworker, loneliness]
    - name: life_transitions
      weight: 0.15
      subtopics: [career_change, relocation, loss_grief, new_role, decision]
    - name: self_worth
      weight: 0.15
      subtopics: [confidence, imposter, self_criticism, perfectionism, identity]
    - name: emotional_regulation
      weight: 0.15
      subtopics: [anger, sadness, overwhelm, numbness, mood_swings]
    - name: edge_cases
      weight: 0.15
      subtopics:
        - crisis_signals    # Suicidal ideation, self-harm mentions
        - medical_advice    # Requests for diagnoses, medication info
        - out_of_scope      # Legal, financial, non-therapeutic
        - vague             # Minimal context, unclear intent
        - hostile           # Aggressive, testing boundaries

  styles:
    terse: 0.15            # "feeling anxious"
    conversational: 0.40   # Natural, flowing
    detailed: 0.25         # Full context provided
    emotional: 0.15        # Intense feelings expressed
    analytical: 0.05       # "I notice a pattern..."

  difficulty:
    easy: 0.30             # Clear emotion, common situation
    medium: 0.50           # Mixed feelings, some complexity
    hard: 0.20             # Ambiguous, layered, edge cases
```

---

## Assessment Rubric

### Structure (12 Criteria)

All criteria assess the **full conversation**, not individual turns. This reduces API costs by 98% (12 calls vs 18N + 6).

**Quality Criteria (CQ1-CQ9):**
| Category | ID | Criterion |
|----------|-----|-----------|
| Comprehension | CQ1, CQ2 | Understanding, ambiguity handling |
| Connection | CQ3, CQ4 | Emotional attunement, pacing |
| Usefulness | CQ5, CQ6 | Added value, empowerment |
| Fit | CQ7 | Calibrated responses |
| Safety | CQ8, CQ9* | Harmful patterns, crisis handling |

**Pattern Criteria (CP1-CP3):**
| ID | Criterion | Condition |
|----|-----------|-----------|
| CP1 | Variety in techniques | ≥3 turns |
| CP2 | Natural and warm | Always |
| CP3* | Arc + coherence + depth | ≥10 turns |

*\* = conditional (can return NA)*

### Scoring

```python
weights = {
    "comprehension": 0.15,
    "connection": 0.20,  # Highest - therapy is relational
    "usefulness": 0.15,
    "fit": 0.10,
    "safety": 0.20,      # GATE: any failure = auto-reject
    "patterns": 0.20,
}
pass_threshold = 0.80
safety_gate = True  # CQ8 or CQ9 failure = automatic rejection
```

**Implementation:** See `assessor.py` and `reference/assessment-rubric.md`

---

## Prompt Optimization

### Approach: DSPy/GEPA

Instead of manual prompt iteration, use automated optimization:

```python
import dspy

def rubric_metric(example, pred, trace=None):
    """Returns score + textual feedback for GEPA."""
    answers, reasonings = evaluate(example.input, pred.response)
    result = score(answers)

    feedback_parts = []
    for criterion_id in result.get("failed_checks", []):
        feedback_parts.append(f"{criterion_id}: {reasonings[criterion_id]}")

    return {
        "score": result["score"],
        "feedback": "\n".join(feedback_parts) or "All criteria passed",
    }

optimizer = dspy.GEPA(
    metric=rubric_metric,
    reflection_lm=dspy.LM("claude-sonnet-4-20250514"),
    auto="light",  # Fast for weekend timeline
)

optimized = optimizer.compile(
    GenerateConversation(),
    trainset=pilot_inputs,
    valset=validation_inputs,
)
```

**Why GEPA:**
- Uses textual feedback from rubric (not just scores)
- Pareto frontier maintains diverse solutions
- 35x more sample-efficient than alternatives

---

## Base Model Candidates

| Model | Size | Context | License | Notes |
|-------|------|---------|---------|-------|
| Qwen 2.5 7B Instruct | 7B | 128K | Apache 2.0 | Strong chat, good multilingual |
| Llama 3.1 8B Instruct | 8B | 128K | Llama 3.1 | Large community, proven |
| Mistral 7B Instruct v0.3 | 7B | 32K | Apache 2.0 | Efficient, shorter context |

**Selection criteria:**
1. Context length (32K sufficient for 30-turn conversations; ~14K tokens typical)
2. Existing chat quality
3. License (open-source friendly)
4. GGUF/quantization support
5. Community tooling

**Context calculation:**
- 30 turns × 2 messages × ~200 tokens = ~12K tokens
- + ~2K system prompt = ~14K tokens
- 32K context provides comfortable 2x margin

**Decision:** Quick research + pilot evaluation on 50 examples.

---

## Weekend Execution Plan

### Day 1 (Saturday)

| Time | Task |
|------|------|
| Morning | Expand therapeutic-frameworks.md (9 styles) |
| Morning | Create input-taxonomy.yaml |
| Midday | Quick base model research, select one |
| Midday | Generate 100 pilot conversations |
| Afternoon | Run GEPA optimization (automated) |
| Evening | Scale to 1.5-2K with optimized prompts |

### Day 2 (Sunday)

| Time | Task |
|------|------|
| Morning | Final data filtering and validation |
| Morning | Submit HuggingFace training job |
| Midday | (Training runs ~2-4 hours) |
| Afternoon | Run evaluation comparison |
| Afternoon | Export GGUF, test locally |
| Evening | Document SKILLs, wrap up |

---

## Deliverables

### Code/Data
- `config/input-taxonomy.yaml` — Therapeutic input distribution
- `config/generation-prompt.md` — Optimized generation prompt
- `config/system-prompt.md` — System prompt for training data
- `output/training_data.jsonl` — Filtered training conversations
- `output/eval_holdout.jsonl` — Held-out evaluation set
- `output/generation_report.json` — Pass rates, iteration history
- `output/failed_examples.jsonl` — For debugging generation prompts
- Fine-tuned model on HuggingFace Hub
- GGUF export for local inference

### Documentation
- Updated `reference/therapeutic-frameworks.md` (9 styles)
- Updated `reference/assessment-rubric.md` (conversation-level, safety gate)
- SKILL documentation for all 5 skills
- `reports/comparison_report.md` — Final evaluation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Personal use | Would you open it when stressed? |
| Rubric improvement | +10-20% over base model |
| Statistical significance | p < 0.05 |
| Qualitative feel | Genuinely helpful, not robotic |
| Safety | Handles boundaries thoughtfully |

---

## Open Questions (To Resolve During Implementation)

1. Final base model selection (after pilot)
2. Optimal conversation length distribution
3. Whether to pursue DPO after SFT

---

## What's Already Done

| Artifact | Status |
|----------|--------|
| Assessment rubric (12 criteria) | ✅ Complete (with safety gate) |
| `assessor.py` | ✅ Complete (Pydantic validation, safety gate, file input) |
| `therapeutic-frameworks.md` (9 styles) | ✅ Complete |
| Data generation skill | ✅ Complete (incl. multi-turn generation spec) |
| Training methods guide | ✅ Complete |
| Evaluation integration guide | ✅ Complete |
| Input taxonomy (SPEC + template) | ✅ Complete |
| System prompt for training | ✅ Complete |

**Not yet implemented:**
- `config/input-taxonomy.yaml` — Actual config file (template exists)
- `config/generation-prompt.md` — Optimized after DSPy iteration
- Conversation generator script — Code from SKILL spec
- DSPy dependency + optimization pipeline
- Unit tests for assessor scoring logic (see TODO below)

---

## TODO: Testing

**Use TDD for critical scoring logic.** The assessor's scoring logic is complex (weighted categories, safety gate, conditional criteria) and bugs here silently corrupt training data.

**Required tests:**
1. `compute_score()` — weighted category scoring
2. Safety gate behavior (any safety failure = auto-reject)
3. NA handling (should count as pass, not affect score)
4. ERROR handling (should count as failure)
5. Minimum turn enforcement
6. Edge cases: all-NA category, all-ERROR, mixed results

**Test approach:**
- Mock API responses to test scoring in isolation
- Use known-good/known-bad conversations for integration tests
- Property-based testing for edge case coverage

**Run before any training job:**
```bash
uv run pytest tests/test_assessor.py -v
```

---

*Last updated: December 2024*
*Timeline: Weekend project*
