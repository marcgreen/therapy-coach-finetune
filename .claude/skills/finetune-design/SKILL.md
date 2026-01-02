---
name: finetune-design
description: Use when preparing to fine-tune an LLM for multi-turn conversations. Covers model selection, rubric design, persona creation, and prompt engineering. Complete this phase before generating any training data.
---

# Fine-tune Design

Design all artifacts needed before generating training data for multi-turn conversation fine-tuning.

## Inputs

- Domain to fine-tune for (customer support, coaching, tutoring, etc.)
- Deployment constraints (hardware, offline requirement, budget)
- Access to domain expertise (or ability to research it)

## Outputs

By the end of this phase, you will have:

- [ ] `model-choice.md` — Selected model with documented tradeoffs
- [ ] `config/input-taxonomy.yaml` — Topics, styles, difficulty, edge cases
- [ ] `config/rubric.yaml` — Binary criteria with calibration examples
- [ ] `config/persona-template.yaml` — Diversity dimensions and distributions
- [ ] `config/prompts/user_sim.md` — User simulator prompt
- [ ] `config/prompts/assistant.md` — Assistant generation prompt
- [ ] `config/system-prompt.md` — System prompt for training data
- [ ] `base-model-eval-results.md` — Baseline evaluation results

---

## Workflow

### Step 1: Base Model Selection

Select the model you'll fine-tune based on:

| Factor | Why It Matters |
|--------|----------------|
| Context window | Max conversation length you can train on |
| Quantization support | GGUF, MLX, QAT for local deployment |
| Base capability | Evaluate before committing |
| Training cost | LoRA/QLoRA vs full fine-tune |
| Deployment target | Ollama, llama.cpp, MLX |

**Gate:** Model chosen with documented tradeoffs in `model-choice.md`

**Reference:** [model-selection-guide.md](model-selection-guide.md)

---

### Step 2: Token Economics

Determine training constraints based on cost:

| Tokens/Example | Cost Impact |
|----------------|-------------|
| <8K | Cheapest, short conversations only |
| 8-16K | Cost-effective, moderate conversations |
| 16-32K | Expensive, long conversations |
| >32K | Very expensive, may require special handling |

**Constraint:** Plan max conversation length based on your budget. 16K is a practical ceiling for most projects.

**Gate:** Max transcript token length defined

**Reference:** [model-selection-guide.md#token-economics](model-selection-guide.md#token-economics)

---

### Step 3: Input Taxonomy

Define the distribution of inputs to generate. A good taxonomy has multiple dimensions:

| Dimension | Question | Examples |
|-----------|----------|----------|
| WHAT | What are they asking about? | Topics, subtopics |
| HOW | How do they communicate? | Style, verbosity, tone |
| WHO | Who are they? | Demographics, context |
| DIFFICULTY | How hard is this to handle? | Easy, medium, hard |
| EDGE CASES | What should trigger special handling? | Boundaries, safety |

**Key lesson:** Allocate ~15% to edge cases. Without explicit representation, the model won't learn to handle them.

**Gate:** Weighted taxonomy with cross-product dimensions in `config/input-taxonomy.yaml`

**Reference:** [taxonomy-guide.md](taxonomy-guide.md)

---

### Step 4: Evaluation Rubric

Design quality criteria for assessing generated conversations.

**Critical requirements:**
- Binary judgments (YES/NO/NA) — not numeric scales
- Grouped into weighted categories
- Safety gates that auto-reject on failure
- 3-8 calibration examples per criterion

**Structure:**
```yaml
categories:
  comprehension:
    weight: 0.15
    criteria: [CQ1, CQ2]
  # ... more categories

criteria:
  CQ1:
    name: "Accurate understanding"
    question: "Does the response demonstrate accurate understanding?"
    na_valid: false  # Must always be assessable
    calibration_examples:
      - type: PASS
        context: "..."
        response: "..."
        reasoning: "..."
      - type: FAIL
        # ...

safety_gates: [CQ8, CQ9]  # Any failure = auto-reject
pass_threshold: 0.80
```

**Gate:** Rubric with calibration examples in `config/rubric.yaml`

**Reference:** [rubric-guide.md](rubric-guide.md)

---

### Step 5: Persona Template

Design user diversity for realistic training data.

**Dimensions to define:**
- Communication style (terse, verbose, emotional, analytical)
- Behavior patterns / "flaws" (resistance, deflection, etc.)
- Trajectory (stable, improving, deteriorating)
- Domain-specific attributes

**Key lesson:** Flaws vary per message, not per conversation. Real people have good days and bad days.

```yaml
persona_template:
  communication_style:
    options: [terse, casual, formal, stream-of-consciousness]
    weights: [0.15, 0.50, 0.25, 0.10]

  flaw_patterns:
    primary: # 50% chance per message
    secondary: # 20% chance each per message

  # 20% of personas should have NO flaw patterns
```

**Gate:** Persona template with distributions in `config/persona-template.yaml`

**Reference:** [persona-guide.md](persona-guide.md)

---

### Step 6: Prompts

Create the three prompts for data generation:

| Prompt | Purpose |
|--------|---------|
| User simulator | Generate realistic user messages with flaws |
| Assistant | Generate high-quality responses |
| System prompt | What gets baked into training data |

**Key lessons for assistant prompt:**
- Length matching: Target 1.0-1.5x user word count, hard limit 2x
- Tentative language for interpretations ("I wonder if..." not "You are...")
- Question discipline: At most 1-2 questions per response
- Anti-patterns list: Specific phrases to avoid

**Gate:** All three prompts drafted

**Reference:** [generation-guide.md](../finetune-generate/generation-guide.md) (in finetune-generate)

---

### Step 7: Base Model Evaluation

Before committing to fine-tune, evaluate the base model on your rubric.

**Process:**
1. Generate 10-20 test scenarios covering your taxonomy
2. Have base model respond to each
3. Assess with your rubric
4. Calculate pass rate

**Decision gate:**

| Pass Rate | Recommendation |
|-----------|----------------|
| >70% | Base model may be sufficient. Consider prompt engineering first. |
| 50-70% | Fine-tuning likely helpful. Moderate improvement expected. |
| <50% | Fine-tuning needed. Significant improvement expected. |

**Gate:** Base model evaluated, decision to proceed documented in `base-model-eval-results.md`

---

## Done When

- [ ] All 8 output files created
- [ ] Base model evaluated against rubric
- [ ] Decision to proceed with fine-tuning documented
- [ ] Ready to start finetune-generate phase

---

## Resources

| Resource | What It Contains |
|----------|------------------|
| [code/SETUP-REFERENCE.md](../code/SETUP-REFERENCE.md) | Project structure and file templates |
| [code/infrastructure.py](../code/infrastructure.py) | Copy-paste ready: LLM backend, checkpointing, slicing, scoring |
| [examples/therapy-domain.md](../examples/therapy-domain.md) | Complete therapy domain example: taxonomy, flaws, rubric criteria |

---

## Next Phase

→ [finetune-generate](../finetune-generate/SKILL.md)
