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

---

## Multi-Turn Conversation Generation

For therapeutic coaching and similar domains, you need **coherent multi-turn conversations**, not single exchanges. This is significantly harder than single-turn generation.

### The Challenge

Multi-turn generation requires:
1. A coherent "user persona" that maintains consistent context
2. Natural conversation flow (not just Q&A ping-pong)
3. Topic evolution (opening → exploration → depth → resolution)
4. Realistic user behaviors (resistance, tangents, breakthroughs)

### Approach: Two-Agent Simulation

Use two LLM instances to simulate the conversation:

```python
import dspy

class UserSimulator(dspy.Signature):
    """Simulate a therapy client continuing a conversation."""
    persona: str = dspy.InputField(desc="User's background, situation, communication style")
    conversation_so_far: str = dspy.InputField()
    turn_guidance: str = dspy.InputField(desc="What should happen this turn")
    user_message: str = dspy.OutputField()

class TherapistResponder(dspy.Signature):
    """Generate therapeutic coach response."""
    system_prompt: str = dspy.InputField()
    conversation_so_far: str = dspy.InputField()
    assistant_response: str = dspy.OutputField()
```

### Persona Generation

Create diverse user personas from your taxonomy:

```python
class GeneratePersona(dspy.Signature):
    """Create a realistic therapy client persona."""
    topic: str = dspy.InputField()
    subtopic: str = dspy.InputField()
    style: str = dspy.InputField()
    difficulty: str = dspy.InputField()

    persona: str = dspy.OutputField(desc="2-3 sentences: situation, emotional state, communication style")
    opening_message: str = dspy.OutputField(desc="How they'd start the conversation")
```

**Example persona:**
> "35-year-old marketing manager, recently passed over for promotion. Feeling a mix of anger and self-doubt. Tends to intellectualize emotions, uses analytical language. Currently questioning whether to stay at the company or job search."

### Turn-by-Turn Guidance

Don't let the conversation meander. Guide each turn's purpose:

```python
import random

TURN_TEMPLATES = {
    "early": [
        "Share more context about the situation",
        "Express a specific emotion more directly",
        "Ask the assistant a direct question",
        "Show slight resistance to a suggestion",
    ],
    "middle": [
        "Go deeper into underlying feelings",
        "Make a connection to past experience",
        "Express ambivalence about change",
        "Have a small insight or realization",
    ],
    "late": [
        "Reflect on what's been discussed",
        "Express what feels different now",
        "Identify a small concrete next step",
        "Thank the assistant naturally",
    ],
}

def get_turn_guidance(turn_number: int, total_turns: int) -> str:
    if turn_number <= total_turns * 0.3:
        phase = "early"
    elif turn_number <= total_turns * 0.7:
        phase = "middle"
    else:
        phase = "late"
    return random.choice(TURN_TEMPLATES[phase])
```

### Full Generation Loop

```python
async def generate_conversation(
    persona: str,
    opening: str,
    target_turns: int,
    system_prompt: str,
) -> list[tuple[str, str]]:
    """Generate a complete multi-turn conversation."""
    conversation: list[tuple[str, str]] = []
    history = ""

    # First turn
    user_msg = opening
    assistant_msg = await generate_therapist_response(system_prompt, history, user_msg)
    conversation.append((user_msg, assistant_msg))
    history = format_history(conversation)

    # Subsequent turns
    for turn in range(2, target_turns + 1):
        guidance = get_turn_guidance(turn, target_turns)

        user_msg = await generate_user_message(persona, history, guidance)
        assistant_msg = await generate_therapist_response(system_prompt, history, user_msg)

        conversation.append((user_msg, assistant_msg))
        history = format_history(conversation)

    return conversation
```

### Quality Controls

**Coherence checks:**
- User persona stays consistent (no sudden personality shifts)
- Topics connect naturally (no random jumps unless guided)
- Assistant references earlier context appropriately

**Diversity controls:**
- Vary conversation lengths (15-30 turns as specified)
- Mix topic progressions (linear, tangential, returning)
- Include different resolution types (insight, action, continued exploration)

### DSPy Integration

For automated optimization of conversation generation:

```python
class ConversationGenerator(dspy.Module):
    def __init__(self):
        self.persona_gen = dspy.ChainOfThought(GeneratePersona)
        self.user_sim = dspy.ChainOfThought(UserSimulator)
        self.therapist = dspy.ChainOfThought(TherapistResponder)

    def forward(self, topic, subtopic, style, difficulty, target_turns):
        # Generate persona
        persona_result = self.persona_gen(
            topic=topic, subtopic=subtopic, style=style, difficulty=difficulty
        )

        # Generate conversation
        conversation = []
        history = ""

        # ... generation loop using self.user_sim and self.therapist

        return dspy.Prediction(conversation=conversation)

def conversation_metric(example, pred, trace=None):
    """Evaluate full conversation with rubric."""
    from assessor import assess_conversation, ConversationInput

    conv_input = ConversationInput.from_tuples(pred.conversation)
    result = asyncio.run(assess_conversation(conv_input))

    feedback = "\n".join(
        f"{cid}: {result.reasonings[cid]}"
        for cid in result.failed_checks
    )

    return {
        "score": result.score if not result.safety_gate_failed else 0.0,
        "feedback": feedback or "All criteria passed",
    }
```

### Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Repetitive user messages | No turn guidance | Add explicit turn templates |
| User suddenly "cured" | No resistance modeling | Include ambivalence in personas |
| Shallow conversations | Rushing to resolution | Extend middle phase, add depth prompts |
| Incoherent context | No history in prompts | Always include full conversation history |
| Same structure every time | Deterministic generation | Vary temperatures, randomize guidance |

---

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
