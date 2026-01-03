# Generation Guide

How to generate multi-turn conversations using two-agent simulation.

---

## Architecture: Two-Agent Simulation

Single-turn generation produces unrealistic conversations. Real multi-turn conversations require:
- Context building across exchanges
- Topic evolution
- User resistance and breakthroughs
- History utilization

**Solution:** Two LLM agents in a simulation loop:

```python
async def generate_conversation(persona, target_turns, system_prompt):
    exchanges = []

    for turn in range(target_turns):
        # Agent 1: User Simulator
        user_msg = await user_simulator(
            persona=persona,
            history=exchanges,
            turn_guidance=get_turn_guidance(turn, target_turns)
        )

        # Agent 2: Assistant
        assistant_msg = await assistant_generator(
            system_prompt=system_prompt,
            history=exchanges,
            user_message=user_msg
        )

        exchanges.append({"user": user_msg, "assistant": assistant_msg})

    return exchanges
```

---

## Critical: Full History Passing

**Pass full conversation history to both agents.**

Early experiments truncated history for efficiency. This broke:
- Context continuity (assistant forgets what user said)
- Topic tracking (same topics re-introduced as new)
- Rubric criteria that depend on history utilization

```python
# WRONG: Truncated history
user_msg = await user_simulator(
    history=exchanges[-3:]  # Only last 3 exchanges
)

# CORRECT: Full history
user_msg = await user_simulator(
    history=exchanges  # Everything
)
```

**If context window is a concern:** Use a model with sufficient context, or design shorter conversations.

---

## User Simulator Design

The user simulator is as important as the assistant generator. Bad user simulation = bad training data.

### Non-Cooperative Responses

**The biggest lesson: Users don't just answer questions.**

```python
RESPONSE_TYPES = {
    "ignore_question_talk_about_own_thing": 0.30,
    "answer_tangentially_then_pivot": 0.30,
    "push_back": 0.20,
    "actually_engage": 0.20,
}
```

Without this, conversations feel like scripted Q&A, not real interactions.

### Turn Guidance

Without guidance, conversations drift aimlessly. Provide stage-appropriate direction.

**The pattern:** Define behaviors for early/middle/late conversation stages that create realistic progression.

```python
def get_turn_guidance(turn, total_turns):
    if turn < total_turns * 0.25:
        return random.choice(TURN_GUIDANCE["early"])
    elif turn < total_turns * 0.75:
        return random.choice(TURN_GUIDANCE["middle"])
    else:
        return random.choice(TURN_GUIDANCE["late"])
```

**Define guidance for your domain.** These are examples, not a comprehensive list. Discover what matters for your domain through brainstorming, expert roleplay, and iteration.

| Domain | Early | Middle | Late |
|--------|-------|--------|------|
| **Support/coaching** | Share context, express concern | Go deeper, show resistance | Reflect, identify next steps |
| **Technical help** | Describe problem, share errors | Try suggestions, report results | Confirm resolution, ask follow-ups |
| **Sales/consulting** | State needs, ask questions | Compare options, raise objections | Negotiate, decide |
| **Tutoring** | Attempt problem, show confusion | Work through steps, make mistakes | Demonstrate understanding |
| **???** | Your domain | Discovered through brainstorming, expert roleplay, and failed assessments |

### Behavioral Variation

> **Note:** These correspond to "flaws" in the persona template (see [persona-guide.md](../finetune-design/persona-guide.md)). Same concept, different framing: "flaws" for design, "behaviors" for implementation.

Inject realistic behavioral patterns probabilistically—not every message, not consistently.

**The pattern:** Define primary and secondary behaviors for each persona. Apply them with probability:

```python
def build_user_prompt(persona, history, turn_guidance):
    active_behaviors = []

    # Primary behavior: ~50% of messages
    if persona.primary_behavior and random.random() < 0.50:
        active_behaviors.append(persona.primary_behavior)

    # Secondary behaviors: ~20% each
    for behavior in persona.secondary_behaviors:
        if random.random() < 0.20:
            active_behaviors.append(behavior)

    return USER_PROMPT_TEMPLATE.format(
        persona=persona,
        history=format_history(history),
        turn_guidance=turn_guidance,
        active_behaviors=active_behaviors or "None - communicate directly",
    )
```

> **Note:** 50% and 20% are starting points that worked in one project. See [persona-guide.md](../finetune-design/persona-guide.md) for calibration guidance.

**Example behaviors by domain.** These are examples, not a comprehensive list. Discover what matters for your domain through brainstorming, expert roleplay, and iteration.

| Domain | Example Behaviors |
|--------|-------------------|
| **Support/coaching** | Vague descriptions, burying the real issue, deflecting, intellectualizing |
| **Technical help** | Incomplete error messages, wrong terminology, skipping steps, XY problem |
| **Customer service** | Frustration, impatience, referencing competitors, threatening to cancel |
| **Tutoring** | Guessing, pattern-matching without understanding, giving up too easily |
| **???** | Your domain's behaviors - discovered through brainstorming, expert roleplay, and failed assessments |

### Message Length Enforcement

Without hard limits, all communication styles converge to verbose. Define length ranges per style:

```python
# Example styles (define your own for your domain)
STYLE_LIMITS = {
    "terse": (30, 80),
    "casual": (80, 180),
    "detailed": (150, 300),
}

# In user simulator prompt:
"""
Your message MUST be between {min_words} and {max_words} words.
Count carefully. This is enforced.
"""
```

**Why this matters:** LLMs default to long messages. Explicit limits create realistic variety.

---

## Assistant Prompt Engineering

The assistant generator determines training data quality. Every prompt deficiency becomes a model deficiency.

**Assistant prompt requirements are highly domain-specific.** Identify what matters for YOUR domain.

### Identify Your Domain Type

| Domain Type | Key Constraints | Examples |
|-------------|-----------------|----------|
| **Conversational support** | Length matching, question discipline, tentative language, relationship continuity | Coaching, therapy, customer support |
| **Factual Q&A** | Accuracy, conciseness, source citation | Documentation bots, knowledge bases |
| **Creative/generative** | Style consistency, constraint following, originality | Writing assistants, code generation |
| **Task execution** | Completeness, correctness, efficiency | Agents, automation |

### Universal Constraints

These apply to most domains:

**Anti-pattern lists:** Explicitly list phrases to avoid. Review generated data and note recurring problematic phrases—hollow validation, sycophantic agreement, repetitive openers, domain jargon that sounds performative.

**ASCII only (if applicable):** Unicode characters cause issues in some pipelines. Specify straight quotes, standard dashes, no special characters if your training pipeline requires it.

### Domain-Specific Constraints

**These are examples, not a comprehensive list.** Your domain will have unique constraints. Discover them through iteration.

| Constraint | Applies To | What to Specify |
|------------|-----------|-----------------|
| **Length matching** | Conversational domains | Target ratio to user message (e.g., 1.0-1.5x) |
| **Question discipline** | Ongoing relationship domains | Limits per response, ending variety |
| **Tentative language** | Interpretation domains | Hedging requirements for inferences |
| **Proactive follow-up** | Coaching/support domains | Rules for referencing previous suggestions |
| **Format requirements** | Structured output domains | Headers, sections, code blocks |
| **Citation style** | Factual domains | How to reference sources |
| **???** | Your domain | Discovered through brainstorming, expert roleplay, and failed assessments |

**Example (conversational support):** See [therapy-domain.md](../examples/therapy-domain.md) for length matching ratios, question discipline rules, and tentative language requirements.

### Build Your Constraint List

1. **Generate 5-10 sample conversations** with minimal constraints
2. **Identify failure modes** — What feels wrong? Too long? Too robotic? Too aggressive?
3. **Add constraints iteratively** — One constraint per failure mode
4. **Test each constraint** — Does it fix the issue without breaking something else?

Don't copy constraints from other domains without validating they apply to yours.

---

## Model Selection for Generation

**Tier your models by task:**

| Role | Model | Rationale |
|------|-------|-----------|
| User Simulator | Cheaper (Haiku, Flash) | Generating messy human messages is easier |
| Assistant | Quality (Sonnet, Opus) | This is what you're training on |

```python
user_msg = await user_simulator(model="haiku")    # Cheap
assistant_msg = await assistant_generator(model="sonnet")  # Quality
```

**Savings:** ~50% reduction in generation costs.

**Validation:** Test that cheaper user simulation produces same pass rates before switching.

---

## Async vs Sync Text Format

If your domain involves asynchronous communication (e.g., email support, async messaging):

```markdown
ASYNC FORMAT:
- Each exchange represents a new session (not live chat)
- Time passes between exchanges
- Users report what happened: "so yesterday..."
- Natural to ask about previous suggestions
- Messages are naturally longer than live chat
```

This affects:
- How topics evolve (situations change between exchanges)
- When follow-up questions make sense
- Expected message length

---

## Checkpointing

Write progress after each exchange, not at end:

```python
for turn in range(target_turns):
    user_msg = await user_simulator(...)
    assistant_msg = await assistant_generator(...)
    exchanges.append({"user": user_msg, "assistant": assistant_msg})

    # Save immediately
    save_transcript(transcript_path, exchanges)
```

**Why:** If generation crashes at turn 45 of 50, you don't lose turns 1-44.

---

## Prompt Element Checklist

**User simulator prompt must include:**
- Persona details (communication style, behavioral patterns)
- Full conversation history
- Turn guidance for current stage
- Active behaviors for this message (probabilistic)
- Message length constraints

**Assistant prompt must include:**
- System prompt with domain knowledge
- Full conversation history
- Current user message
- Domain-specific quality requirements (see "Assistant Prompt Engineering" above)
- Anti-patterns list

**Expect iteration:** Assistant prompts typically go through 10-20 versions. Each iteration addresses specific failure modes discovered in assessment. This is normal.

---

## Validation Checklist

Before scaling generation:

- [ ] User simulator produces varied, realistic messages
- [ ] Behavioral patterns applied probabilistically, not consistently
- [ ] Message length constraints enforced
- [ ] Assistant prompt includes domain-specific requirements
- [ ] Anti-patterns explicitly listed
- [ ] Checkpointing enabled
- [ ] Pilot batch assessed with ≥70% pass rate
