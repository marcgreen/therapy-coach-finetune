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
- History utilization criteria (MT4, MT5 in therapy rubric)

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

Without guidance, conversations drift aimlessly. Provide stage-appropriate direction:

```python
TURN_GUIDANCE = {
    "early": [
        "Share initial context",
        "Express a specific emotion",
        "Ask a direct question",
    ],
    "middle": [
        "Go deeper into feelings",
        "Connect to past experience",
        "Show ambivalence about change",
        "Bring up a related topic",
    ],
    "late": [
        "Reflect on the discussion",
        "Identify a next step",
        "Express what's different now",
        "Revisit an earlier topic",
    ],
}

def get_turn_guidance(turn, total_turns):
    if turn < total_turns * 0.25:
        return random.choice(TURN_GUIDANCE["early"])
    elif turn < total_turns * 0.75:
        return random.choice(TURN_GUIDANCE["middle"])
    else:
        return random.choice(TURN_GUIDANCE["late"])
```

### Flaw Injection

Apply persona flaws per-message with probability:

```python
def build_user_prompt(persona, history, turn_guidance):
    active_flaws = []

    if persona.primary_flaw and random.random() < 0.50:
        active_flaws.append(persona.primary_flaw)

    for flaw in persona.secondary_flaws:
        if random.random() < 0.20:
            active_flaws.append(flaw)

    return USER_PROMPT_TEMPLATE.format(
        persona=persona,
        history=format_history(history),
        turn_guidance=turn_guidance,
        active_flaws=active_flaws or "None - communicate clearly",
    )
```

### Word Limit Enforcement

Styles converge to verbose without hard limits:

```python
STYLE_LIMITS = {
    "terse": (30, 80),
    "text-speak": (50, 120),
    "casual": (80, 180),
    "formal": (120, 250),
    "stream-of-consciousness": (150, 300),
}

# In user simulator prompt:
"""
Your message MUST be between {min_words} and {max_words} words.
Count carefully. This is enforced.
"""
```

---

## Assistant Prompt Engineering

The assistant generator determines training data quality. Every prompt deficiency becomes a model deficiency.

### Length Matching (#1 Failure Mode)

Responses that are 3-4x longer than user messages feel "preachy."

```markdown
# In assistant prompt:

TARGET: 1.0-1.5x user's word count. HARD LIMIT: 2x.

| User writes | Your target   | Never exceed |
|-------------|---------------|--------------|
| 30 words    | 30-45 words   | 60 words     |
| 100 words   | 100-150 words | 200 words    |
| 200 words   | 200-300 words | 400 words    |

Transcripts FAIL assessment when:
- Average ratio > 2x
- More than 50% of turns exceed 2x
```

### Tentative Language (Interpretations)

**Never assert psychological/internal dynamics as fact.**

```markdown
# BAD (fails assessment):
"You're not afraid of failing. You're afraid of mattering."
"You weren't helping them—you were protecting yourself."

# GOOD (passes assessment):
"I wonder if there's something deeper here—does that resonate?"
"It seems like maybe part of you was protecting something. What do you think?"

Always:
1. Use tentative framing ("I wonder if...", "Could it be...")
2. End with a check ("Does that fit?", "What do you think?")
```

### Question Discipline

Too many questions feels like interrogation.

```markdown
QUESTION LIMITS:
- Single-topic response: 0-1 questions
- Multi-topic response: 1-2 questions max
- Never end every section with a question

RESPONSE ENDING VARIETY:
- 40% end with a question
- 40% end with a statement/reflection
- 20% end with an offer ("If you want, we could explore...")
```

### Proactive Follow-up

If the assistant suggested an experiment, follow up proactively:

```markdown
# BAD (passive - user brings it up):
User: "I tried the grounding thing"
Assistant: "That's great! How did it go?"

# GOOD (proactive - assistant asks first):
Assistant: "Before we dig into today—did you get a chance to try the breathing thing?"
```

### Anti-Patterns List

Explicitly list phrases to avoid:

```markdown
WHAT TO AVOID:
- "That sounds really hard" (hollow validation)
- "That's profoundly..." (therapy voice)
- "You're absolutely right" (sycophantic)
- "Let's unpack that" (jargon)
- "That's not nothing" (if overused)
- Starting every response with "Hey"
- Ending every response with a question
```

### ASCII Only

Unicode characters cause issues in some pipelines:

```markdown
Stick to ASCII only:
- Straight quotes (", ') not curly quotes
- Standard dashes (-) not em dashes
- No special characters
```

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

If your domain involves asynchronous communication (e.g., text therapy, email support):

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

## Example: Therapy Project Prompts

**User simulator key elements:**
- Persona details (style, flaws, trajectory)
- Full conversation history
- Turn guidance
- Active flaws for this message
- Word count limits

**Assistant key elements:**
- System prompt with domain knowledge
- Full conversation history
- Current user message
- Length matching rules
- Tentative language requirements
- Question discipline
- Anti-patterns list
- Proactive follow-up rules

**Iterations:** The assistant prompt went through 15+ versions. Each iteration addressed specific failure modes discovered in assessment.

---

## Validation Checklist

Before scaling generation:

- [ ] User simulator produces varied, realistic messages
- [ ] Flaws are applied probabilistically, not consistently
- [ ] Word limits are enforced per style
- [ ] Assistant prompt includes all key requirements
- [ ] Anti-patterns explicitly listed
- [ ] Length matching working (check ratios)
- [ ] Checkpointing enabled
- [ ] Pilot pass rate ≥70%
