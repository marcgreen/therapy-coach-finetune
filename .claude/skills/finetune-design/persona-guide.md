# Persona Design Guide

How to design user personas for realistic, diverse training data.

---

## The Problem

Generic personas produce homogeneous conversations:

```yaml
# BAD: Shallow persona
persona:
  name: "Alex"
  issue: "anxiety"
  style: "conversational"
```

Every "anxious person" sounds the same. The model learns to handle one type of user, not the full distribution.

---

## The Solution: Multi-Dimensional Personas

Personas should vary across multiple dimensions that affect conversation dynamics:

| Dimension | What It Controls |
|-----------|------------------|
| Communication style | How they express themselves |
| Behavior patterns | How they engage with help |
| Trajectory | How the situation evolves |
| Domain-specific | Context relevant to your domain |

---

## Persona Template Structure

```yaml
persona_template:
  # Identity (for consistency across conversation)
  name: str
  age_range: str
  background: str

  # Communication
  communication_style:
    options: [terse, casual, formal, stream-of-consciousness]
    weights: [0.15, 0.50, 0.25, 0.10]

  # Behavior patterns ("flaws")
  flaw_patterns:
    primary: str | null      # 50% chance per message
    secondary: list[str]     # 20% chance each per message

  # Situation evolution
  trajectory: Literal["stable", "improving", "deteriorating", "volatile"]

  # Domain-specific
  topic_seeds: list[dict]    # Starting topics with complexity
  # ... other domain attributes
```

---

## Communication Styles

How the user expresses themselves in text:

| Style | Description | Word Range | Example |
|-------|-------------|------------|---------|
| Terse | Minimal words | 30-80 | "feeling anxious today" |
| Casual | Natural, flowing | 80-180 | "so yeah I've been thinking about..." |
| Formal | Complete sentences | 120-250 | "I wanted to discuss..." |
| Stream-of-consciousness | Long, wandering | 150-300 | Multiple topics, tangents |

**Key lesson:** Enforce word limits in generation prompt. Without hard limits, all styles converge to verbose.

```python
STYLE_LIMITS = {
    "terse": (30, 80),
    "casual": (80, 180),
    "formal": (120, 250),
    "stream-of-consciousness": (150, 300),
}
```

**Age correlation:** Consider weighting styles by demographic. Younger users → more casual/text-speak. Older users → more formal.

---

## Behavior Patterns (Flaws)

Real users don't communicate cleanly. They exhibit patterns that complicate interactions.

### Communication Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| Burying the lede | Real issue mentioned last | "Work's fine... oh and I had a panic attack" |
| Rambling | Drifts between topics | Starts with work stress, ends up on childhood |
| Vague | Not enough context | "Things have just been hard lately" |
| Contradicting | Says opposite things | "I don't care" → later "Her opinion devastates me" |
| Intellectualizing | Analyzes instead of feeling | "I suppose theoretically I might feel anxiety" |

### Resistance Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| Yes-but | Rejects while appearing to engage | "That's a good idea but..." |
| Deflecting | Uses humor to avoid depth | "Haha anyway enough about my crisis" |
| Minimizing | Downplays severity | "It's not that bad, others have real problems" |
| Testing | Pushes to see response | Asks for diagnosis, medication advice |
| Reassurance-seeking | Wants validation not exploration | "I'm not being unreasonable, right?" |

### Memory/Consistency Issues

| Pattern | Description | Example |
|---------|-------------|---------|
| Forgetting insight | Returns to old patterns | Had breakthrough, now ignores it |
| Rewriting history | Remembers differently | "I never said I was angry" (they did) |
| Mood-dependent recall | Current mood colors memory | When down: "Nothing has ever worked" |

---

## Critical: Flaws Vary Per Message

**Don't make users consistently difficult.** Real people have good and bad moments.

```yaml
flaw_application:
  primary_pattern:
    probability_per_message: 0.50  # Shows up ~half the time

  secondary_patterns:
    probability_per_message: 0.20  # Each shows up ~20%

  result:
    - Some messages: NO flaw showing (clear day)
    - Some messages: Primary only
    - Some messages: Multiple flaws stacking (rough day)
```

**Implementation:** See template section below for the `apply_flaws` function.

---

## Trajectory

How the user's situation evolves over the conversation:

| Trajectory | Description | Effect |
|------------|-------------|--------|
| Stable | Situation remains similar | Consistent themes |
| Improving | Things get better | Insights build, breakthroughs |
| Deteriorating | Things get worse | New problems, compounding |
| Volatile | Swings between better/worse | Unpredictable, mood-dependent |

**Key lesson:** Trajectory affects the situation, not communication style. A "deteriorating" trajectory means life gets harder — not that the user becomes more difficult to talk to.

---

## Flaw-Free Personas

**20% of personas should have NO flaw patterns.**

Not everyone is difficult. Some users are:
- Clear communicators who know what they want
- Ready to engage productively
- Self-aware and articulate

Initial distribution was 50% no-flaw — too high, felt unrealistic. 20% provides good balance.

---

## Example: Therapy Project Personas

> **Adapt for your domain:** This example uses therapy-specific attributes
> (attachment style, topic seeds). Replace with domain-relevant attributes.

```yaml
persona_template:
  name: str
  age_range: Literal["18-25", "26-35", "36-45", "46-55", "56+"]

  personality_traits: list[str]  # 3-5 traits
  attachment_style: Literal["anxious", "avoidant", "secure", "disorganized"]

  communication_style:
    options: [terse, text-speak, casual, formal, stream-of-consciousness]
    age_weighted: true  # Younger → text-speak, Older → formal

  flaw_patterns:
    primary:
      options: [burying_lede, yes_but, minimizing, intellectualizing, null]
      weights: [0.15, 0.15, 0.15, 0.15, 0.40]  # 40% no primary flaw

    secondary:
      options: [deflecting, contradicting, reassurance_seeking, testing]
      max_secondary: 2

  trajectory:
    options: [stable, improving, deteriorating, volatile]
    weights: [0.30, 0.30, 0.25, 0.15]

  topic_seeds:
    count: 4-6
    complexity_range: [0.3, 0.9]  # Low to high complexity
```

**Generated persona example:**
```yaml
name: "Jordan"
age_range: "26-35"
personality_traits: ["analytical", "self-critical", "reserved"]
attachment_style: "avoidant"
communication_style: "casual"
primary_flaw: "intellectualizing"
secondary_flaws: ["deflecting"]
trajectory: "improving"
topic_seeds:
  - topic: "work_stress"
    complexity: 0.6
  - topic: "romantic_relationship"
    complexity: 0.8
  - topic: "self_worth"
    complexity: 0.5
```

---

## Validation Checklist

Before finalizing your persona template:

- [ ] Communication styles defined with word limits
- [ ] Behavior patterns (flaws) catalogued for your domain
- [ ] Flaw application is per-message, not per-conversation
- [ ] ~20% of personas have no flaw patterns
- [ ] Trajectory options defined
- [ ] Distribution weights set
- [ ] Domain-specific attributes included
- [ ] Expert role-play critique applied (see [assessment-guide.md#expert-role-play-critique](../finetune-generate/assessment-guide.md#expert-role-play-critique))

---

## Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Shallow personas | Homogeneous conversations | Multi-dimensional template |
| Consistent flaws | Unrealistic, model learns narrow pattern | Per-message probability |
| All users difficult | Unrealistic, skewed training | 20% no-flaw personas |
| No word limits | Styles converge to verbose | Hard limits per style |
| Static trajectory | Conversation feels flat | Situation evolves over time |

---

## Template: Your Personas

Copy and adapt this template for your domain.

### Persona Template

```yaml
# config/persona-template.yaml
persona_template:
  # Identity (for consistency)
  name: str
  background: str

  # Communication
  communication_style:
    options: [terse, casual, formal, verbose]
    weights: [0.15, 0.50, 0.25, 0.10]
    word_limits:
      terse: [30, 80]
      casual: [80, 180]
      formal: [120, 250]
      verbose: [150, 300]

  # Behavior patterns (domain-specific)
  flaw_patterns:
    primary: str | null           # 50% chance per message
    secondary: list[str]          # 20% chance each per message

  # Situation evolution
  trajectory: Literal["stable", "improving", "deteriorating", "volatile"]

  # Domain-specific attributes
  # ... add your domain's relevant attributes
```

### Multi-Domain Flaw Examples

| Domain | Communication Flaws | Resistance Flaws | Emotional Flaws |
|--------|---------------------|------------------|-----------------|
| **Customer Support** | vague_complaint, multiple_issues, wrong_product | escalation_threat, policy_challenge | frustration, entitlement |
| **Tutoring** | shows_no_work, asks_for_answer, vague_confusion | gives_up_easily, claims_tried | test_anxiety, fixed_mindset |
| **Sales** | price_focused, competitor_comparison | objection_stacking, ghosting | decision_paralysis, fomo |
| **Medical Triage** | symptom_underreporting, dr_google, multiple_concerns | treatment_refusal, skepticism | health_anxiety, denial |

### Flaw Application Pattern

```python
def apply_flaws(persona, message_number: int) -> list[str]:
    """Apply flaws probabilistically per message."""
    active_flaws = []

    # Primary flaw: 50% chance per message
    if persona.primary_flaw and random.random() < 0.50:
        active_flaws.append(persona.primary_flaw)

    # Secondary flaws: 20% chance each
    for flaw in persona.secondary_flaws:
        if random.random() < 0.20:
            active_flaws.append(flaw)

    return active_flaws  # May be empty, one, or multiple
```

---

## Domain Example

> **Adapt for your domain:** The example in the guide uses therapy personas.
> Replace flaw patterns with domain-appropriate equivalents.

**Therapy Project:** See [examples/therapy-domain.md](../examples/therapy-domain.md) for complete flaw taxonomy with 20+ patterns across 4 categories.
