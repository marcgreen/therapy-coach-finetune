# Input Taxonomy

An input taxonomy defines the distribution of user messages to generate. Without systematic variation, synthetic data clusters around common patterns and the model overfits.

## The Diversity Problem

Most taxonomies only capture **WHAT** the user is asking about:

```yaml
# Incomplete taxonomy - only WHAT
topics: [billing, technical, account]
difficulty: [easy, medium, hard]
```

This misses critical variation in **WHO** the user is, **HOW** they communicate, and **HOW** their problem presents. A model trained on this will handle "billing question from a neurotypical adult" but fail on "billing question from a frustrated teenager with ADHD communication patterns."

## Comprehensive Taxonomy Framework

A complete taxonomy covers five dimensions:

| Dimension | Question | Examples |
|-----------|----------|----------|
| **WHAT** | What are they asking about? | Topics, subtopics, difficulty |
| **HOW** | How do they communicate? | Style, cadence, cognitive patterns |
| **WHO** | Who are they? | Age, culture, communication pattern |
| **RELATIONSHIP** | How do they relate to help? | Naive, skeptical, testing |
| **PRESENTATION** | How does the problem show up? | Clean, complex, hidden |

### Dimension 1: WHAT (Content)

```yaml
topics:
  - name: "topic_name"
    weight: 0.30
    subtopics: [...]

difficulty:
  easy: 0.30       # Clear intent, common scenario
  medium: 0.50     # Some ambiguity, requires judgment
  hard: 0.20       # Edge cases, complex scenarios

edge_cases:
  - out_of_scope   # Should trigger refusal
  - ambiguous      # Needs clarification
  - multi_intent   # User wants several things
```

### Dimension 2: HOW (Communication Style)

```yaml
styles:
  terse: 0.15            # "help me"
  conversational: 0.40   # "Hey, I've been thinking about..."
  detailed: 0.25         # Multi-paragraph context
  emotional: 0.15        # Strong feelings expressed
  analytical: 0.05       # Logical, structured

interaction_cadence:
  frequent_short: 0.50   # Quick back-and-forth, short messages
  infrequent_detailed: 0.50  # Long, substantial messages (journal-style)

cognitive_patterns:
  balanced: 0.85         # Realistic perspective
  distorted: 0.15        # Catastrophizing, all-or-nothing (tests yes-bot detection)
```

### Dimension 3: WHO (Identity)

**This dimension is often completely missing from taxonomies.**

```yaml
age_context:
  young_adult: 0.30      # 18-25: identity formation, career start
  middle_adult: 0.45     # 26-50: responsibilities, midlife
  older_adult: 0.15      # 50+: transitions, health, legacy
  teenager: 0.10         # 13-17: developmental, school, identity

cultural_framing:
  individualist: 0.55    # "I feel", "I want", personal growth focus
  collectivist: 0.25     # "My family expects", duty, shame, harmony
  mixed: 0.20            # Bicultural tension, navigating multiple value systems

communication_pattern:
  neurotypical: 0.65     # Standard conversational patterns
  direct_literal: 0.15   # Autistic-pattern: precise, explicit, literal
  tangential_energetic: 0.10  # ADHD-pattern: topic-jumping, high energy
  limited_vocabulary: 0.10    # Struggles to articulate (uses "bad" for many feelings)
```

**Why this matters:**
- Without age variation, the model learns "adult professional" language only
- Without cultural framing, the model imposes individualist values on collectivist users
- Without communication patterns, the model fails neurodivergent users systematically

### Dimension 4: RELATIONSHIP (Help-Seeking Stance)

```yaml
help_seeking:
  explicit: 0.30         # "What should I do?" / "Can you help?"
  implicit: 0.70         # Expresses need, expects guidance without asking

help_relationship:
  naive: 0.25            # First time seeking help, doesn't know what to expect
  experienced: 0.35      # Knows the domain, may have expectations
  skeptical: 0.15        # Doubts it will work, "my partner made me try this"
  dependent: 0.10        # Overly reliant on external validation
  testing: 0.15          # Pushing boundaries, checking if AI will break guidelines
```

### Dimension 5: PRESENTATION (Problem Complexity)

```yaml
presentation:
  clean_single: 0.40     # One clear issue, straightforward
  comorbid: 0.25         # Multiple issues intertwined
  somatic: 0.10          # Physical symptoms masking other concerns
  decoy: 0.10            # Surface problem hides real issue
  spiraling: 0.15        # Jumps between concerns, hard to focus

temporality:
  acute: 0.20            # Just happened, high urgency
  chronic: 0.40          # Longstanding pattern
  triggered: 0.20        # Recent event activated old pattern
  building: 0.20         # Escalating across conversation
```

## Full Example: Human-Facing AI

```yaml
taxonomy:
  # WHAT
  topics:
    - name: billing
      weight: 0.25
      subtopics: [charges, refunds, subscriptions]
    - name: technical
      weight: 0.30
      subtopics: [bugs, setup, integration]
    # ... more topics

  difficulty:
    easy: 0.40
    medium: 0.40
    hard: 0.20

  # HOW
  styles:
    frustrated: 0.25
    confused: 0.30
    professional: 0.25
    urgent: 0.15
    appreciative: 0.05

  interaction_cadence:
    frequent_short: 0.60
    infrequent_detailed: 0.40

  # WHO
  age_context:
    young_adult: 0.35
    middle_adult: 0.45
    older_adult: 0.15
    teenager: 0.05

  cultural_framing:
    individualist: 0.60
    collectivist: 0.25
    mixed: 0.15

  communication_pattern:
    neurotypical: 0.70
    direct_literal: 0.15
    tangential_energetic: 0.10
    limited_vocabulary: 0.05

  # RELATIONSHIP
  help_seeking:
    explicit: 0.50
    implicit: 0.50

  help_relationship:
    naive: 0.30
    experienced: 0.40
    skeptical: 0.15
    dependent: 0.05
    testing: 0.10

  # PRESENTATION
  presentation:
    clean_single: 0.50
    comorbid: 0.20
    somatic: 0.05
    decoy: 0.10
    spiraling: 0.15

  temporality:
    acute: 0.30
    chronic: 0.40
    triggered: 0.15
    building: 0.15
```

## Generating from Taxonomy

```python
def sample_from_taxonomy(taxonomy: dict) -> dict:
    """Sample one configuration from taxonomy."""
    tax = taxonomy["taxonomy"]

    return {
        # WHAT
        "topic": weighted_choice(tax["topics"])["name"],
        "subtopic": random.choice(topic["subtopics"]),
        "difficulty": weighted_choice(tax["difficulty"]),
        # HOW
        "style": weighted_choice(tax["styles"]),
        "cadence": weighted_choice(tax["interaction_cadence"]),
        "cognitive_patterns": weighted_choice(tax.get("cognitive_patterns", {"balanced": 1.0})),
        # WHO
        "age_context": weighted_choice(tax.get("age_context", {"adult": 1.0})),
        "cultural_framing": weighted_choice(tax.get("cultural_framing", {"individualist": 1.0})),
        "communication_pattern": weighted_choice(tax.get("communication_pattern", {"neurotypical": 1.0})),
        # RELATIONSHIP
        "help_seeking": weighted_choice(tax.get("help_seeking", {"explicit": 1.0})),
        "help_relationship": weighted_choice(tax.get("help_relationship", {"naive": 1.0})),
        # PRESENTATION
        "presentation": weighted_choice(tax.get("presentation", {"clean_single": 1.0})),
        "temporality": weighted_choice(tax.get("temporality", {"chronic": 1.0})),
    }
```

## Generation Prompt Integration

When generating, include all dimensions in the prompt:

```python
GENERATION_PROMPT = """Generate a realistic user message.

=== SCENARIO ===
Topic: {topic} ({subtopic})
Difficulty: {difficulty}

=== WHO THE USER IS ===
Age context: {age_context}
Cultural framing: {cultural_framing}
Communication pattern: {communication_pattern}

=== HOW THEY COMMUNICATE ===
Style: {style}
Cadence: {cadence}

=== RELATIONSHIP TO HELP ===
Help-seeking: {help_seeking}
Help relationship: {help_relationship}

=== PROBLEM PRESENTATION ===
Presentation: {presentation}
Temporality: {temporality}
"""
```

## Diversity Validation

After generation, verify diversity across ALL dimensions:

```python
def check_diversity(records: list[dict], dimensions: list[str]) -> dict:
    """Check that each dimension has adequate coverage."""
    coverage = {}
    for dim in dimensions:
        values = [r["metadata"].get(dim) for r in records if r["metadata"].get(dim)]
        unique = set(values)
        counts = {v: values.count(v) for v in unique}

        # Check if any value is >80% of total (over-represented)
        max_pct = max(counts.values()) / len(values) if values else 0
        coverage[dim] = {
            "unique_values": len(unique),
            "max_concentration": max_pct,
            "pass": max_pct < 0.80,
        }
    return coverage
```

## Anti-patterns

| Pattern | Problem | Fix |
|---------|---------|-----|
| Only WHAT dimensions | Model fails on WHO/HOW variation | Add identity and relationship dimensions |
| No cultural variation | Model imposes majority values | Add cultural_framing dimension |
| No neurodivergent patterns | Model fails 15-20% of users | Add communication_pattern dimension |
| No relationship variation | Model can't handle skepticism/testing | Add help_relationship dimension |
| Only "clean" problems | Model fails on complexity | Add presentation and temporality |
| Uniform weighting | Rare-but-important cases underrepresented | Weight by real-world distribution |

## Minimum Viable Taxonomy

If resource-constrained, prioritize in this order:

1. **WHAT** (topics, difficulty) - Basic requirement
2. **HOW** (styles) - User experience varies significantly
3. **WHO: communication_pattern** - Neurodivergent users need representation
4. **WHO: cultural_framing** - Prevents value imposition
5. **RELATIONSHIP** - Handles resistant/testing users
6. **PRESENTATION** - Handles complex real-world scenarios

The first four are critical. The last two can be added in iteration.
