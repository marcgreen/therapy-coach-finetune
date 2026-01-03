# Input Taxonomy Guide

How to design a comprehensive input taxonomy for diverse training data.

---

## The Problem

A flat list of topics produces repetitive, shallow training data:

```yaml
# BAD: One-dimensional
topics: [billing, technical, account, general]
```

Real-world inputs vary across multiple dimensions simultaneously. A "billing question" from a frustrated expert is completely different from a "billing question" from a confused novice.

---

## The Framework: Five Dimensions

A comprehensive taxonomy captures variation across five dimensions:

| Dimension | Question | Examples |
|-----------|----------|----------|
| **WHAT** | What are they asking about? | Topics, subtopics, complexity |
| **HOW** | How do they communicate? | Style, verbosity, tone |
| **WHO** | Who are they? | Experience, context, background |
| **RELATIONSHIP** | How do they relate to help? | Trusting, skeptical, testing |
| **PRESENTATION** | How does the problem show up? | Clear, vague, multi-part |

### Dimension 1: WHAT (Content)

The subject matter of the conversation:

```yaml
topics:
  - name: "primary_topic"
    weight: 0.30
    subtopics:
      - subtopic_1
      - subtopic_2
      - subtopic_3

  - name: "secondary_topic"
    weight: 0.25
    subtopics: [...]

difficulty:
  easy: 0.30      # Clear intent, common scenario
  medium: 0.50    # Some ambiguity, requires judgment
  hard: 0.20      # Edge cases, complex scenarios
```

### Dimension 2: HOW (Communication Style)

How the user expresses themselves:

> **Note:** Style names are domain-customizable. The examples below are generic.
> Therapy might use: terse, text-speak, casual, formal, stream-of-consciousness.
> Customer support might use: terse, frustrated, detailed, polite, demanding.

```yaml
style:
  terse: 0.15           # "need help with X"
  conversational: 0.40  # Natural, flowing
  detailed: 0.25        # Full context provided
  emotional: 0.15       # Feelings prominent
  analytical: 0.05      # "I notice a pattern..."
```

### Dimension 3: WHO (User Context)

Who the user is and their background:

```yaml
experience:
  novice: 0.30      # New to the domain
  intermediate: 0.50
  expert: 0.20      # Knows the domain well

context:
  # Domain-specific attributes
  # e.g., for support: enterprise vs consumer
  # e.g., for therapy: first-time vs long-term
```

### Dimension 4: RELATIONSHIP (Stance Toward Help)

How the user relates to receiving help:

```yaml
stance:
  open: 0.40           # Ready to engage
  skeptical: 0.25      # "Will this actually help?"
  testing: 0.15        # Probing boundaries
  resistant: 0.20      # Doesn't really want change
```

### Dimension 5: PRESENTATION (Problem Structure)

How the problem manifests:

```yaml
presentation:
  clear: 0.40          # Obvious what's needed
  vague: 0.25          # Needs clarification
  multi_part: 0.20     # Several issues at once
  buried: 0.15         # Real issue hidden
```

---

## Weights Matter

Weights control the distribution of training data. Get them wrong and your model will be miscalibrated.

**Common mistakes:**

| Mistake | Consequence | Better Approach |
|---------|-------------|-----------------|
| Uniform weights | Too many edge cases | Match real-world frequency |
| No edge cases | Model fails on boundaries | Allocate ~15% to edges |
| All "medium" difficulty | Model overconfident | Include genuinely hard cases |

**Calibration approach:**
1. Start with your best guess
2. Generate a pilot batch
3. Review distribution — does it feel realistic?
4. Adjust weights and regenerate

---

## Edge Cases: Explicit Representation

Edge cases won't appear naturally — you must explicitly allocate them:

```yaml
edge_cases:
  weight: 0.15  # ~15% of training data

  types:
    - out_of_scope        # Should trigger refusal or redirect
    - boundary_violation  # Asks for something inappropriate
    - ambiguous_intent    # Genuinely unclear what they want
    - multi_intent        # Wants several conflicting things
    - hostile             # Aggressive, testing limits
```

**Why 15%?**
- Too low (5%): Model won't learn to handle edges
- Too high (30%): Model becomes paranoid about normal requests
- 15% provides exposure without skewing the distribution

---

## Cross-Product Sampling

When generating, sample across dimensions independently:

```python
def sample_input():
    return {
        "topic": weighted_choice(taxonomy["topics"]),
        "difficulty": weighted_choice(taxonomy["difficulty"]),
        "style": weighted_choice(taxonomy["style"]),
        "experience": weighted_choice(taxonomy["experience"]),
        "stance": weighted_choice(taxonomy["stance"]),
        "presentation": weighted_choice(taxonomy["presentation"]),
    }
```

This creates natural variation. A "billing + terse + expert + skeptical" combination is different from "billing + detailed + novice + open" even though the topic is the same.

---

## Example: Therapy Project

> **Adapt for your domain:** This example uses therapy-specific topics
> (anxiety, relationships, life_transitions). Replace with domain-relevant topics.

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
      subtopics: [confidence, imposter, self_criticism, perfectionism]
    - name: emotional_regulation
      weight: 0.15
      subtopics: [anger, sadness, overwhelm, numbness, mood_swings]
    - name: edge_cases
      weight: 0.15
      subtopics:
        - crisis_signals      # Suicidal ideation, self-harm
        - medical_advice      # Requests for diagnoses
        - out_of_scope        # Legal, financial
        - hostile             # Aggressive, testing

  styles:
    terse: 0.15
    conversational: 0.40
    detailed: 0.25
    emotional: 0.15
    analytical: 0.05

  difficulty:
    easy: 0.30
    medium: 0.50
    hard: 0.20
```

**Lessons learned:**
1. Initial uniform distribution produced too many edge cases — adjusted weights
2. Added "flaws" dimension separately (see persona-guide.md)
3. Subtopics within topics provided necessary variety

---

## Validation Checklist

Before finalizing your taxonomy:

- [ ] All major topics covered with subtopics
- [ ] Weights sum to 1.0 within each dimension
- [ ] Edge cases explicitly represented (~15%)
- [ ] Multiple difficulty levels included
- [ ] Distribution feels realistic (not too many extremes)
- [ ] Pilot generation shows good variety
- [ ] Expert role-play critique applied (see [assessment-guide.md#expert-role-play-critique](../finetune-generate/assessment-guide.md#expert-role-play-critique))

---

## Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Flat topic list | Shallow, repetitive data | Multi-dimensional taxonomy |
| No edge cases | Model fails at boundaries | Explicit 15% allocation |
| Uniform weights | Unrealistic distribution | Match real-world frequency |
| Only "medium" difficulty | Model overconfident | Include genuinely hard cases |
| Static taxonomy | Misses emerging patterns | Iterate based on failures |

---

## Template: Your Taxonomy

Copy and adapt this template for your domain.

### Complete Taxonomy Template

```yaml
# config/taxonomy.yaml
taxonomy:
  # ===================
  # DIMENSION 1: WHAT (Content)
  # ===================
  topics:
    - name: "[topic_1]"
      weight: 0.XX           # Weights must sum to 1.0
      subtopics:
        - [subtopic_1a]
        - [subtopic_1b]
        - [subtopic_1c]

    - name: "[topic_2]"
      weight: 0.XX
      subtopics:
        - [subtopic_2a]
        - [subtopic_2b]

    - name: "edge_cases"
      weight: 0.15           # Always allocate ~15% to edges
      subtopics:
        - out_of_scope       # Should trigger refusal/redirect
        - boundary_violation # Inappropriate requests
        - ambiguous_intent   # Unclear what they want
        - hostile            # Aggressive, testing limits

  difficulty:
    easy: 0.30               # Clear intent, common scenarios
    medium: 0.50             # Some ambiguity, requires judgment
    hard: 0.20               # Edge cases, complex scenarios

  # ===================
  # DIMENSION 2: HOW (Communication Style)
  # ===================
  styles:
    terse: 0.15              # Minimal words
    conversational: 0.40     # Natural, flowing
    detailed: 0.25           # Full context provided
    emotional: 0.15          # Feelings prominent
    analytical: 0.05         # Pattern-focused

  # ===================
  # DIMENSION 3: WHO (User Context)
  # ===================
  experience:
    novice: 0.30             # New to domain
    intermediate: 0.50       # Some familiarity
    expert: 0.20             # Deep knowledge

  # Domain-specific context (customize these)
  context:
    [context_dimension_1]: [options with weights]
    [context_dimension_2]: [options with weights]

  # ===================
  # DIMENSION 4: RELATIONSHIP (Stance Toward Help)
  # ===================
  stance:
    open: 0.40               # Ready to engage
    skeptical: 0.25          # Doubtful it will help
    testing: 0.15            # Probing boundaries
    resistant: 0.20          # Doesn't want change

  # ===================
  # DIMENSION 5: PRESENTATION (Problem Structure)
  # ===================
  presentation:
    clear: 0.40              # Obvious what's needed
    vague: 0.25              # Needs clarification
    multi_part: 0.20         # Several issues at once
    buried: 0.15             # Real issue hidden
```

### Multi-Domain Topic Examples

These are examples, not a comprehensive list. Discover what matters for your domain through brainstorming, expert roleplay, and iteration.

| Domain | Primary Topics | Edge Cases |
|--------|----------------|------------|
| **Customer Support** | billing, technical, account, shipping, returns, features | escalation_demand, legal_threat, competitor_mention, profanity |
| **Tutoring** | concept_explanation, problem_solving, exam_prep, homework_help | do_my_homework, plagiarism_request, unrelated_subject, inappropriate |
| **Sales/Onboarding** | product_features, pricing, comparison, integration, security | competitor_bashing, unrealistic_timeline, budget_mismatch, tire_kicker |
| **Medical Triage** | symptom_assessment, medication_question, appointment, follow_up | emergency_symptoms, diagnosis_request, prescription_request, mental_health_crisis |
| **???** | Your domain | Discovered through brainstorming, expert roleplay, and failed assessments |

### Multi-Domain Context Examples

These are examples, not a comprehensive list. Discover what matters for your domain through brainstorming, expert roleplay, and iteration.

| Domain | Experience Levels | Domain-Specific Context |
|--------|-------------------|------------------------|
| **Customer Support** | new_customer, returning, power_user | plan_tier: [free, pro, enterprise], account_age: [new, established, long_term] |
| **Tutoring** | beginner, developing, advanced | grade_level: [elementary, middle, high, college], learning_style: [visual, verbal, hands_on] |
| **Sales** | cold_lead, warm_lead, existing_customer | deal_stage: [awareness, consideration, decision], company_size: [smb, mid_market, enterprise] |
| **Medical** | first_visit, regular_patient, chronic_condition | urgency: [routine, concerning, urgent], insurance: [insured, uninsured, medicare] |
| **???** | Your domain | Discovered through brainstorming, expert roleplay, and failed assessments |

### Sampling Implementation

```python
import random

def weighted_choice(options: dict[str, float]) -> str:
    """Select from weighted options."""
    items = list(options.keys())
    weights = list(options.values())
    return random.choices(items, weights=weights, k=1)[0]

def sample_input(taxonomy: dict) -> dict:
    """Sample across all dimensions independently."""
    return {
        "topic": weighted_choice(taxonomy["topics"]),
        "difficulty": weighted_choice(taxonomy["difficulty"]),
        "style": weighted_choice(taxonomy["styles"]),
        "experience": weighted_choice(taxonomy["experience"]),
        "stance": weighted_choice(taxonomy["stance"]),
        "presentation": weighted_choice(taxonomy["presentation"]),
    }

def sample_batch(taxonomy: dict, n: int) -> list[dict]:
    """Generate n independent samples."""
    return [sample_input(taxonomy) for _ in range(n)]
```

---

## Domain Example

> **Adapt for your domain:** The example in the guide uses therapy topics.
> For other domains, see the multi-domain tables above and
> [examples/therapy-domain.md](../examples/therapy-domain.md) for the complete therapy taxonomy.
