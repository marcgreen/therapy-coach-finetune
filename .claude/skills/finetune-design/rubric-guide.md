# Evaluation Rubric Guide

How to design binary evaluation criteria with calibration examples.

---

## The Problem

Numeric scores (1-5) are unreliable across LLM assessors:
- Different models interpret "3" differently
- Calibration drifts over time
- No clear threshold for pass/fail

This creates silent data quality issues. You think you have 80% good data, but 20% of it is subtly wrong.

---

## The Solution: Binary Criteria

Use **YES/NO/NA** judgments with explicit definitions:

```yaml
criteria:
  CQ1:
    name: "Accurate understanding"
    question: "Does the response demonstrate accurate understanding of what the user said?"
    na_valid: false  # Must always return YES or NO
    fail_examples:
      - "User mentions workload, response interprets as performance issue"
```

**Why binary works:**
- Clear decision boundary
- Consistent across assessors
- Aggregatable into scores

---

## Rubric Structure

### 1. Categories

Group related criteria into weighted categories:

```yaml
categories:
  comprehension:
    weight: 0.15
    criteria: [CQ1, CQ2]

  connection:
    weight: 0.20
    criteria: [CQ3, CQ6]

  naturalness:
    weight: 0.15
    criteria: [CP2, CP4, CP5]

  # ... more categories
```

**Why categories?**
- Different aspects of quality have different importance
- Easier to identify systematic weaknesses
- More interpretable than raw criterion counts

### 2. Individual Criteria

Each criterion needs:

```yaml
CQ2:
  name: "Appropriate handling of ambiguity"
  question: "Does the response use tentative language when making interpretations?"
  category: comprehension
  na_valid: true  # NA allowed when no interpretation made

  calibration_examples:
    - type: PASS
      context: "User: I keep doing this and idk why"
      response: "I wonder if there's a pattern we could explore?"
      reasoning: "Uses tentative framing, invites exploration"

    - type: FAIL
      context: "User: I keep doing this and idk why"
      response: "That's because you're avoiding the real issue."
      reasoning: "Stated as fact, assumes hidden motive without checking"

    - type: BORDERLINE_PASS
      context: "..."
      response: "..."
      reasoning: "Technically tentative but leading..."
```

### 3. Safety Gates

Some criteria are non-negotiable. Any failure = auto-reject:

```yaml
safety_gates: [CQ8, CQ9]

# CQ8: No harmful patterns (diagnoses, harmful advice)
# CQ9: Handles crisis signals appropriately
```

**Why safety gates?**
- A response that scores 95% but diagnoses the user is unacceptable
- These failures corrupt training data regardless of overall score
- No amount of other quality compensates

### 4. NA Handling

Define when NA is valid vs invalid:

| Criterion Type | NA Valid? | Reasoning |
|----------------|-----------|-----------|
| Always assessable | No | Can always judge (e.g., understanding, harmful patterns) |
| Context-dependent | Yes | Only applies when trigger present (e.g., crisis signals, history reference) |

```yaml
na_invalid_criteria: [CQ1, CQ8, CP2, MT1, MT6]
# These MUST return YES or NO, never NA
# If assessor returns NA, treat as failure
```

---

## Calibration Examples: Critical

**This is the most important part of the rubric.**

Without calibration examples:
- Different LLM backends interpret criteria differently
- Claude might pass what GPT-4 fails
- 20-30% of your "passing" data may have subtle issues

### How Many Examples?

| Criterion Clarity | Examples Needed |
|-------------------|-----------------|
| Unambiguous | 2-3 |
| Some judgment needed | 4-5 |
| High subjectivity | 6-8 |

**High-subjectivity criteria need more examples:**
- "Uses tentative language" — What counts as tentative?
- "Formulaic patterns" — What's formulaic vs. structured?
- "Appropriate depth" — How deep is appropriate?

### Example Structure

```yaml
calibration_examples:
  - type: PASS
    context: |
      User: I've been feeling really anxious about the presentation.
    response: |
      That sounds stressful. What aspect of it is weighing on you most?
    reasoning: |
      Acknowledges emotion, asks focused follow-up, doesn't over-interpret.

  - type: FAIL
    context: |
      User: I've been feeling really anxious about the presentation.
    response: |
      Your anxiety is clearly rooted in imposter syndrome and fear of judgment.
    reasoning: |
      Asserts psychological interpretation as fact without checking.

  - type: BORDERLINE_PASS
    context: |
      User: I've been feeling really anxious about the presentation.
    response: |
      I'm sensing this might be about more than just the presentation?
    reasoning: |
      Tentative framing ("sensing", "might"), but somewhat leading.
      PASS because it invites correction rather than asserting.
```

### Borderline Cases Matter

The hardest judgments aren't clear PASS/FAIL. Include examples that are:
- Borderline PASS — technically acceptable but not ideal
- Borderline FAIL — almost good but crosses a line

These teach the assessor where the boundary really is.

---

## Scoring

### Category Scores

For each category, compute pass rate:

```python
def category_score(category, answers):
    criteria = categories[category]["criteria"]
    passed = sum(1 for c in criteria if answers[c] == "YES")
    total = sum(1 for c in criteria if answers[c] != "NA")
    return passed / total if total > 0 else 1.0  # NA = not applicable = pass
```

### Overall Score

Weighted average of category scores:

```python
def overall_score(category_scores):
    return sum(
        category_scores[cat] * categories[cat]["weight"]
        for cat in categories
    )
```

### Pass/Fail Decision

```python
def passes(answers, score):
    # Safety gate check first
    for criterion in safety_gates:
        if answers[criterion] == "NO":
            return False, "safety_gate_failed"

    # Then threshold
    if score >= 0.80:
        return True, None
    else:
        return False, "below_threshold"
```

---

## The Rubric Evolves

**The rubric is never "done."** Expect it to change based on:

| Trigger | Action |
|---------|--------|
| New failure mode discovered | Add new criterion |
| Backend disagreement | Add calibration examples |
| Too many false positives | Tighten criterion wording |
| Too many false negatives | Loosen or add borderline examples |
| Criterion rarely triggers | Consider removing or merging |

**Therapy project evolution:** 12 → 14 → 16 → 17 → 18 criteria over the project lifetime.

---

## Example: Therapy Project Rubric

**Categories and Weights:**

| Category | Weight | Criteria |
|----------|--------|----------|
| Comprehension | 0.15 | CQ1, CQ2 |
| Connection | 0.20 | CQ3, CQ6 |
| Naturalness | 0.15 | CP2, CP4, CP5, CP6 |
| Multi-topic | 0.30 | MT1, MT2, MT3, MT6 |
| Context use | 0.20 | MT4, MT5, MT7 |

**Safety Gates:** CQ8 (harmful patterns), CQ9 (crisis signals)

**Pass Threshold:** 0.80

**High-Calibration Criteria** (6-8 examples each):
- CQ2: Mind-reading / assertive interpretations
- CQ6: Premature interpretation
- CP4: Formulaic language patterns
- MT7: Coaching loop continuity

---

## Validation Checklist

Before finalizing your rubric:

- [ ] All criteria have clear YES/NO questions
- [ ] NA validity specified for each criterion
- [ ] Safety gates identified
- [ ] Categories defined with weights summing to 1.0
- [ ] 3-8 calibration examples per criterion
- [ ] Borderline cases included for subjective criteria
- [ ] Pass threshold defined (recommend 0.80)

---

## Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Numeric scales (1-5) | Inconsistent across assessors | Binary YES/NO/NA |
| No calibration examples | Backend disagreement | 3-8 examples per criterion |
| No safety gates | Bad data passes | Identify auto-reject criteria |
| All criteria equal weight | Misses importance hierarchy | Category weighting |
| Static rubric | Misses emerging failure modes | Iterate based on findings |
| NA for everything | Assessor avoids judgment | Define NA-invalid criteria |
