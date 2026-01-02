# SOP 1: Domain Knowledge Extraction for Fine-tuning

> **Lessons learned from the Therapeutic Coaching Fine-tuning Project**

This SOP documents how to extract domain knowledge and build the foundational artifacts required for fine-tuning: taxonomies, rubrics, and reference materials.

---

## Overview

Before generating synthetic data, you need three foundational artifacts:

| Artifact | Purpose | Example |
|----------|---------|---------|
| **Input Taxonomy** | Defines what the model will encounter | Topics, styles, difficulty levels |
| **Evaluation Rubric** | Defines what "good" looks like | 17 criteria with binary judgments |
| **Domain Reference** | Expert knowledge the model should embody | Therapeutic frameworks, safety rules |

**Key insight:** These artifacts are iteratively refined. You will revise them based on generation failures and assessment results. Plan for 3-5 iteration cycles.

---

## Lesson 1: Build Multi-Dimensional Input Taxonomies

### The Problem

A flat list of topics produces repetitive, shallow training data. Real-world inputs vary across multiple dimensions simultaneously.

### The Solution

Structure your taxonomy with **cross-product dimensions**:

```yaml
taxonomy:
  topics:
    - name: anxiety
      weight: 0.20
      subtopics: [work_stress, social, health, general_worry, panic]
    - name: relationships
      weight: 0.20
      subtopics: [romantic, family, friendship, coworker, loneliness]
    # ... more topics with subtopics

  styles:
    terse: 0.15        # "feeling anxious"
    conversational: 0.40
    detailed: 0.25
    emotional: 0.15
    analytical: 0.05

  difficulty:
    easy: 0.30         # Clear emotion, common situation
    medium: 0.50       # Mixed feelings, some complexity
    hard: 0.20         # Ambiguous, layered, edge cases
```

### Lessons Learned

1. **Weight distributions matter.** Our initial uniform distribution produced too many edge cases. Adjust weights to match real-world frequency.

2. **Include "flaws" as a dimension.** Real users exhibit communication patterns that complicate the interaction:
   - Burying the lede (real issue mentioned last)
   - Yes-but resistance (rejects suggestions while engaging)
   - Minimizing ("It's not that bad")
   - Intellectualizing (analyzes feelings instead of feeling them)

3. **Edge cases need explicit representation.** We allocated 15% to edge cases including:
   - Crisis signals (suicidal ideation, self-harm mentions)
   - Boundary violations (requests for diagnoses, medication advice)
   - Out-of-scope requests (legal, financial)
   - Hostile/testing behavior

---

## Lesson 2: Design Binary Evaluation Rubrics

### The Problem

Numeric scores (1-5) are unreliable across LLM assessors. Different models interpret "3" differently. This creates silent data quality issues.

### The Solution

Use **binary (YES/NO/NA) criteria** with explicit definitions:

```python
CRITERIA = {
    "CQ1": {
        "name": "Accurate understanding",
        "question": "Does the response demonstrate accurate understanding of what the user said?",
        "fail_examples": [
            "User mentions workload, response interprets as performance issue"
        ]
    },
    "CQ2": {
        "name": "Appropriate handling of ambiguity",
        "question": "Does the response use tentative language when making interpretations?",
        "fail_examples": [
            "You're not afraid of failing. You're afraid of mattering."
        ]
    }
    # ... 17 total criteria
}
```

### Lessons Learned

1. **Separate safety gates from weighted criteria.** We learned that safety failures (CQ8: harmful patterns, CQ9: crisis signals) must be auto-reject regardless of overall score. A response that scores 95% but diagnoses the user is unacceptable.

2. **Define when NA is valid.** Some criteria only apply in certain contexts:
   - MT4 (History utilization): NA if no prior history to reference
   - MT7 (Coaching loop continuity): NA if no prior experiments suggested
   - CQ9 (Crisis signals): NA if no crisis signals present

3. **Define when NA is invalid.** Force the assessor to make a judgment:
   - CQ1 (Understanding): Always assessable
   - CQ8 (Harmful patterns): Always assessable
   - MT1 (Topic coverage): Always assessable if topics exist

4. **Weight categories, not individual criteria.** Group related criteria:
   ```python
   weights = {
       "comprehension": 0.15,    # CQ1, CQ2
       "connection": 0.20,       # CQ3, CQ6
       "naturalness": 0.15,      # CP2, CP4, CP5
       "multi_topic": 0.30,      # MT1, MT2, MT3, MT6 <- highest weight
       "context_use": 0.20,      # MT4, MT5, MT7
   }
   ```

---

## Lesson 3: Add Calibration Examples (Critical)

### The Problem

Different LLM assessors interpret criteria differently. In our project:
- Claude gave transcript 1000 a perfect 1.0 score
- Gemini and GPT-4 caught real issues Claude missed
- 20-30% of training data had subtle quality issues

**This is catastrophic for fine-tuning.** Training on bad examples teaches the model bad patterns.

### The Solution

Add **contrastive calibration examples** to each criterion:

```markdown
## CQ2: Appropriate handling of ambiguity

**Question:** Does the response avoid assertive interpretations without tentative framing?

### PASS Examples

**Example 1:**
User: "idk why i keep doing this"
Response: "I wonder if there's a pattern we could explore together?"
Reasoning: Uses tentative framing ("I wonder if"), invites exploration rather than asserting.

### FAIL Examples

**Example 1:**
User: "idk why i keep doing this"
Response: "That's because you're avoiding the real issue underneath."
Reasoning: Stated as fact, no tentative framing, assumes hidden motive without checking.

**Example 2:**
User: "i feel weird about it"
Response: "I'm wondering if that weird feeling is actually anger you're not letting yourself feel?"
Reasoning: Sounds tentative but leads to predetermined conclusion without gathering data first.
```

### Lessons Learned

1. **High-variability criteria need more examples.** We identified four criteria with high backend disagreement:
   - CQ2 (Mind-reading): 6-8 examples
   - CQ6 (Premature interpretation): 6-8 examples
   - CP4 (Formulaic language): 6-8 examples
   - MT7 (Coaching loop continuity): 6-8 examples

2. **Include borderline cases.** The hardest judgments are not clear PASS/FAIL but subtle distinctions. Include examples that are "borderline PASS" or "borderline FAIL" with explicit reasoning.

3. **Examples must be realistic length.** Short 1-2 turn adversarial cases don't capture:
   - Context fatigue (assessor missing issues in turn 23 of 50)
   - Cumulative patterns (CP4/CP5 are about patterns across many turns)
   - History effects (MT4/MT5/MT7 only matter with substantial history)

4. **Total calibration examples:** We ended up with 72 examples across 17 criteria (3-8 per criterion).

---

## Lesson 4: Document Domain Reference Materials

### The Problem

The model needs to embody expert knowledge that can't be fully specified in a system prompt. Without reference materials, generation prompts become inconsistent.

### The Solution

Create explicit reference documents:

1. **Therapeutic Frameworks Reference** (for our domain):
   ```markdown
   ## CBT (Cognitive Behavioral Therapy)
   - Focus: Thought patterns, cognitive distortions
   - Techniques: Thought records, evidence examination, behavioral experiments
   - When to use: Recurring negative thought patterns, catastrophizing

   ## DBT (Dialectical Behavior Therapy)
   - Focus: Emotional regulation, distress tolerance
   - Techniques: Grounding (5-4-3-2-1), TIPP, radical acceptance
   - When to use: Emotional flooding, panic, self-harm risk
   ```

2. **Safety Rules Reference**:
   ```markdown
   ## Clinical Labels - NEVER USE
   - "That's dissociation"
   - "This is health anxiety"
   - "You have anxious attachment"

   ## Instead - Describe the EXPERIENCE
   - "That sense of watching yourself from outside..."
   - "The worry about your body - that loop of 'what if'..."
   ```

3. **Response Structure Reference**:
   ```markdown
   ## Multi-Topic Response Format
   For 3+ topics:
   1. Start directly with first topic section (no generic opener)
   2. Use explicit sections: **[Topic label]:** 2-4 sentences
   3. Optional woven connection when topics interact
   ```

### Lessons Learned

1. **Reference materials become prompt content.** These documents are injected into generation prompts. Keep them concise and actionable.

2. **Versioning matters.** We iterated the assistant prompt through 15+ versions. Track changes so you can correlate prompt versions with generation quality.

3. **Anti-patterns are as important as patterns.** Document what NOT to do:
   - "DON'T end every response with a question"
   - "DON'T use Claude-isms like 'You're absolutely right'"
   - "DON'T start every opener with 'Hey'"

---

## Lesson 5: Plan for Iterative Refinement

### The Reality

Your first rubric will be wrong. Your first taxonomy will be incomplete. This is expected.

### The Process

```
Week 1: Initial artifacts
  ├── Draft taxonomy (best guess)
  ├── Draft rubric (8-10 criteria)
  └── Draft reference materials

Week 2: Pilot generation (50-100 examples)
  ├── Generate with draft artifacts
  ├── Assess with draft rubric
  ├── Analyze failures → identify missing criteria
  └── Update artifacts

Week 3-4: Iterate (repeat 2-3 times)
  ├── Add criteria for new failure modes
  ├── Add calibration examples where assessors disagree
  ├── Refine taxonomy weights based on pass rates
  └── Expand reference materials for gaps

Week 5+: Scale generation
  └── Artifacts stabilized, now scale volume
```

### Lessons Learned

1. **Rubric criteria emerge from failures.** We started with 12 criteria. Added 5 more (MT1-MT7) after seeing multi-topic handling failures.

2. **Calibration examples emerge from backend disagreement.** Run the same transcripts through multiple backends (Claude, GPT-4, Gemini). Where they disagree, add calibration examples.

3. **The rubric is never "done."** Even after 7,500+ transcripts, we were still refining criteria wording and calibration examples.

---

## Artifact Checklist

Before starting data generation, ensure you have:

- [ ] **Input taxonomy** with weighted topics, styles, difficulty levels
- [ ] **Flaw patterns** for realistic human communication (if applicable)
- [ ] **Evaluation rubric** with:
  - [ ] Binary (YES/NO/NA) criteria
  - [ ] Safety gates (auto-reject criteria)
  - [ ] Weighted category groupings
  - [ ] NA-valid vs NA-invalid specification
  - [ ] 3-8 calibration examples per criterion
- [ ] **Domain reference materials**:
  - [ ] Expert knowledge documentation
  - [ ] Safety rules and boundaries
  - [ ] Response structure guidelines
  - [ ] Anti-patterns to avoid

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Numeric scores (1-5) | Inconsistent across assessors | Binary YES/NO/NA |
| Single-dimension taxonomy | Repetitive data | Cross-product of dimensions |
| No calibration examples | Backend disagreement | Contrastive examples per criterion |
| No safety gates | Bad examples in training data | Auto-reject on safety criteria |
| Static artifacts | Miss emerging failure modes | Plan for 3-5 iteration cycles |

---

*Last updated: January 2026*
*Based on therapeutic coaching fine-tuning project with 7,500+ transcripts*
