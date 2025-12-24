---
name: finetune-prep
description: This skill should be used when the user wants to "prepare for fine-tuning", "create synthetic training data", "build evaluation rubric", "extract domain knowledge with LLM", "design data generation pipeline", or needs to establish quality standards before training an LLM on a specialized domain.
---

# Fine-Tuning Preparation with LLM-Based Domain Extraction

## Overview

This skill guides the complete preparation process for fine-tuning LLMs on specialized domains using **LLM-as-expert** for domain knowledge extraction, synthetic data generation, and evaluation design.

**Core principle:** Quality training data requires explicit domain knowledge, behavioral anchors, and binary evaluation criteria - all created BEFORE generating a single training example.

**Key insight:** Using LLMs to role-play domain experts works when expertise is encoded into structured prompts with concrete behavioral anchors, failure mode guards, and binary quality gates.

## When to Use

Use this skill when:
- Preparing to fine-tune a model for a specialized domain (therapy, legal, customer support, etc.)
- Building a synthetic data pipeline from LLM-generated expertise
- Need to define "good" before generating training data
- Working with 7B-70B parameter models where base performance is poor
- Have no direct access to domain experts but can research frameworks

**Do NOT use for:**
- Simple prompt engineering (no fine-tuning needed)
- Domains where you have abundant real training data
- Tasks where base model already performs well
- Quick prototyping without quality gates

## The Process

```
Phase 1: Domain Knowledge Extraction
         ↓
Phase 2: Behavioral Anchor Definition
         ↓
Phase 3: Binary Evaluation Rubric
         ↓
Phase 4: Evaluation Prompt Design
         ↓
Phase 5: Generation Prompt Design
```

**Critical order:** Evaluation criteria MUST be defined before generation prompts. You cannot improve what you cannot measure.

## Phase 1: Domain Knowledge Extraction

**Goal:** Research and synthesize domain frameworks into structured reference documentation.

### Steps

1. **Identify relevant frameworks for the domain**
   - Therapy → CBT, DBT, ACT
   - Legal → Legal reasoning frameworks, precedent analysis
   - Customer support → Service recovery, escalation frameworks
   - Code review → Style guides, best practices, security patterns

2. **Research each framework**
   - Use WebSearch for current best practices (include year: 2025)
   - Focus on: core principles, specific techniques, conversation patterns
   - Capture: what practitioners DO (observable behaviors), not just theory

3. **Search for documented failure modes**
   - AI chatbot failures in this domain (search: "AI [domain] mistakes research")
   - Common human errors in the domain
   - Documented safety issues and ethical violations

4. **Create framework reference document**
   - Synthesize frameworks with concrete examples
   - Include conversation snippets showing techniques in practice
   - Document failure modes with specific examples
   - Cross-reference techniques (when to use X vs Y)

**Output:** `references/domain-frameworks.md` - comprehensive reference covering frameworks, techniques, communication principles, and failure modes.

**Quality check:**
- Does it include CONCRETE examples (not just abstractions)?
- Are techniques named and explained with conversation examples?
- Are failure modes specific (with real examples)?
- Could you generate 100 conversations from this without running out of patterns?

## Phase 2: Behavioral Anchor Definition

**Goal:** Transform abstract qualities into observable DO/DON'T behaviors.

**Why this matters:** "Be empathetic" is not actionable. "Acknowledge emotion before problem-solving" is.

### Steps

1. **Extract ALWAYS DO behaviors**
   - From frameworks: What should EVERY response do?
   - From communication principles: Observable actions (reflect back, ask permission, etc.)
   - Make each item behavioral (verb-based, observable)
   - Target: 6-10 specific behaviors

2. **Extract NEVER DO behaviors**
   - From failure modes: What AI chatbots do wrong?
   - From domain ethics: What crosses boundaries?
   - From safety research: What causes harm?
   - Target: 8-12 specific anti-patterns

3. **Ensure orthogonality**
   - Each behavior checks something distinct
   - No redundancy ("validate feelings" vs "show empathy" → pick one, define clearly)
   - Behaviors should be independently verifiable

**Output:** Behavioral anchors section in `references/domain-frameworks.md` with clear DO/DON'T lists.

**Quality check:**
- Are behaviors observable (not "be kind" but "avoid dismissive language like 'at least'")?
- Are they actionable (can you write a prompt incorporating them)?
- Do they cover the failure modes you documented?

## Phase 3: Binary Evaluation Rubric

**Goal:** Create composable binary questions that measure quality without LLM judgment ambiguity.

**Core principle:** Binary questions (YES/NO) are more reliable than Likert scales (1-5) for LLM-as-judge.

### Steps

1. **Identify evaluation dimensions**
   - Start with user experience goals (feel heard? get value? not annoyed?)
   - Map to observable categories (comprehension, connection, usefulness, fit)
   - Add safety as automatic-fail gate
   - Target: 4-6 categories

2. **Write binary questions per category**
   - Each question checks ONE specific thing
   - Phrased for presence ("Does it do X?") when possible
   - Absence-based only for safety ("Does it avoid Y?")
   - Target: 3-4 questions per category

3. **Ensure orthogonality**
   - Test: Could response pass Q1 but fail Q2 in same category?
   - If always correlated, questions are redundant → merge or remove
   - Each question should catch distinct failure mode

4. **Define scoring logic**
   - Category scores = sum of YES answers / questions in category
   - Weighted final score = sum of (category score × weight)
   - Safety = gate (any NO → automatic fail)
   - Pass threshold (typically 0.70-0.80)

5. **Write example evaluations**
   - Strong response (should score high)
   - Mediocre response (should fail)
   - Safety violation (automatic fail)
   - Include reasoning for each question

**Output:** `references/evaluation-rubric.md` with binary questions, scoring logic, and examples.

**Quality check:**
- Total questions: 15-20 (not more - judge fatigue)
- Each question is truly binary (no ambiguity)
- Orthogonal (no double-counting)
- Safety gate is explicit
- Examples demonstrate the rubric works

## Phase 4: Evaluation Prompt Design

**Goal:** Create prompt that applies rubric consistently to any conversation.

### Steps

1. **Structure the prompt**
   ```
   Role & Context → Rubric → Input Format → Output Format → Examples
   ```

2. **Role & Context**
   - You are evaluating [domain] responses
   - Goal: Apply binary rubric to assess quality
   - Output will be used to filter training data

3. **Present rubric**
   - All categories and questions
   - Definitions for any ambiguous terms
   - Safety questions first (fail-fast)

4. **Define input format**
   - User message + Assistant response
   - Optional: conversation context

5. **Require structured output**
   - JSON with question IDs mapped to true/false
   - Example: `{"CP1": true, "CP2": false, ...}`

6. **Include calibration examples**
   - 2-3 examples with expected answers
   - Show edge cases and reasoning

**Output:** `prompts/evaluation-prompt.md`

**Quality check:**
- Prompt is self-contained (doesn't require reading other docs)
- Examples cover edge cases
- Output format is machine-parseable
- Safety questions come first

## Phase 5: Generation Prompt Design

**Goal:** Create prompt that produces training data passing your rubric.

**Critical:** Generation prompt encodes all the domain knowledge you've extracted.

### Steps

1. **Structure the prompt**
   ```
   Role & Expertise → Framework Grounding → Behavioral Anchors →
   Failure Mode Guards → Few-Shot Examples → Output Format
   ```

2. **Role & Expertise**
   - Who is the model being? (therapist, lawyer, support agent)
   - What's the relationship? (coach, advisor, assistant)
   - What frameworks inform this role?

3. **Framework Grounding**
   - Reference specific techniques from domain-frameworks.md
   - When to use which technique
   - How techniques manifest in language (concrete examples)

4. **Behavioral Anchors**
   - Embed ALWAYS DO list
   - Embed NEVER DO list
   - Make each item explicit (not "be empathetic" but specific behaviors)

5. **Failure Mode Guards**
   - Address each documented failure mode
   - Provide specific counters ("If user mentions X → respond with Y pattern")
   - Include crisis detection if domain-appropriate

6. **Few-Shot Examples**
   - 3-5 exemplar conversations
   - Show diversity (different techniques, situations, tones)
   - Each example should pass your rubric
   - Annotate WHY each is good

7. **Output Format**
   - Specify conversation structure
   - Turn-taking expectations
   - Length guidelines

**Output:** `prompts/generation-prompt.md`

**Quality check:**
- Could someone unfamiliar with your domain use this prompt to generate good examples?
- Does it address all failure modes you documented?
- Are behavioral anchors embedded (not referenced)?
- Few-shot examples actually exemplify the quality you want?

## Workflow Integration

Once all 5 phases complete:

1. **Pilot generation** (100 examples)
   - Use generation prompt
   - Run through evaluation prompt
   - Check pass rate (target >70%)

2. **Manual spot-check** (10 examples)
   - You personally review
   - Do they feel right?
   - Any failure modes missed?

3. **Iterate prompts if needed**
   - Low pass rate → strengthen generation prompt
   - Wrong things passing → tighten rubric
   - Systematic failures → add failure mode guards

4. **Scale generation** (2K-5K examples)
   - Once pilot validated
   - Same prompts, larger volume
   - Automated filtering via evaluation prompt

## Common Mistakes to Avoid

| Mistake | Why It Fails | Fix |
|---------|-------------|-----|
| "Skip frameworks, just use GPT's knowledge" | GPT's implicit knowledge has gaps and biases | Explicit framework research grounds generation |
| "Write generation prompt first" | Can't hit a target you haven't defined | Evaluation criteria first, always |
| "Use 1-5 scales for evaluation" | LLM judges are inconsistent with Likert | Binary questions are more reliable |
| "100 questions in rubric for thoroughness" | Judge fatigue, noise, cost | 15-20 orthogonal questions maximum |
| "Behavioral anchors can be abstract" | "Be empathetic" ≠ actionable | Observable verbs only |
| "Skip failure mode research" | You'll rediscover all known problems | Literature search saves time |
| "Generate then evaluate to see what works" | Expensive iteration cycle | Design evaluation criteria first |

## File Organization

After completing this process, you should have:

```
project/
├── skills/
│   └── finetune-prep/
│       ├── SKILL.md (this file)
│       └── references/
│           ├── domain-frameworks.md
│           └── evaluation-rubric.md
├── reference/
│   ├── domain-frameworks.md (working version)
│   └── evaluation-rubric.md (working version)
└── prompts/
    ├── evaluation-prompt.md
    └── generation-prompt.md
```

## Next Steps (Outside This Skill)

After preparation is complete:
1. Pilot data generation (use generation prompt)
2. Evaluation and filtering (use evaluation prompt)
3. Training (use existing HF trainer skill/tools)
4. Post-training evaluation (same rubric on model outputs)

## Red Flags - Stop and Reconsider

These thoughts indicate you're skipping critical steps:

- "Domain frameworks are obvious, I'll skip research" → Research anyway, capture specifics
- "I'll write rubric and generation prompt together" → Evaluation first, always
- "Abstract qualities are fine for behavioral anchors" → Make them observable
- "10 questions is too few" → Fewer, better questions > many weak signals
- "I'll test prompts by generating and seeing what happens" → Define quality criteria first
- "This domain is too subjective for binary questions" → Decompose into observable components

**All of these mean: Follow the process. Quality compounds.**

## Key Principles

1. **Evaluation before generation** - Define good before making examples
2. **Explicit over implicit** - LLM knowledge is incomplete, make it explicit
3. **Observable over abstract** - "Validate feelings" → "Acknowledge emotion before problem-solving"
4. **Binary over scaled** - YES/NO > 1-5 for LLM judges
5. **Orthogonal over redundant** - 15 distinct signals > 30 correlated ones
6. **Failure-informed over theory-only** - Research what goes wrong in practice

## Time Investment

**Expected time for full preparation:**
- Phase 1 (Frameworks): 2-3 hours (research + synthesis)
- Phase 2 (Anchors): 1 hour (extraction + refinement)
- Phase 3 (Rubric): 2-3 hours (design + examples)
- Phase 4 (Eval prompt): 1 hour (writing + testing)
- Phase 5 (Gen prompt): 2 hours (encoding knowledge + examples)

**Total: 8-10 hours of preparation**

This feels like overhead but saves 20-40 hours of iteration on bad training data.

## Success Criteria

You've completed this process successfully when:

- [ ] Domain frameworks doc covers techniques with conversation examples
- [ ] Failure modes are documented with specific instances
- [ ] Behavioral anchors are observable (verb-based)
- [ ] Rubric has 15-20 orthogonal binary questions
- [ ] Rubric examples show it catches failures
- [ ] Evaluation prompt is self-contained with calibration examples
- [ ] Generation prompt embeds all domain knowledge explicitly
- [ ] Generation prompt includes failure mode guards
- [ ] Few-shot examples in generation prompt pass your rubric
- [ ] Pilot generation (100 examples) has >70% pass rate

**Deploy to training pipeline only after all checkboxes pass.**

---

*This skill encodes a methodology for fine-tuning preparation developed through critical analysis of synthetic data quality, evaluation reliability, and LLM-as-expert limitations.*
