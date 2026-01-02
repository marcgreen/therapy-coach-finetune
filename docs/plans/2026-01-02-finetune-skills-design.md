# Fine-tuning Skills Design

> Design document for reusable Claude Code skills that capture the fine-tuning pipeline for multi-turn conversation domains.

**Date:** 2026-01-02
**Status:** Approved for implementation

---

## Goal

Create generalized skills that enable starting a NEW fine-tuning project in any multi-turn conversation domain (not just therapy) and implementing the full pipeline: persona generation, rubric design, data generation, assessment, training, and evaluation.

---

## Design Decisions

### 1. Three-Phase Structure

| Phase | Skill | Purpose |
|-------|-------|---------|
| 1 | `finetune-design` | All upfront design work before generation |
| 2 | `finetune-generate` | Iterative generation loop until training data ready |
| 3 | `finetune-train` | Training, evaluation, deployment |

### 2. Skills as Workflow + Reference

Each skill contains:
- `SKILL.md` — Lean workflow checklist with decision gates
- `*-guide.md` — Detailed reference material with generalized frameworks + therapy examples

### 3. SOPs Transformed, Not Referenced

Existing SOPs (`docs/sop/01-*.md`, `02-*.md`, `03-*.md`) are transformed into skill reference guides:
- Generalized frameworks extracted
- Therapy-specific content becomes "Example:" illustrations
- SOPs remain as historical artifacts but not primary reference

### 4. Key Content Additions

Missing from current SOPs:
- Base model selection (context window, quantization, deployment target)
- Token economics (16K limit for cost-effective training)
- Base model evaluation decision point (>70% pass → reconsider finetuning)

---

## Skill Structure

```
.claude/skills/
├── finetune-design/
│   ├── SKILL.md                    # Workflow: 7 steps with gates
│   ├── model-selection-guide.md    # Base model + token economics
│   ├── taxonomy-guide.md           # Input taxonomy framework
│   ├── rubric-guide.md             # Binary criteria + calibration
│   └── persona-guide.md            # Diversity dimensions + flaws
│
├── finetune-generate/
│   ├── SKILL.md                    # The iterative loop
│   ├── generation-guide.md         # Two-agent sim, prompts
│   └── assessment-guide.md         # Multi-backend, patterns, audit
│
└── finetune-train/
    ├── SKILL.md                    # Training + evaluation workflow
    └── training-guide.md           # HF Jobs, MLX, GGUF, evaluation
```

---

## Phase 1: finetune-design

### SKILL.md Structure

```markdown
# Fine-tune Design

## Purpose
Design all artifacts needed before generating training data.

## Inputs
- Domain to fine-tune for
- Deployment constraints (hardware, offline requirement, budget)

## Workflow

### Step 1: Base Model Selection
Select target model based on:
- Context window (affects max conversation length)
- Quantization support (GGUF, MLX)
- Base capability in domain
- Training cost and method support

**Gate:** Model chosen with documented tradeoffs
**Reference:** model-selection-guide.md

### Step 2: Token Economics
Determine training constraints:
- 16K tokens/example = cost-effective threshold
- Plan conversation length accordingly
- Calculate expected training cost

**Gate:** Max transcript length defined
**Reference:** model-selection-guide.md#token-economics

### Step 3: Input Taxonomy
Define input distribution:
- Topics with subtopics and weights
- Communication styles
- Difficulty levels
- Edge cases

**Gate:** Weighted taxonomy with cross-product dimensions
**Reference:** taxonomy-guide.md

### Step 4: Evaluation Rubric
Design quality criteria:
- Binary (YES/NO/NA) criteria grouped by category
- Weighted categories
- Safety gates (auto-reject)
- NA-valid vs NA-invalid specification
- 3-8 calibration examples per criterion

**Gate:** Rubric with calibration examples
**Reference:** rubric-guide.md

### Step 5: Persona Template
Design user diversity:
- Diversity dimensions for your domain
- Flaw/behavior patterns
- Distribution targets

**Gate:** Persona template with distributions
**Reference:** persona-guide.md

### Step 6: Prompts
Create generation prompts:
- User simulator prompt
- Assistant prompt (with anti-patterns)
- System prompt for training data

**Gate:** All three prompts drafted
**Reference:** generation-guide.md (in finetune-generate)

### Step 7: Base Model Evaluation
Evaluate before committing to finetune:
- Run base model on 10-20 rubric scenarios
- Calculate pass rate

**Decision Gate:**
- >70% pass rate → Reconsider if finetuning needed
- 50-70% pass rate → Proceed, moderate improvement expected
- <50% pass rate → Proceed, significant improvement needed

## Outputs
- [ ] model-choice.md (model, rationale, constraints)
- [ ] config/input-taxonomy.yaml
- [ ] config/rubric.yaml (with calibration examples)
- [ ] config/persona-template.yaml
- [ ] config/prompts/user_sim.md
- [ ] config/prompts/assistant.md
- [ ] config/system-prompt.md
- [ ] base-model-eval-results.md

## Done When
✓ All outputs created
✓ Base model evaluated
✓ Decision to proceed documented
```

### Reference Guides

#### model-selection-guide.md
- Context window considerations
- Quantization formats (GGUF, MLX, QAT)
- Training method support (LoRA, QLoRA, full)
- Cost estimation framework
- **Token Economics section:**
  - 16K threshold explanation
  - Cost scaling by token count
  - Conversation length planning
- **Therapy Example:** Gemma 3 12B selection rationale

#### taxonomy-guide.md
- Multi-dimensional taxonomy framework
- The 5 dimensions: WHAT/HOW/WHO/RELATIONSHIP/PRESENTATION
- Weighting strategies
- Edge case coverage (15% target)
- **Therapy Example:** Topics, styles, difficulty, flaws

#### rubric-guide.md
- Binary criteria structure (YES/NO/NA)
- Category grouping and weighting
- Safety gates concept
- NA-valid vs NA-invalid
- Calibration examples methodology
- **Therapy Example:** 17 criteria, calibration examples

#### persona-guide.md
- Diversity dimensions framework
- Flaw/behavior patterns taxonomy
- Trajectory modeling
- Distribution targets
- Flaw probability per message (not per conversation)
- **Therapy Example:** Attachment styles, communication styles, flaw patterns

---

## Phase 2: finetune-generate

### SKILL.md Structure

```markdown
# Fine-tune Generate

## Purpose
Iteratively generate and filter training data until quality stabilizes.

## Inputs
All outputs from finetune-design:
- Model choice and constraints
- Taxonomy, rubric, persona template
- Prompts (user-sim, assistant, system)

## Workflow

### The Loop (Critical: Small Batches)

**Generate 5 transcripts at a time, not 50-100.** This enables rapid iteration on BOTH generation AND assessment.

```
┌─────────────────────────────────────────────────────────┐
│  TIGHT LOOP (5 transcripts per iteration)               │
│                                                         │
│  Generate 5 → Assess → Analyze failures                 │
│                             │                           │
│              ┌──────────────┴──────────────┐            │
│              ▼                             ▼            │
│        Generation issues?           Assessment issues?  │
│              │                             │            │
│              ▼                             ▼            │
│        Fix gen prompts              Fix assessor:       │
│        (user-sim, assistant)        - Criteria wording  │
│                                     - Calibration ex.   │
│                                     - Add new criteria  │
│              │                             │            │
│              └──────────────┬──────────────┘            │
│                             ▼                           │
│                         Repeat                          │
│                                                         │
│  Exit when: ≥70% pass rate stable across 2-3 batches    │
└─────────────────────────────────────────────────────────┘
```

**Why 5 transcripts?**
- Small enough for human to actually READ each one carefully
- Fast feedback (minutes, not hours)
- See patterns without wasting compute
- Iterate on assessor while criteria are fresh in mind
- Catch rubric blind spots early

**Human-in-the-loop is essential (not optional):**

The human reviews BOTH the transcripts AND the assessment results:

| Human reviews... | Looking for... |
|------------------|----------------|
| Transcripts | Quality issues the rubric might miss |
| Assessment results | False positives (passed but shouldn't have) |
| Assessment results | False negatives (failed but seems fine) |
| Both together | Gaps in what the rubric even checks |

**Without human review:**
- You're optimizing against a potentially broken metric
- False positives silently corrupt training data
- False negatives waste generation effort
- Rubric blind spots never get discovered

**Dual iteration is essential:**
- Generation prompts evolve based on what fails
- Assessor evolves based on what human notices is wrong
- Calibration examples accumulate from human-identified edge cases
- New criteria emerge from failure modes the rubric missed

### Step 1: Tight Iteration Loop

For each batch of 5 transcripts:

1. **Generate** 5 transcripts
2. **Assess** with rubric (all backends)
3. **Human reviews** both transcripts and assessments
   - Read each transcript: Is this actually good?
   - Read each assessment: Did the rubric catch what matters?
   - Note: false positives, false negatives, missing criteria
4. **Iterate** based on human judgment:
   - Fix generation prompts (if transcript quality issues)
   - Fix assessor prompt/criteria (if assessment issues)
   - Add calibration examples (if edge cases found)
5. **Repeat** until quality stabilizes

**Gate (before scaling):**
- ≥70% pass rate AND human satisfied → Proceed to scale
- 50-70% OR human sees issues → Continue iterating
- <50% → Major revision needed (prompts or rubric)

### Step 2: Assessment (During Scale)
Run rubric assessment on all transcripts.
- Use multiple backends (Claude, Gemini, GPT-4)
- Take strictest result
- Track disagreements for calibration

**Reference:** assessment-guide.md

### Step 3: Audit
Run quantitative pattern analysis:
- Phrase repetition (model "tells")
- Structural rigidity (formulaic patterns)
- Response length ratios
- Praise distribution

**Gate:** No phrase in >50% of responses
**Reference:** assessment-guide.md#audit-patterns

### Step 4: Analyze
For each failing transcript:
- Which criteria failed?
- What patterns cause failures?
- Are failures fixable or systemic?

### Step 5: Iterate
Based on analysis:
- Update prompts (user-sim, assistant)
- Refine rubric criteria or calibration
- Add structural constraints if needed

**Iterate until:** Pass rate stabilizes at target (typically 50-70%)

### Step 6: Scale
Once stable:
- Generate target volume
- Continue assessment + audit
- Apply fixup where viable (entailment-preserving)
- Reject unfixable transcripts

### Step 7: Slice
Create training examples from transcripts:
- Random slice points (seeded by transcript ID)
- Validate token limits (<16K target)
- Expected yield: ~8-10 slices per 50-turn transcript

**Reference:** assessment-guide.md#slicing-strategy

## Outputs
- [ ] training_data.jsonl
- [ ] generation_stats.md (pass rates, criterion breakdown)
- [ ] prompt_versions/ (history of prompt iterations)

## Done When
✓ Target training example count reached
✓ Pass rate stable across last 2-3 batches
✓ Audit patterns within thresholds
✓ training_data.jsonl validated
```

### Reference Guides

#### generation-guide.md
- Two-agent simulation architecture
- Full history passing (critical for context)
- Turn guidance to prevent meandering
- User simulator design:
  - Non-cooperative response distribution (70% don't just answer)
  - Flaw pattern injection
  - Writing style enforcement with word limits
- Assistant prompt engineering:
  - Length matching (1.0-1.5x user, hard limit 2x)
  - Tentative language for interpretations
  - Question discipline
  - Response ending variety
  - Anti-patterns to avoid
- **Therapy Example:** Prompt iterations, specific failures

#### assessment-guide.md
- Conversation-level assessment (not per-turn)
- Multi-backend strategy
- Structured output (JSON schema)
- Pre-computed statistics (LLMs can't count)
- Fixup strategy (entailment-preserving)
- **Assessor Iteration section (Critical):**
  - The rubric is never "done" — it evolves with the data
  - When to adjust criteria wording
  - When to add calibration examples (backend disagreement)
  - When to add new criteria (failure modes rubric missed)
  - Signs of false positives/negatives
  - Therapy evolution: 12 → 14 → 16 → 17 → 18 criteria
- **Audit Patterns section:**
  - Phrase repetition detection
  - Structural rigidity metrics
  - Response length analysis
- **Slicing Strategy section:**
  - Random with bounds
  - Token validation
  - Leakage-safe splitting
- **Infrastructure section:**
  - Checkpointing
  - Incremental writes
  - Retry with backoff
- **Therapy Example:** Backend disagreements, pattern analysis results, assessor evolution

---

## Phase 3: finetune-train

### SKILL.md Structure

```markdown
# Fine-tune Train

## Purpose
Train the model and verify improvement.

## Inputs
- training_data.jsonl from finetune-generate
- Model choice from finetune-design

## Workflow

### Step 1: Dataset Preparation
- Format data for training framework
- Push to HuggingFace Hub (or local)
- Verify format and access

### Step 2: Training Configuration
Choose training approach:
- HuggingFace Jobs (cloud GPU)
- MLX local (Apple Silicon)
- Other cloud providers

Configure:
- QLoRA parameters (r, alpha, target modules)
- Training hyperparameters (epochs, batch size, learning rate)
- max_length based on model vocab size + GPU

**Reference:** training-guide.md

### Step 3: Submit Training
- Pre-create output repos (HF Jobs permission issue)
- Submit job with correct token
- Monitor progress

**Reference:** training-guide.md#hf-jobs-issues

### Step 4: GGUF Conversion
- Merge adapter with base model
- Convert to GGUF format
- Download for local inference

### Step 5: Evaluation
Generate full conversations with both models:
- 10-15 NEW personas (not in training)
- 3 conversations per persona per model
- Same user simulator for both (controlled comparison)

Assess and compare:
- Run rubric assessment on all transcripts
- Calculate mean scores, pass rates
- Statistical test (p < 0.05)

**Success criteria:**
- Improvement ≥10% with p < 0.05
- No safety regressions

### Step 6: Sanity Checks
- Perplexity on held-out set (did training work?)
- Small human eval sample (5-10 conversations)
- Capability regression test (general abilities intact?)

## Outputs
- [ ] Fine-tuned model (HuggingFace Hub)
- [ ] GGUF file (local deployment)
- [ ] evaluation_report.md

## Done When
✓ Training completed successfully
✓ GGUF converted and tested locally
✓ Evaluation shows significant improvement
✓ No safety regressions
```

### Reference Guide

#### training-guide.md
- **HuggingFace Jobs section:**
  - 9 critical issues and solutions (from SOP 3)
  - Working script template
  - Token permissions
  - Gemma vocabulary cost (262K vocab → OOM)
  - GPU selection guide
- **MLX Local section:**
  - Setup and configuration
  - mask_prompt=true
  - Adapter conversion
- **GGUF Conversion section:**
  - Merge adapter
  - llama.cpp conversion
  - Quantization options
- **Evaluation Methodology section:**
  - Full-conversation generation approach
  - Why this is rigorous for conversations
  - Statistical comparison
  - Sanity checks (perplexity, human eval, regression)
- **Therapy Example:** Training runs, evaluation results

---

## Implementation Plan

### Phase 1: Create Skill Structure
1. Create directory structure under `.claude/skills/`
2. Write three SKILL.md files (lean workflow)

### Phase 2: Transform SOPs into Reference Guides
1. Extract generalized frameworks from SOPs
2. Keep therapy examples as "Example:" sections
3. Add missing content (model selection, token economics, base model eval)

### Phase 3: Validation
1. Review skills for completeness
2. Verify all SOP lessons are captured
3. Test skill invocation

---

## Migration from Current Skills

Current skills to be replaced:
- `finetune-prep` → Absorbed into `finetune-design`
- `generating-finetuning-data` → Absorbed into `finetune-generate`
- `judging-transcript-quality` → Absorbed into `finetune-generate` (assessment-guide.md#audit-patterns)

Current SOPs become historical reference:
- `docs/sop/01-*.md` → Content extracted to design guides
- `docs/sop/02-*.md` → Content extracted to generate guides
- `docs/sop/03-*.md` → Content extracted to training-guide.md
