# SOP 2: Synthetic Data Generation and Assessment

> **Lessons learned from the Therapeutic Coaching Fine-tuning Project**

This SOP documents the core pipeline: generating synthetic training data and assessing its quality. This is the most critical phase - garbage in, garbage out.

---

## Overview

The generation-assessment loop is iterative, not linear:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Generate │───▶│ Assess   │───▶│ Analyze  │          │
│  └──────────┘    └──────────┘    └────┬─────┘          │
│       ▲                               │                │
│       │         ┌──────────┐          │                │
│       └─────────│ Improve  │◀─────────┘                │
│                 │ Prompts  │                           │
│                 └──────────┘                           │
│                                                        │
│  Exit when: pass rate stabilizes at target             │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** Expect 3-5 prompt iteration cycles before achieving stable pass rates. Do NOT scale volume until pass rates stabilize.

---

## Part 1: Multi-Turn Conversation Generation

### Lesson 1: Use Two-Agent Simulation

#### The Problem

Single-turn generation produces unrealistic conversations. Real therapeutic coaching involves:
- Multi-turn context building
- Topic evolution across exchanges
- User resistance and breakthroughs
- History utilization

#### The Solution

Use two LLM agents in a simulation loop:

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

        # Agent 2: Therapist/Assistant
        assistant_msg = await therapist_generator(
            system_prompt=system_prompt,
            history=exchanges,
            user_message=user_msg
        )

        exchanges.append({"user": user_msg, "assistant": assistant_msg})

    return exchanges
```

#### Lessons Learned

1. **Full history must be passed to both agents.** Early experiments truncated history for efficiency. This broke context continuity and caused MT4/MT5 (history utilization) failures.

2. **Turn guidance prevents meandering.** Without explicit guidance, conversations drift aimlessly:
   ```python
   TURN_TEMPLATES = {
       "early": ["Share more context", "Express specific emotion", "Ask direct question"],
       "middle": ["Go deeper into feelings", "Connect to past experience", "Show ambivalence"],
       "late": ["Reflect on discussion", "Identify next step", "Express what's different"],
   }
   ```

3. **User simulator must NOT be too cooperative.** Our biggest lesson:
   ```python
   # Distribution of user response types
   RESPONSE_TYPES = {
       "ignore_question_talk_about_own_thing": 0.30,
       "answer_tangentially_then_pivot": 0.30,
       "push_back": 0.20,
       "actually_engage": 0.20,
   }
   ```
   Without this, conversations feel like scripted Q&A, not real therapy.

---

### Lesson 2: Persona Generation Matters

#### The Problem

Generic personas produce homogeneous conversations. Every "anxious person seeking help" sounds the same.

#### The Solution

Generate rich, specific personas with multiple dimensions:

```python
PERSONA_TEMPLATE = {
    "name": str,
    "age_range": str,
    "personality_traits": list[str],
    "attachment_style": str,  # anxious, avoidant, secure, disorganized
    "communication_style": Literal["terse", "text-speak", "casual", "formal", "stream-of-consciousness"],
    "topic_seeds": list[dict],  # 4-6 topics with complexity ratings
    "flaw_patterns": {
        "primary": str,  # 50% chance per message
        "secondary": list[str],  # 20% chance each per message
    },
    "trajectory": Literal["volatile", "improving", "deteriorating", "stable"],
}
```

#### Lessons Learned

1. **Flaw patterns vary per message, not per conversation.** Don't make the user consistently defensive. Real people have good days and bad days:
   ```
   Primary pattern: ~50% chance it shows up per message
   Secondary patterns: ~20% chance each per message
   Some messages: NO flaw showing (clear day)
   Some messages: Multiple flaws stacking (rough day)
   ```

2. **Trajectory shapes the arc, not the communication style.** A "deteriorating" trajectory means life gets harder (new problems, compounding issues) - not that communication becomes worse.

3. **Writing style must be enforced with word limits:**
   ```
   terse: 30-80 words
   text-speak: 50-120 words
   casual: 80-180 words
   formal: 120-250 words
   stream-of-consciousness: 150-300 words
   ```
   Without hard limits, all styles converge to verbose.

---

### Lesson 3: Assistant Prompt Engineering is Half the Battle

#### The Problem

The assistant generator prompt determines training data quality. Every prompt deficiency becomes a model deficiency.

#### The Solution

Our assistant prompt evolved through 15+ iterations. Key sections:

**1. Length Matching (our #1 failure mode):**
```markdown
TARGET: 1.0-1.5x user's word count. HARD LIMIT: 2x.

| User writes | Your target   | Never exceed |
|-------------|---------------|--------------|
| 30 words    | 30-45 words   | 60 words     |
| 100 words   | 100-150 words | 200 words    |
| 200 words   | 200-300 words | 400 words    |

Transcripts FAIL when avg ratio >2x OR >50% turns exceed 2x.
```

**2. No Mind-Reading (critical for CQ2):**
```markdown
NEVER assert psychological dynamics as fact:
- BAD: "You're not afraid of failing. You're afraid of mattering."
- BAD: "You weren't helping them—you were protecting yourself."

GOOD: Use tentative language AND end with a check:
- "I wonder if there's something deeper here—does that resonate?"
- "It seems like maybe part of you was protecting something. What do you think?"
```

**3. Question Discipline:**
```markdown
- Ask at most ONE question total in most responses
- For multi-topic: TWO questions max, addressing different topics
- DON'T end every section with a question (feels like interrogation)
```

**4. Proactive Follow-up (critical for MT7):**
```markdown
In the NEXT response, YOU MUST proactively ask about prior experiments.
Don't wait for user to mention it—YOU bring it up FIRST.

GOOD: "Before we dig into today—did you try the breathing thing?"
BAD: User: "I tried the grounding" -> You: "That's great! How did it go?"
     (They brought it up, not you—passive, not proactive)
```

#### Lessons Learned

1. **Calibration examples in the prompt work.** Include 2-3 PASS/FAIL examples for each major criterion directly in the generation prompt. The generator learns from examples, not just instructions.

2. **Anti-patterns must be explicit.** We added a "WHAT TO AVOID" section listing specific phrases:
   ```
   - "That sounds really hard" (hollow validation)
   - "That's profoundly..." (therapy voice)
   - "You're absolutely right" (Claude-ism)
   - "Let's unpack that" (therapy jargon)
   ```

3. **Unicode characters cause problems.** Add explicit instruction: "Stick to ASCII only (straight quotes, no curly quotes or special dashes)."

---

### Lesson 4: Async Text Format Changes Everything

#### The Problem

Live chat therapy and async text therapy are fundamentally different:
- Live chat: Rapid back-and-forth, short messages
- Async text: Days between exchanges, longer messages, life updates

#### The Solution

Our format explicitly models async:
```markdown
ASYNC TEXT THERAPY FORMAT:
- Each exchange represents a NEW DAY (not live chat)
- Users report developments: things that happened, what they tried
- Reference time naturally: "so yesterday...", "since we talked..."
- You might check in: "How did that deadline end up going?"
```

#### Lessons Learned

1. **Time passing enables realistic topic evolution.** Between exchanges, situations change, new problems emerge, old problems resolve.

2. **Follow-up becomes natural.** In async, asking "Did you try X?" makes sense because days have passed. In live chat it would be awkward.

3. **Message length expectations differ.** Async messages are naturally longer than live chat.

---

## Part 2: Assessment Pipeline

### Lesson 5: Multi-Backend Assessment Reveals Hidden Issues

#### The Problem

A single LLM assessor has blind spots. In our project:
- Claude (Sonnet 4) gave transcript 1000 a perfect 1.0
- Gemini 2.5 Flash caught 4 criterion failures
- GPT-4o caught 3 criterion failures

**20-30% of data that passes one assessor fails another.** Training on these examples teaches bad patterns.

#### The Solution

Run comparative assessment with multiple backends:

```python
BACKENDS = ["claude", "gemini", "openai"]

async def assess_with_multiple_backends(transcript):
    results = {}
    for backend in BACKENDS:
        results[backend] = await assess_transcript(transcript, backend=backend)

    # Flag disagreements
    scores = [r.score for r in results.values()]
    if max(scores) - min(scores) > 0.15:
        return {"status": "DISAGREEMENT", "results": results}

    # Use strictest assessment
    return min(results.values(), key=lambda r: r.score)
```

#### Lessons Learned

1. **Use the strictest assessment for training data.** If any backend fails a transcript, investigate. False negatives are better than false positives for training data.

2. **Disagreements reveal calibration gaps.** When backends disagree on a criterion, add calibration examples for that criterion.

3. **Backend choice matters for specific criteria:**
   - Claude: Often too lenient on CQ2 (mind-reading), CP4 (formulaic patterns)
   - Gemini: Better at catching structural rigidity
   - GPT-4: Good at catching clinical labels (CQ8)

---

### Lesson 6: Structured Output is Essential

#### The Problem

Free-form assessment output is unreliable. LLMs add explanatory text, change formats, or miss criteria.

#### The Solution

Use structured output (JSON schema) for assessment:

```python
class AssessmentResult(BaseModel):
    criteria: dict[str, CriterionResult]

class CriterionResult(BaseModel):
    answer: Literal["YES", "NO", "NA", "ERROR"]
    reasoning: str  # Required explanation
```

#### Lessons Learned

1. **Claude CLI has native JSON schema support.** Use `--json-schema` flag instead of prompt hacking:
   ```bash
   claude -p "$prompt" --json-schema "$schema" --output-format json
   ```

2. **Require reasoning for every criterion.** This catches assessor confusion and enables debugging.

3. **ERROR is a valid answer.** If the assessor can't evaluate a criterion, it should say so rather than guessing.

---

### Lesson 7: Transcript-Level Quality Analysis

#### The Problem

Pass rates tell you IF data is good. They don't tell you WHY data is bad or what patterns make it unsuitable for fine-tuning.

#### The Solution

Run quantitative pattern analysis on the full dataset:

**Structural Rigidity:**
```python
# Check for formulaic structure
bold_headers = []
for transcript in transcripts:
    for exchange in transcript["exchanges"]:
        bold_count = exchange["assistant"].count("**") // 2
        bold_headers.append(bold_count)

# Red flag: 100% same structure, avg headers > 3
```

**Phrase Repetition (Model "Tells"):**
```python
SIGNATURE_PHRASES = [
    "that's not nothing", "i want to", "that makes sense",
    "that's actually", "that's real", "that's growth"
]

# Red flag: Any phrase in >50% of responses
```

**Response Length Ratio:**
```python
ratios = []
for exchange in all_exchanges:
    ratio = len(exchange["assistant"]) / max(len(exchange["user"]), 1)
    ratios.append(ratio)

# Red flag: Avg ratio > 2x, or >50% of turns exceed 2x
```

#### Lessons Learned

1. **Rubric assessment doesn't catch repetition.** A response can pass all 17 criteria while using "that's not nothing" in 60% of responses. This creates model "tells."

2. **Run quantitative checks on FULL dataset, not samples.** Patterns emerge at scale that are invisible in spot-checks.

3. **Track praise distribution across conversation arc.** If late-conversation praise is 2x early-conversation praise, the model learns artificial positivity escalation.

---

### Lesson 8: Fixup vs. Rejection Trade-offs

#### The Problem

Some transcripts fail for fixable reasons (assertive language, clinical labels). Rejecting them wastes data. But fixes can break conversation continuity.

#### The Solution

Implement entailment-preserving fixup:

```python
FIXUP_PROMPT = """
Rewrite this therapist response to fix the identified issue.

CONSTRAINTS:
1. Convert assertions to tentative hypotheses
2. Remove clinical/diagnostic labels entirely
3. Keep the therapeutic insight - just soften delivery
4. CRITICAL: Your rewrite MUST naturally lead to the user's next message

ORIGINAL RESPONSE:
{problematic_response}

USER'S NEXT MESSAGE (must still make sense after your rewrite):
{next_user_message}

If the fix would break continuity, return: UNFIXABLE
"""
```

#### Lessons Learned

1. **Entailment constraint is non-negotiable.** If the user's next message no longer makes sense after the fix, the fix broke the conversation. Better to truncate.

2. **Re-assess after fixup.** The fix might introduce new problems. Always re-run assessment.

3. **Some failures are unfixable.** Safety gate failures (CQ8, CQ9) often require truncating the transcript at that point rather than fixing.

4. **Track fixup statistics.** If >30% of transcripts need fixup, the generation prompt needs revision.

---

## Part 3: Scaling and Filtering

### Lesson 9: Pilot Before Scaling

#### The Problem

Generating 1000+ transcripts at 15% pass rate wastes massive compute and produces insufficient training data.

#### The Solution

Run a pilot of 50-100 transcripts before scaling:

| Pilot Pass Rate | Action |
|-----------------|--------|
| >= 50% | Proceed to scale |
| 40-50% | Minor prompt iteration, re-pilot |
| 25-40% | Major prompt revision needed |
| < 25% | Fundamental issue - revisit rubric or taxonomy |

#### Lessons Learned

1. **Do not skip the pilot.** We wasted a weekend generating 500 transcripts at 35% pass rate. Could have caught issues in a 50-transcript pilot.

2. **Pilot with diverse personas.** Don't just test easy cases. Include hard personas, edge cases, and different communication styles.

3. **Analyze pilot failures by criterion.** If 80% of failures are on two criteria, focus prompt revision there.

---

### Lesson 10: Slicing Strategy for Long-Context Training

#### The Problem

A 50-turn conversation is one "transcript" but can produce multiple training examples by slicing at different points.

#### The Solution

Random slice points with bounds:

```python
MIN_CONTEXT = 3    # First slice at exchange 3 minimum
MIN_GAP = 2        # At least 2 exchanges between slices
MAX_GAP = 5        # At most 5 exchanges between slices

def get_slice_points(total_turns, transcript_id):
    # Seed by transcript ID for reproducibility
    seed = hash(transcript_id)
    rng = random.Random(seed)

    points = []
    current = MIN_CONTEXT

    while current <= total_turns:
        points.append(current)
        gap = rng.randint(MIN_GAP, MAX_GAP)
        current += gap

    # Always include final turn
    if points[-1] != total_turns:
        points.append(total_turns)

    return points
```

#### Lessons Learned

1. **Random beats dense-at-end.** Dense-at-end slicing biases toward late-conversation dynamics. Random provides uniform coverage.

2. **Seeding ensures reproducibility.** Same transcript always produces same slices.

3. **Token validation per slice.** Long transcripts can exceed context limits. Validate and truncate:
   ```python
   MAX_TOKENS = 120_000  # Buffer for 128K context window
   ```

4. **Expected yield:** ~8-10 slices per 50-turn transcript.

---

## Pipeline Checklist

Before starting generation:
- [ ] Pilot complete with >= 50% pass rate
- [ ] Generation prompts stable (no changes in last 2 pilots)
- [ ] Assessment rubric calibrated (backend agreement > 80%)

During generation:
- [ ] Checkpointing enabled (resume on failure)
- [ ] Progress tracking (transcripts generated, assessed, passed)
- [ ] Error logging for analysis

After generation:
- [ ] Quantitative pattern analysis (structure, phrases, lengths)
- [ ] Slicing with token validation
- [ ] Final pass rate and criterion breakdown

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Single assessor backend | Blind spots in assessment | Multi-backend comparison |
| Scaling before stable pass rate | Wasted compute, insufficient data | Pilot-iterate-scale |
| Free-form assessment output | Unreliable parsing | Structured JSON schema |
| Ignoring phrase repetition | Model develops "tells" | Quantitative pattern analysis |
| Fixing without entailment check | Broken conversation continuity | Entailment-preserving fixup |
| Dense-at-end slicing | Biased coverage | Random with bounds |
| Too-cooperative user simulator | Unrealistic Q&A conversations | 70% non-cooperative responses |

---

## Cost and Time Estimates

Based on our project (therapeutic coaching, 7,500 transcripts):

| Phase | Per-Transcript Cost | Time |
|-------|---------------------|------|
| Generation (25-turn) | ~$0.10-0.15 (Sonnet) | 2-3 min |
| Assessment (17 criteria) | ~$0.05-0.08 (Sonnet) | 1-2 min |
| Multi-backend assessment | ~$0.15-0.20 total | 3-5 min |
| Fixup (if needed) | ~$0.03-0.05 | 30 sec |

**Total for 1000 transcripts:** ~$150-250, 50-80 hours (with parallelization)

**Use Claude Code CLI for zero marginal cost** if you have unlimited usage.

---

*Last updated: January 2026*
*Based on therapeutic coaching fine-tuning project with 7,500+ transcripts*
