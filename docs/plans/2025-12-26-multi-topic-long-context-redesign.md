# Multi-Topic Long-Context Therapeutic Model Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date:** 2025-12-26
**Status:** Design approved, ready for implementation

---

## Problem Statement

The original design assumed single-topic, back-and-forth therapeutic conversations (8-30 turns). Real text-based therapy works differently:

- **Multi-topic messages**: Users send paragraphs covering several concerns simultaneously
- **Response depth varies**: Quick acknowledgment for updates vs. deep exploration for complex/new issues
- **Continuous context**: Sessions are continuations of one long conversation, not independent interactions
- **History matters**: Prior discussions inform current responses; topics are revisited over time

This redesign addresses these requirements.

---

## Core Paradigm Shift

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| **Conversation model** | Single-topic, 8-30 turns, independent sessions | Multi-topic, continuous history, one long conversation |
| **User messages** | One topic per message | Multiple paragraphs covering several topics |
| **Response structure** | Address the one thing | Segmented responses, woven when appropriate |
| **Context usage** | Each conversation starts fresh | Model references and builds on full history |
| **Training data** | Many short conversations (~3M tokens) | Fewer long transcripts, varied slices (~26M tokens) |

### New Conversation Model

```
One continuous conversation:
├── Exchange 1: Topics A, B introduced
├── Exchange 2: Topic A deepens, Topic C added, B briefly updated
├── Exchange 3: Topics B, C, new topic D
├── [time gap - "next session"]
├── Exchange 4: Check-in on A, D continues, E introduced
├── ...continues indefinitely...
```

**Key properties:**
- An "exchange" = one user message (potentially multi-topic, 200-800 words) + one assistant response
- Topics have lifecycles: introduced → explored → sometimes resolved → sometimes revisited
- Some topics are quick updates, others need depth
- History accumulates; model must learn to reference it selectively

---

## Training Approach

### Long-Context Training (Option 1)

We explicitly train on examples with substantial prior history. Each training example:

```json
{
  "messages": [
    {"role": "system", "content": "...system prompt..."},
    {"role": "user", "content": "...exchange 1 user..."},
    {"role": "assistant", "content": "...exchange 1 assistant..."},
    ...
    {"role": "user", "content": "...current user message..."},
    {"role": "assistant", "content": "...response to predict..."}
  ]
}
```

The model learns: given this history + this message → produce this response.

### Supervision scope (training objective)

We **ensure every assistant message in the included history is high-quality and acceptable to supervise**. This allows standard SFT training over all assistant turns in the training JSON without loss masking.

### Leakage-safe splitting + transcript-level filtering (MVP-critical)

- **Split by transcript/persona first**, then slice within each split. This prevents evaluation leakage from overlapping histories/personas across many slices.
- **Filter at transcript level first**: assess the full transcript as one continuous conversation and only keep passing transcripts for slicing. This prevents one bad assistant turn from contaminating many derived examples.

### Example Length Distribution (Selective)

| Length | % | Examples | Avg tokens |
|--------|---|----------|------------|
| Short (0-5K) | 20% | 400 | 3K |
| Medium (5-15K) | 50% | 1000 | 10K |
| Long (15-30K) | 25% | 500 | 22K |
| Very long (30K+) | 5% | 100 | 40K |
| **Total** | 100% | **2000** | **~26M tokens** |

### Source Transcript Generation (Hybrid)

Generate transcripts of varying lengths, then slice longer ones:

- ~20 short transcripts (5-10 exchanges) → ~100 examples
- ~30 medium transcripts (15-25 exchanges) → ~400 examples
- ~50 long transcripts (30-50 exchanges) → ~1500 examples (multiple slices each)

This ensures the model sees:
- Brand new conversations (no history to use)
- Building relationships (growing history)
- Deep established relationships (rich history to reference)

---

## Human Flaw Taxonomy

For realistic user simulation, we systematically inject human messiness.

### Communication Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Burying the lede** | Real issue mentioned last or casually | "Work's fine, sleep is okay... oh and I had a panic attack Thursday" |
| **Rambling/tangential** | Drifts between topics without clear structure | Starts with work stress, ends up talking about childhood |
| **Vague/underspecified** | Doesn't give enough context | "Things have just been hard lately" |
| **Contradicting self** | Says opposite things across exchanges | "I don't care what she thinks" → later "Her opinion devastates me" |
| **Intellectualizing** | Analyzes feelings instead of feeling them | "I suppose theoretically I might feel some anxiety" |

### Resistance Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Yes-but** | Rejects suggestions while appearing to engage | "That's a good idea but it won't work because..." |
| **Deflecting with humor** | Uses jokes to avoid depth | "Haha anyway enough about my existential crisis" |
| **Minimizing** | Downplays severity | "It's not that bad, other people have real problems" |
| **Testing boundaries** | Pushes to see how assistant responds | Asks for diagnosis, medication advice |
| **Seeking reassurance** | Wants to be told it's fine vs. exploring | "I'm not being unreasonable, right?" |

### Memory/Consistency Issues

| Pattern | Description | Example |
|---------|-------------|---------|
| **Forgetting prior insight** | Returns to old pattern despite earlier realization | Had breakthrough about boundaries, now ignores it |
| **Rewriting history** | Remembers past events differently | "I never said I was angry at her" (they did) |
| **Mood-dependent recall** | Current mood colors memory of past | When down: "Nothing has ever worked" |

### Emotional Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Catastrophizing** | Everything is worst-case | "If I fail this, my career is over" |
| **Black-and-white thinking** | No nuance | "She either loves me completely or hates me" |
| **Emotional flooding** | Overwhelm, can't process | Long message that spirals without clear ask |
| **Flat affect** | Describes heavy things without emotion | "My dad died. Anyway, about work..." |

---

## Updated Assessment Rubric

### Safety Gate (Auto-reject on failure)

| ID | Criterion | Description |
|----|-----------|-------------|
| **CQ8** | Avoids harmful patterns | No harmful coping endorsement, no diagnoses, no guarantees |
| **CQ9** | Handles crisis signals | Acknowledges severity when present, doesn't minimize |

Any failure on CQ8 or CQ9 = automatic rejection, regardless of score.

### Weighted Criteria

```python
weights = {
    "comprehension": 0.15,    # CQ1, CQ2
    "connection": 0.20,       # CQ3, CQ6
    "naturalness": 0.15,      # CP2, CP4, CP5
    "multi_topic": 0.30,      # MT1, MT2, MT3, MT6  ← NEW, highest weight
    "context_use": 0.20,      # MT4, MT5  ← NEW
}

pass_threshold = 0.80
```

### Criteria Definitions

**Comprehension (0.15):**
| ID | Criterion | Checks For |
|----|-----------|------------|
| CQ1 | Accurate understanding | Understanding holds across all topics in message |
| CQ2 | Appropriate handling of ambiguity | Clarifies when unclear, doesn't over-assume |

**Connection (0.20):**
| ID | Criterion | Checks For |
|----|-----------|------------|
| CQ3 | Emotional attunement | Catches and validates emotions for each topic appropriately |
| CQ6 | Empowers user | Returns agency, frames advice optionally |

**Naturalness (0.15):**
| ID | Criterion | Checks For |
|----|-----------|------------|
| CP2 | Natural and warm | Reads like real conversation, not robotic |
| CP4 | Avoids formulaic openers | Not template-y "AI teller" openings |
| CP5 | Avoids question endings | Doesn't end every response with a question |

**Multi-Topic (0.30):** ← NEW, highest weight
| ID | Criterion | Checks For |
|----|-----------|------------|
| MT1 | Topic coverage | All topics in user message addressed (none dropped) |
| MT2 | Appropriate depth | Quick ack for updates, deeper engagement for complex/new |
| MT3 | Priority judgment | When topics compete, reasonable focus choices |
| MT6 | Segmentation clarity | Response structure makes clear which topic being addressed |

**Context Use (0.20):** ← NEW
| ID | Criterion | Checks For |
|----|-----------|------------|
| MT4 | History utilization | References prior context when it adds value (not forced) |
| MT5 | Thread continuity | Picks up old topics correctly, doesn't treat as new |

---

## Implementation Architecture

### Claude Code CLI Backend

Use Claude Code CLI for both generation and assessment (zero marginal cost).

```python
# claude_backend.py
import subprocess
import json

def ask_claude(prompt: str, system: str | None = None) -> str:
    """Call Claude Code CLI and return response."""
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    if system:
        cmd.extend(["--system", system])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr}")

    response = json.loads(result.stdout)
    return response["result"]
```

### Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Generate User Persona                              │
│  ─────────────────────────────                              │
│  • Detailed persona (personality, attachment style)         │
│  • 4-6 initial topic seeds with varying complexity          │
│  • Flaw patterns assigned from taxonomy                     │
│  Output: persona dict                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Generate Transcript (exchange loop)                │
│  ───────────────────────────────────────────                │
│  For each exchange:                                         │
│  1. Feed full history to user simulator                     │
│  2. User sim generates multi-topic message (applies flaws)  │
│  3. Feed full history + user msg to assistant generator     │
│  4. Assistant generates response                            │
│  5. Append exchange to history                              │
│  6. Repeat for target length                                │
│                                                             │
│  Between "sessions": note time gap in prompt                │
│  Output: full transcript                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Slice into Training Examples                       │
│  ────────────────────────────────────                       │
│  From full transcript, create examples with varying history │
│  • Exchange 3: history = exchanges 1-2 (~2K tokens)         │
│  • Exchange 10: history = exchanges 1-9 (~10K tokens)       │
│  • Exchange 25: history = exchanges 1-24 (~30K tokens)      │
│  Output: training_examples.jsonl                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Assess Examples                                    │
│  ───────────────────────                                    │
│  For each example:                                          │
│  • Run through updated rubric (safety gate + weighted)      │
│  • Filter at 0.80 threshold                                 │
│  Output: filtered_training_data.jsonl                       │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

**New/Rewritten:**
| File | Purpose |
|------|---------|
| `claude_backend.py` | Thin wrapper for Claude CLI calls |
| `transcript_generator.py` | Main orchestrator — personas, generation loop, assembly |
| `assessor.py` | Revised criteria, Claude CLI backend |
| `config/flaw-taxonomy.yaml` | Human patterns taxonomy |
| `config/prompts/user_sim.md` | User simulator prompt template |
| `config/prompts/assistant.md` | Assistant prompt template |
| `config/prompts/assessor.md` | Assessment prompt template |
| `SPEC.md` | Complete rewrite for new paradigm |
| `reference/assessment-rubric.md` | Updated rubric |

**Keep (modified):**
- Base model: Gemma 3 12B (128K context)
- Project structure
- Training approach: SFT with QLoRA

**Remove/Archive:**
- Single-topic conversation assumptions
- Turn-based taxonomy
- OpenAI API dependency (for generation/assessment)

---

## Implementation Plan

### Phase 1: Foundation

- [ ] Create `claude_backend.py` — CLI wrapper with error handling
- [ ] Create `config/flaw-taxonomy.yaml` — Full taxonomy from design
- [ ] Update `SPEC.md` — Codify new paradigm
- [ ] Update `reference/assessment-rubric.md` — New criteria

### Phase 2: Generation

- [ ] Create prompt templates in `config/prompts/`
- [ ] Create `transcript_generator.py` — Persona generation + exchange loop
- [ ] Test: Generate 1 short transcript end-to-end

### Phase 3: Assessment

- [ ] Rewrite `assessor.py` — New criteria, Claude backend
- [ ] Test: Assess the generated transcript (full transcript as one conversation)
- [ ] Validate rubric catches expected issues (especially MT1/MT6 and MT4/MT5)

### Minimum segmentation standard (for reliable MT judging)

To make MT judging reliable without extra topic-tracking infrastructure, every assistant response to a multi-topic user message MUST:

- **By default, start directly with the first topic section**.
- **Optional acknowledgment opener**: at most **0–1 grounded sentence** *only if it adds value*. These should be **uncommon** (aim well under ~25% of responses), since repeated validation openers are penalized by CP4. Avoid stock “That sounds hard” openers.
- **Use explicit per-topic sections** (2–4) with short labels in the user’s language
- Provide **2–6 sentences** per topic section (reflect specifics when needed, then one helpful move: clarify, normalize, reframe, offer an option, or propose a small next step)
- Optionally add **one “Woven connection” line** only when two topics clearly interact

This is intentionally minimal: it anchors MT1/MT6 for the judge while avoiding excessive/formulaic validation (see CP4).

### Phase 4: Pilot

- [ ] Generate 3 transcripts (1 short, 1 medium, 1 long)
- [ ] Assess each full transcript, review results
- [ ] Keep only passing transcripts for slicing
- [ ] Iterate on prompts/rubric as needed (MVP: prompts first)

### Phase 5: Scale

- [ ] Generate full transcript set (~100 transcripts)
- [ ] Assess full transcripts; keep only passing transcripts
- [ ] Create train/eval split (90/10) **by transcript/persona**
- [ ] Slice within each split into ~2K training examples

### Phase 6: Training

- [ ] Upload filtered data
- [ ] Run QLoRA fine-tuning on Gemma 3 12B
- [ ] Evaluate against base model

---

## Success Criteria

1. **Personal use test**: Would you actually use this for self-care?
2. **Multi-topic handling**: Model addresses all topics appropriately
3. **Context utilization**: Model references history when relevant
4. **Realistic feel**: Conversations don't feel synthetic or robotic
5. **Safety**: Model handles boundaries and crisis signals correctly

---

## Resolved Questions

| Question | Resolution |
|----------|------------|
| **Token limit per exchange** | Cap at ~800 tokens user, ~600 tokens assistant. Ensures variety, prevents runaway generation. |
| **Time gap simulation** | No explicit separators. Time between messages is implicit/unknown — more realistic. The user just picks up where they left off. |
| **Topic resolution** | Don't explicitly signal. Let topics naturally fade or get revisited based on user behavior. More realistic than formal "resolved" states. |
| **Segmentation format** | Plain text labels with bold: `**Work stress:** ...`. Simpler than markdown headers, trains on common format. |

---

*Design approved 2025-12-26. Ready for implementation.*
