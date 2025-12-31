# E2E Pipeline Validation Plan

> **Goal:** Validate the full fine-tuning pipeline end-to-end using existing transcripts before scaling to full generation.

**Date:** 2025-12-30
**Last Updated:** 2025-12-31

---

## Current State (Updated)

### Existing Assets

| Asset | Status | Notes |
|-------|--------|-------|
| **Transcripts** | ✅ 41 total | 34 short, 3 medium, 1 long, 3 very_long |
| **Assessor** | ✅ Complete | `assessor.py` with `assess_batch()` for batch processing |
| **Assessment Results** | ⚠️ 29 assessed | Only short transcripts assessed so far |
| **Token Analysis** | ✅ Complete | `scripts/analyze_transcripts.py` with tiktoken |
| **LLM Backend** | ✅ Complete | `llm_backend.py` - Claude, OpenAI, Google |

### Transcript Token Analysis (via tiktoken)

| Category | Count | Avg Tokens | Max Tokens | Context Safe? |
|----------|-------|------------|------------|---------------|
| Short | 34 | ~26K | 37K | ✅ All safe |
| Medium | 3 | ~32K | 47K | ✅ All safe |
| Long | 1 | 91K | 91K | ✅ Safe |
| Very Long | 3 | ~93K | 122K | ⚠️ 1 exceeds 120K |

**Only 1 transcript** (transcript_2000 at 122K tokens) may need truncation.

### Scripts Created

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/analyze_transcripts.py` | ✅ Complete | Token analysis with tiktoken |
| `scripts/assess_remaining.py` | ✅ Created | Assess unassessed transcripts via `assess_batch()` |

---

## What Still Needs To Be Done

### Step 1: Assess Remaining Transcripts
**Status: Script created, NOT YET RUN**

```bash
uv run python scripts/assess_remaining.py
```

This will assess 12 remaining transcripts:
- 5 short (0000-series)
- 3 medium
- 1 long
- 3 very_long

### Step 2: Gather Passing Transcripts
**Status: Not started**

Read from both checkpoint files, collect paths of passing transcripts.

### Step 3: Slice Into Training Examples
**Status: Not started**

Key decisions made:
- **Random density** (not dense-at-end) - uniform coverage of early/late conversation
- **Bounds:** min_context=3, min_gap=2, max_gap=5
- **Token counting:** tiktoken (not char estimate)
- **Truncation:** For >120K tokens, keep most recent N exchanges that fit

Expected: ~200-250 training examples from ~30 passing transcripts

### Step 4: Push to HuggingFace Hub
**Status: Not started**

### Step 5: Train with HF Jobs
**Status: Not started**

### Step 6: Convert to GGUF
**Status: Not started**

### Step 7: Evaluate
**Status: Not started**

---

## Key Design Decisions Made

### 1. Random Slicing (Not Dense-at-End)

**Original plan:** Sparse early, dense late
**New decision:** Random with min_gap=2, max_gap=5

**Rationale:** With only ~30 transcripts, we need uniform coverage of early and late conversation dynamics. Dense-at-end would bias toward late-conversation style.

### 2. Actual Token Counting

**Original plan:** ~4 chars per token estimate
**New decision:** tiktoken with cl100k_base encoding

**Rationale:** Accurate token counts prevent surprises during training.

### 3. Truncation Strategy

For transcripts exceeding 120K tokens at late slice points:
- Keep most recent N exchanges that fit
- Log truncation for visibility

### 4. Use Existing Infrastructure

**Lesson learned:** Use `assess_batch()` instead of reimplementing checkpointing/error handling.

---

## Detailed Implementation Plan

See `docs/plans/2025-12-30-e2e-finetune-pipeline.md` for full task breakdown with code.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All transcripts assessed | ~30+ passing |
| Training examples generated | ~200-300 |
| Training completes | Loss converges |
| GGUF exports | Loads in llama.cpp |
| Evaluation runs | Comparison generated |

**Note:** This is pipeline validation, not production training. We're not expecting statistically significant improvement with ~250 examples.
