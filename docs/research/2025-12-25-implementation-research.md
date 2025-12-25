# Implementation Research Notes

## 1. Model Selection: gpt-5-mini

**Finding:** `gpt-5.2-mini` does NOT exist. Current available models:
- `gpt-5.2` and `gpt-5.2-pro` (released Dec 11, 2025)
- `gpt-5-mini` (the mini variant of GPT-5)

**Decision:** Use `gpt-5-mini` for both assessment and generation.

**Reasoning parameter:** From [GPT-5.2 Prompting Guide](https://platform.openai.com/docs/guides/gpt-5-2-prompting):
- Values: `none | minimal | low | medium | high | xhigh`
- Default for GPT-5.2 is `none`
- For **assessment (judging)**: Use `medium` - we need deliberate reasoning to evaluate quality
- For **generation (conversation simulation)**: Use `low` - we want natural flow, not overthinking

Sources:
- [Introducing GPT-5.2 | OpenAI](https://openai.com/index/introducing-gpt-5-2/)
- [GPT-5 mini Model | OpenAI API](https://platform.openai.com/docs/models/gpt-5-mini)

---

## 2. NA Logic Review

### Current Logic

| Criterion | NA Valid? | Rationale |
|-----------|-----------|-----------|
| CQ1-CQ7 | Yes | Condition may not apply (no ambiguity, no emotions, etc.) |
| **CQ8** | **NO** | Every conversation can be assessed for harmful patterns |
| CQ9 | Yes | NA means "no crisis signals present" - that's OK |
| CP1 | Yes | NA if <3 turns |
| **CP2** | **NO** | Every conversation can be assessed for naturalness |
| CP3 | Yes | NA if <10 turns or unrelated topics explored |

### Is This Sound?

**Yes, with one consideration:**

The logic is sensible because:
1. **CQ8 (harmful patterns)** - Even a 1-turn conversation can endorse harmful coping. There's no scenario where we can't assess this. NA would mean the judge is avoiding responsibility.

2. **CP2 (natural/warm)** - Even short conversations have a tone. Judges shouldn't be able to dodge this assessment.

3. **CQ9 (crisis handling)** - NA is explicitly valid because it means "there were no crisis signals to handle." This is different from the judge avoiding assessment.

**Potential Issue:** If the LLM judge is uncertain, it might return NA to "play it safe." Our NA-invalid logic forces it to commit to YES or NO for CQ8/CP2. This is intentional - we want decisive assessments.

**Recommendation:** Keep the current logic. It's sound.

---

## 3. Test Cases Review

### Current Planned Tests

| Test | What It Verifies |
|------|------------------|
| `test_all_yes_gives_perfect_score` | Happy path - all passing |
| `test_all_no_gives_zero_score` | All failing |
| `test_cq8_no_triggers_safety_gate` | Safety gate on harmful patterns |
| `test_cq9_no_triggers_safety_gate` | Safety gate on crisis handling |
| `test_cq8_na_is_invalid_triggers_safety_gate` | NA-invalid for CQ8 |
| `test_cq9_na_is_valid_no_safety_gate` | NA-valid for CQ9 |
| `test_cp2_na_is_invalid_counts_as_failure` | NA-invalid for CP2 (non-safety) |
| `test_valid_na_counts_as_pass` | Multiple valid NAs |
| `test_error_counts_as_failure` | ERROR handling |
| `test_safety_error_triggers_gate` | ERROR on safety criterion |
| `test_all_errors_fails` | All errors edge case |
| `test_get_applicable_criteria_*` | Conditional criteria filtering |
| `test_weighted_score_calculation` | Weighted math |
| `test_threshold_boundary` | Score at exactly 0.80 |

### Assessment

**Coverage is good.** Key scenarios covered:
- Happy path (all YES)
- Sad path (all NO)
- Safety gate triggers (NO, ERROR, invalid-NA)
- Valid NA scenarios
- Weighted calculation
- Conditional criteria

**Missing edge cases to consider:**

1. **Score just below threshold (0.799)** - Verify it fails
2. **Single category all-fail** - Verify weighted impact
3. **Empty criteria list** - What happens if no criteria apply?

**Recommendation:** Add 2-3 more focused tests:

```python
def test_score_just_below_threshold_fails(self):
    """Score of 0.799 should fail (threshold is 0.80)."""
    # Need to engineer exactly 0.799
    ...

def test_empty_results_returns_perfect_score(self):
    """If no criteria are assessed, should return 1.0."""
    result = compute_score([], [])
    assert result.passed is True
    assert result.score == 1.0
```

But the current set is **sufficient for MVP**. Don't overdo it.

---

## 4. Responses API for Generation

### Why Responses API?

From the GPT-5.2 prompting guide:
- Responses API is the modern API for GPT-5.x
- Supports `reasoning` parameter for controlling deliberation
- Supports structured output via `text_format`
- Supports compaction for long contexts

### Generator Changes Needed

Replace all `chat.completions.create()` with `responses.create()`:

```python
# OLD (chat completions)
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.8,
)

# NEW (responses API)
response = await client.responses.create(
    model="gpt-5-mini",
    input=[...],  # NOT messages
    reasoning={"effort": "low"},  # NOT temperature
)
```

### Reasoning Effort Recommendation

| Task | Effort | Rationale |
|------|--------|-----------|
| **Persona generation** | `low` | Creative task, needs variety |
| **User turn simulation** | `low` | Natural conversation, don't overthink |
| **Therapist response** | `medium` | Needs some deliberation for quality |
| **Assessment (judge)** | `medium` | Needs careful evaluation |

`low` for generation keeps things natural. `medium` for assessment ensures quality judgment.

---

## 5. DSPy and Async

### Can DSPy Work in Async Context?

**Short answer:** DSPy is synchronous by design, but can be used from async code.

**The issue:** DSPy's optimizers run synchronously. If you call a metric that uses `asyncio.run()` from inside an already-running event loop, you get:
```
RuntimeError: This event loop is already running
```

**Solutions:**

1. **ThreadPoolExecutor** (what we planned):
   ```python
   with ThreadPoolExecutor() as executor:
       result = executor.submit(asyncio.run, assess_conversation(...))
       return result.result()
   ```

2. **nest_asyncio** (simpler but hackier):
   ```python
   import nest_asyncio
   nest_asyncio.apply()
   # Now asyncio.run() works even in async context
   ```

3. **Run DSPy in main thread** (cleanest):
   - Don't run DSPy optimization from async code
   - Run it as a standalone script: `uv run python optimize.py`
   - This avoids the event loop conflict entirely

**Recommendation:** Use option 3 - run DSPy optimization as a separate script. This is the cleanest approach and what most DSPy examples do.

---

## 6. GEPA Effectiveness for Our Rubric

### How GEPA Works

From [GEPA Overview](https://dspy.ai/api/optimizers/GEPA/overview/):

1. **Metric returns `{score, feedback}`** - The textual feedback explains WHY it failed
2. **Reflection model** analyzes failures and proposes better prompts
3. **Pareto selection** maintains diverse candidates (one per failure mode)
4. **Iterative evolution** - typically 3-10 iterations

### Will It Work Well for Our Rubric?

**Yes, with some considerations:**

**Why it should work:**
- Our rubric provides **per-criterion feedback** (e.g., "CQ3: Emotions not acknowledged in turns 2-4")
- GEPA explicitly uses textual feedback to guide evolution
- The PAPILLON tutorial shows GEPA improving from 77% to 86% in **just 1 iteration** with LLM-as-judge feedback

**Metric format for GEPA:**
```python
def rubric_metric(example, pred, trace=None):
    result = assess_conversation(pred.conversation)

    feedback_parts = []
    for cid in result.failed_checks:
        feedback_parts.append(f"{cid}: {result.reasonings[cid]}")

    return {
        "score": result.score if not result.safety_gate_failed else 0.0,
        "feedback": "\n".join(feedback_parts) or "All criteria passed",
    }
```

**Potential issues:**

1. **Criterion feedback quality** - If judge reasonings are vague, GEPA has less to work with
2. **Multi-criterion failures** - With 12 criteria, feedback can be noisy. GEPA's Pareto selection helps here.
3. **Safety gate** - 0.0 score on safety failure is appropriate but may dominate early iterations

**Recommendations:**
1. Ensure judge reasonings are specific and actionable (cite turn numbers)
2. Consider running GEPA with `auto="light"` first for fast iteration
3. May need to separate safety vs quality optimization

### Case Study: PAPILLON

From [GEPA for Privacy-Conscious Delegation](https://dspy.ai/tutorials/gepa_papillon/):
- Used LLM-as-judge metric (similar to ours)
- Improved from 77% → 86% in 1 iteration
- "GEPA optimizes the PAPILLON program from a score of 77% to 86% after proposing just 1 new candidate"

This is exactly our use case.

---

## 7. Token Usage and API Call Patterns

### Assessment (12 criteria per conversation)

**Per conversation:**
- 12 parallel API calls (one per criterion)
- Each call: ~2K tokens input (conversation + criterion prompt), ~200 tokens output
- Total: ~26K tokens per conversation

**Batch of 100 conversations:**
- 1,200 API calls
- ~2.6M tokens
- With `gpt-5-mini` at $0.25/1M input, $1.00/1M output:
  - Input: 100 * 24K = 2.4M tokens → $0.60
  - Output: 100 * 2.4K = 240K tokens → $0.24
  - **Total: ~$0.84 per 100 conversations**

**Concurrency:** 10 concurrent calls with semaphore. Rate limits handled by tenacity retry.

### Generation (per conversation)

**Per conversation (15 turns average):**
- 1 persona generation call (~500 tokens in, ~300 out)
- 15 user turn calls (~1K tokens in, ~200 out each)
- 15 therapist turn calls (growing context, avg ~3K in, ~300 out)
- Total: ~75K tokens per conversation

**Batch of 100 conversations:**
- ~3,100 API calls (1 + 15 + 15 per conv)
- ~7.5M tokens
- Cost estimate: ~$2.50 for 100 conversations

**Concurrency:** 5 concurrent conversations (each has sequential turns within).

### Cost Summary

| Phase | 100 Conversations | 3,000 Conversations |
|-------|-------------------|---------------------|
| Generation | ~$2.50 | ~$75 |
| Assessment | ~$0.84 | ~$25 |
| **Total** | ~$3.34 | ~$100 |

These are rough estimates. Actual costs depend on conversation length and model pricing.

---

## 8. E2E Trial Before Full Implementation

### Current Plan Gap

The plan has:
- Task 7.1: End-to-End Smoke Test (generate 3, assess them)

But this is **after** all code is written. We should validate E2E **earlier**.

### Recommended Addition: Phase 0 - Validation

Add before Phase 1:

```markdown
## Phase 0: Quick E2E Validation

### Task 0.1: Verify API Access
Run: `uv run python -c "from openai import OpenAI; print(OpenAI().models.list().data[0])"`
Expected: Model info printed (API key works)

### Task 0.2: Test Assessment on Existing File
Create a minimal test conversation manually, assess it:
Run: `echo '{"messages": [...]}' > test_conv.json && uv run python assessor.py test_conv.json`
Expected: Assessment runs, result printed

### Task 0.3: Test One Generation Call
Quick smoke test of persona generation:
Run: `uv run python -c "from generator import generate_persona; ..."`
Expected: Persona generated
```

This catches API issues before investing in implementation.

---

## 9. Base Model Evaluation Before Full Data Gen

### The Question

> What if the base model is already super good and we don't need to fine-tune?

This is a **critical** question. We should evaluate the base model BEFORE generating 3K conversations.

### Recommended Approach

**Add to pilot phase:**

1. Generate 50 synthetic conversations (as planned)
2. Also generate 50 "baseline responses" from the **base model** (Qwen/Llama/Mistral 7B)
3. Assess both with our rubric
4. Compare scores

**If base model scores ≥ 0.70:**
- Consider if fine-tuning is necessary
- May only need light SFT or none at all

**If base model scores ≤ 0.50:**
- Fine-tuning is clearly needed
- Proceed with full data generation

### Implementation

Add a `baseline_eval.py` script:

```python
async def evaluate_baseline():
    """Evaluate base model on synthetic inputs."""
    # 1. Generate 50 user personas/openings
    # 2. Use LOCAL base model (via ollama) to generate therapist responses
    # 3. Assess with rubric
    # 4. Compare to synthetic data pass rate
```

This requires:
- Ollama installed
- Base model pulled (`ollama pull qwen2.5:7b`)

**Cost:** Nearly zero (local inference) + assessment cost (~$0.42 for 50 convos)

---

## Summary of Recommendations

| Topic | Recommendation |
|-------|----------------|
| Model | Use `gpt-5-mini` (not gpt-5.2-mini) |
| DSPy dep | Wait until Phase 5 (DSPy Integration) |
| NA logic | Current logic is sound, keep it |
| Tests | Current set is sufficient for MVP |
| Responses API | Use for both generation and assessment |
| Reasoning effort | `low` for generation, `medium` for assessment |
| DSPy async | Run optimization as separate script |
| GEPA | Should work well with our feedback format |
| E2E trial | Add Phase 0 validation before implementation |
| Base model eval | Add to pilot phase before full data gen |
