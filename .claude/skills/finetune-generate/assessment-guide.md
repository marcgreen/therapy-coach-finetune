# Assessment Guide

How to assess conversation quality, iterate on the assessor, and audit for hidden issues.

---

## Conversation-Level Assessment

**Assess entire conversations, not individual turns.**

### Why Conversation-Level?

| Approach | API Calls | Cost | Quality |
|----------|-----------|------|---------|
| Per-turn, per-criterion | 17 × N turns | Very high | Misses conversation arc |
| Per-turn, batched | N turns | High | Still misses arc |
| **Conversation-level** | **17 calls** | **Low** | **Captures full context** |

Conversation-level assessment:
- 98% API cost reduction
- Better for criteria that span turns (patterns, history use)
- Captures relationship development, consistency

### Implementation

```python
async def assess_transcript(transcript, backend="claude"):
    results = {}

    for criterion_id, criterion in CRITERIA.items():
        prompt = build_assessment_prompt(
            transcript=transcript,
            criterion=criterion,
        )

        response = await call_llm(prompt, backend=backend)
        results[criterion_id] = parse_response(response)

    return results
```

**Further optimization:** Batch all criteria into a single call with structured output.

---

## Multi-Backend Assessment

**A single LLM assessor has blind spots.**

In one project, the same transcript received:
- Claude: perfect 1.0
- Gemini: 4 criterion failures
- GPT-5: 3 criterion failures

**20-30% of data that passes one assessor fails another.**

### Strategy

```python
BACKENDS = ["claude", "gemini", "openai"]

async def assess_with_multiple_backends(transcript):
    results = {}

    for backend in BACKENDS:
        results[backend] = await assess_transcript(transcript, backend)

    # Check for disagreement
    scores = [compute_score(r) for r in results.values()]
    if max(scores) - min(scores) > 0.15:
        return {"status": "DISAGREEMENT", "results": results}

    # Use strictest assessment
    return min(results.values(), key=lambda r: compute_score(r))
```

### Disagreement Resolution

When backends disagree (spread > 0.15):

| Pattern | Resolution |
|---------|------------|
| 2 pass, 1 fail | **Fail** — investigate the failure criterion |
| All different scores | Review reasoning, add calibration examples |
| Same verdict, different reasoning | Pass if verdict agrees, note for review |

**Key principle:** False positives corrupt training data. When in doubt, reject.

### Backend Characteristics

| Backend | Strengths | Weaknesses |
|---------|-----------|------------|
| Claude | Good overall | Often too lenient on own patterns |
| Gemini | Catches structural issues | Sometimes inconsistent |
| GPT-5 | Good at clinical labels, safety | Expensive |

**Recommendation:** Use at least two backends. Take the stricter result.

---

## Structured Output

**Free-form output is unreliable.** Use JSON schema:

```python
class CriterionResult(BaseModel):
    answer: Literal["YES", "NO", "NA", "ERROR"]
    reasoning: str  # Required explanation

class AssessmentResult(BaseModel):
    criteria: dict[str, CriterionResult]
```

**With Claude CLI:**
```bash
claude -p "$prompt" --json-schema "$schema" --output-format json
```

**Why require reasoning:**
- Catches assessor confusion
- Enables debugging
- Reveals calibration issues

---

## Expert Role-Play Critique

**Role-play domain experts to stress-test design artifacts.** This applies to ALL design phases—not just rubric refinement.

Different experts see different failure modes. A domain practitioner catches different issues than a UX researcher than an AI safety researcher. Role-playing experts surfaces blind spots you'd otherwise miss.

### When to Use This Technique

| Phase | What to Critique | Why |
|-------|-----------------|-----|
| **Taxonomy design** | Topic categories, edge cases, weights | Ensure coverage, identify missing user types |
| **Persona creation** | Communication styles, flaws, trajectories | Catch unrealistic patterns, missing populations |
| **Prompt engineering** | User simulator, assistant generator prompts | Find instruction gaps, conflicting requirements |
| **Rubric design** | Criteria, calibration examples, weights | Identify failure modes rubric would miss |
| **Iteration** | Failed transcripts, passing-but-off transcripts | Diagnose systemic issues |

### Process

**Step 1: Identify relevant experts for your domain**

Claude should identify experts based on the specific domain. Consider:

- **Domain practitioners** — People who do the job the assistant is replacing/augmenting
- **Methodology creators** — Founders of frameworks used in the domain
- **Researchers** — Academics who study this domain
- **User advocates** — People who represent different user populations
- **AI/ML experts** — For evaluation methodology and safety

**Example identification prompt:**
```
I'm designing [rubric/taxonomy/prompts] for a [domain] assistant.

Identify 5-7 experts (real or fictional) whose perspectives would help stress-test this design:
- At least 2 domain practitioners with different methodological approaches
- At least 1 researcher who studies this domain critically
- At least 1 advocate for a vulnerable user population
- At least 1 AI/ML evaluation expert

For each expert, provide:
- Name and credentials/perspective
- What unique blind spots they would catch
- What they would be especially critical of
```

**Step 2: Role-play each expert's critique**

```
Role-play as [Expert Name], a [credentials/perspective].

Critically review this [artifact type] for [domain]:

[INSERT ARTIFACT]

Focus on:
- What failure modes would this miss?
- What would pass this but still be harmful/inadequate?
- What would fail this but actually be appropriate?
- What user populations does this not serve well?
- What assumptions are baked in that should be questioned?

Be constructively critical. Suggest specific improvements.
```

**Step 3: Include fictional edge users**

Beyond experts, role-play challenging users:
- A skeptical user who's been burned before
- A vulnerable user in crisis or distress
- A user from a different cultural context
- A user with non-standard communication patterns
- A user who will try to game or break the system

**Step 4: Synthesize into improvements**

After gathering critiques, identify:
- Common themes across experts
- Contradictory advice (requires judgment call)
- Quick wins vs fundamental redesigns
- New criteria, calibration examples, or edge cases to add

### Example: Therapy Domain

```
Experts identified and consulted:
- Marsha Linehan (DBT creator) → Caught missing dialectical synthesis in validation criteria
- William Miller (MI creator) → Identified that empowerment criteria missed solution origin
- Irvin Yalom (existential therapy) → Added presence-without-insight as valid outcome
- Emily Bender (computational linguist) → Reframed "AI cannot feel" to "user inference"
- Percy Liang (LLM evaluation) → Switched from confirm-first to evidence-first reasoning
```

This technique improved the rubric more than any single refinement cycle.

---

## Pre-Computed Statistics

**LLMs can't count reliably.** Pre-compute anything quantitative:

```python
def compute_length_stats(conversation):
    ratios = []
    for exchange in conversation:
        user_words = len(exchange["user"].split())
        assistant_words = len(exchange["assistant"].split())
        ratio = assistant_words / max(user_words, 1)
        ratios.append(ratio)

    return {
        "avg_ratio": sum(ratios) / len(ratios),
        "pct_over_2x": len([r for r in ratios if r > 2]) / len(ratios),
        "max_ratio": max(ratios),
    }

# Inject into prompt:
"""
Length stats (pre-computed):
- Average ratio: {avg_ratio:.2f}
- Percent over 2x: {pct_over_2x:.0%}
- Max ratio: {max_ratio:.2f}

Threshold: PASS if avg < 1.5 AND pct_over_2x < 25%
"""
```

**Applies to:** Word counts, question counts, section counts, any quantitative criterion.

---

## Assessor Iteration

**The rubric is never "done."** It evolves as you discover issues.

### When to Iterate

| Observation | Action |
|-------------|--------|
| Criterion too strict (good transcripts failing) | Loosen wording, add borderline-pass examples |
| Criterion too lenient (bad transcripts passing) | Tighten wording, add fail examples |
| Backend disagreement | Add calibration examples for that criterion |
| New failure mode discovered | Add new criterion |
| Criterion rarely triggers | Consider removing or merging |
| Pass rates high but transcripts feel "off" | Use Expert Role-Play Critique (see above) |

### How to Iterate

1. **Identify the issue** — Which criterion? What's the pattern?
2. **Adjust wording** — Make the question clearer
3. **Add calibration examples** — Show the edge cases
4. **Re-assess affected transcripts** — Verify improvement
5. **Document the change** — Track rubric evolution

### Evolution is Normal

One project's rubric evolved: 12 → 14 → 16 → 17 → 18 criteria.

New criteria emerged from:
- Multi-topic handling (MT1-MT7)
- Response length (CP2)
- Coaching continuity (MT7)

Each addition came from discovering failure modes existing criteria missed.

---

## Fixup Strategy

Some failures are fixable. Others require rejection.

### Entailment-Preserving Fixup

**Constraint:** Fixed response must naturally lead to user's next message.

```python
FIXUP_PROMPT = """
Rewrite this response to fix the identified issue.

CONSTRAINTS:
1. Fix the specific problem: {issue_type}
2. Keep the core insight/value
3. CRITICAL: Rewrite must naturally lead to user's next message

ORIGINAL RESPONSE:
{problematic_response}

USER'S NEXT MESSAGE (must still make sense after your rewrite):
{next_user_message}

If the fix would break continuity, return: UNFIXABLE
"""
```

### When to Fix vs Reject

| Failure Type | Action |
|--------------|--------|
| Assertive language | Fix — soften to tentative |
| Clinical labels | Fix — remove diagnostic terms |
| Safety gate failures | Reject or truncate |
| Structural issues | Usually reject |
| Broken continuity | Truncate at that point |

### Fixup Statistics

**If >30% of transcripts need fixup:** Generation prompts need revision.

Don't rely on fixup as the solution — fix the root cause in generation.

---

## Audit Patterns

**Rubric assessment catches individual failures. Audit catches systemic issues.**

### Phrase Repetition

Check for "model tells" — phrases used too frequently:

> **Domain-specific:** Identify phrases that appear too often in your generated data.
> The examples below are illustrative—build your own list for your domain.

```python
SIGNATURE_PHRASES = [
    "that's not nothing",
    "i want to",
    "that makes sense",
    "that's actually",
    "that's real",
    "that's growth",
]

def check_phrase_repetition(transcripts):
    phrase_counts = {p: 0 for p in SIGNATURE_PHRASES}
    total_responses = 0

    for transcript in transcripts:
        for exchange in transcript["exchanges"]:
            response = exchange["assistant"].lower()
            total_responses += 1
            for phrase in SIGNATURE_PHRASES:
                if phrase in response:
                    phrase_counts[phrase] += 1

    # Red flag: Any phrase in >50% of responses
    for phrase, count in phrase_counts.items():
        if count / total_responses > 0.50:
            print(f"WARNING: '{phrase}' appears in {count/total_responses:.0%}")
```

**Action:** Add overused phrases to anti-patterns list, regenerate.

### Structural Rigidity

Check for formulaic structure:

```python
def check_structural_rigidity(transcripts):
    header_counts = []

    for transcript in transcripts:
        for exchange in transcript["exchanges"]:
            # Count bold headers like **Topic:**
            bold_count = exchange["assistant"].count("**") // 2
            header_counts.append(bold_count)

    # Red flag: 100% same structure
    if len(set(header_counts)) == 1 and header_counts[0] > 3:
        print(f"WARNING: All responses have exactly {header_counts[0]} headers")
```

**Note:** Topic headers are NOT formulaic. They're good structure for multi-topic responses. Evaluate content variety WITHIN the structure.

### Response Length Distribution

```python
def check_length_ratios(transcripts):
    ratios = []

    for transcript in transcripts:
        for exchange in transcript["exchanges"]:
            user_len = len(exchange["user"].split())
            asst_len = len(exchange["assistant"].split())
            ratios.append(asst_len / max(user_len, 1))

    avg_ratio = sum(ratios) / len(ratios)
    pct_over_2x = len([r for r in ratios if r > 2]) / len(ratios)

    if avg_ratio > 2.0:
        print(f"WARNING: Average ratio {avg_ratio:.2f} exceeds 2x")
    if pct_over_2x > 0.50:
        print(f"WARNING: {pct_over_2x:.0%} of responses exceed 2x length")
```

### Praise Distribution

Check for artificial positivity escalation:

```python
def check_praise_distribution(transcripts):
    PRAISE_PATTERNS = ["that's great", "wonderful", "amazing", "fantastic"]

    early_praise = 0
    late_praise = 0
    early_count = 0
    late_count = 0

    for transcript in transcripts:
        exchanges = transcript["exchanges"]
        midpoint = len(exchanges) // 2

        for i, exchange in enumerate(exchanges):
            response = exchange["assistant"].lower()
            has_praise = any(p in response for p in PRAISE_PATTERNS)

            if i < midpoint:
                early_count += 1
                if has_praise:
                    early_praise += 1
            else:
                late_count += 1
                if has_praise:
                    late_praise += 1

    early_rate = early_praise / early_count
    late_rate = late_praise / late_count

    if late_rate > 2 * early_rate:
        print(f"WARNING: Late praise rate ({late_rate:.0%}) > 2x early ({early_rate:.0%})")
```

---

## Slicing Strategy

Create training examples from full transcripts:

```
50-turn transcript → ~8-10 training examples
```

### Random Slice Points

```python
import hashlib

MIN_CONTEXT = 3     # First slice at exchange 3 minimum
MIN_GAP = 2         # At least 2 exchanges between slices
MAX_GAP = 5         # At most 5 exchanges between slices

def get_slice_points(total_turns: int, transcript_id: str) -> list[int]:
    """Generate random slice points with stable seeding."""
    # Use SHA256 for stable seeding (Python's hash() varies between runs)
    seed = int(hashlib.sha256(transcript_id.encode()).hexdigest()[:8], 16)
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

### Token Validation

Each slice must fit within your token budget:

```python
MAX_TOKENS = 16_000  # Based on token economics decision

def validate_slice(transcript, slice_point):
    messages = format_as_messages(transcript[:slice_point])
    token_count = count_tokens(messages)

    if token_count > MAX_TOKENS:
        # Find largest valid slice
        for i in range(slice_point - 1, MIN_CONTEXT - 1, -1):
            if count_tokens(format_as_messages(transcript[:i])) <= MAX_TOKENS:
                return i
        return None  # Can't fit

    return slice_point
```

### Leakage Prevention

**Critical:** Split by transcript/persona FIRST, then slice within each split.

```python
# WRONG: Slice all, then split
all_slices = [slice for t in transcripts for slice in get_slices(t)]
train, val = random_split(all_slices)  # Leakage!

# CORRECT: Split by transcript, then slice
train_transcripts, val_transcripts = split_by_transcript(transcripts)
train_slices = [slice for t in train_transcripts for slice in get_slices(t)]
val_slices = [slice for t in val_transcripts for slice in get_slices(t)]
```

---

## Infrastructure

### Retry with Backoff

```python
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5)
)
async def call_llm(prompt, backend):
    ...
```

**Backend-specific strategies:**

| Backend | Attempts | Wait Strategy |
|---------|----------|---------------|
| Claude CLI | 10 | Fixed 1-hour (usage limit resets hourly) |
| Google | 10 | Extract delay from error, fallback to exponential |
| OpenAI | 5 | Standard exponential backoff |

**Google's custom retry strategy** (extracts suggested delay from 429 errors):

```python
import re

def _extract_google_retry_delay(exception: BaseException) -> float | None:
    """Extract retryDelay from Google API error response.

    Google 429 errors include RetryInfo: {'retryDelay': '16.412038513s'}
    """
    error_str = str(exception)
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]", error_str)
    if match:
        return float(match.group(1))
    return None

def _google_wait_strategy(retry_state) -> float:
    """Use Google's suggested retry delay when available."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if exception:
        delay = _extract_google_retry_delay(exception)
        if delay is not None:
            return delay + 1.0  # Add buffer
    # Fallback: exponential backoff
    return min(120.0, max(5.0, (2 ** retry_state.attempt_number) * 2))
```

### Progress Tracking

Track throughout generation:

```python
@dataclass
class GenerationStats:
    transcripts_generated: int = 0
    transcripts_assessed: int = 0
    transcripts_passed: int = 0
    criterion_failures: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self):
        if self.transcripts_assessed == 0:
            return 0
        return self.transcripts_passed / self.transcripts_assessed
```

### Minimum Length Enforcement

Prevent gaming with very short conversations:

```python
MIN_TURNS_FOR_ASSESSMENT = 3

def assess_if_valid(transcript):
    if len(transcript["exchanges"]) < MIN_TURNS_FOR_ASSESSMENT:
        return {"status": "TOO_SHORT", "score": None}
    return assess_transcript(transcript)
```

### Batch Checkpointing

For crash-resilient batch processing, use JSONL append pattern:

```python
def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load completed conversation IDs from checkpoint file."""
    if not checkpoint_path.exists():
        return set()

    completed = set()
    with open(checkpoint_path) as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if "conversation_id" in record:
                        completed.add(record["conversation_id"])
                except json.JSONDecodeError:
                    pass  # Skip malformed lines
    return completed

def append_checkpoint(checkpoint_path: Path, result: AssessmentResult) -> None:
    """Append a result atomically (single write, immediate flush)."""
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
```

**Checkpoint structure (JSONL):**
```json
{"conversation_id": "transcript_5000", "pass": true, "score": 0.867, ...}
{"conversation_id": "transcript_5001", "pass": false, "score": 0.733, ...}
```

**Resume pattern:**
```python
completed_ids = load_checkpoint(checkpoint_path)
to_process = [c for c in conversations if c.id not in completed_ids]
# Process only remaining conversations
```

---

## Cost Estimates

Example costs (Sonnet backend, conversational domain):

| Task | Per-Transcript | Time |
|------|----------------|------|
| Generation (25-turn) | ~$0.10-0.15 | 2-3 min |
| Assessment (17 criteria) | ~$0.05-0.08 | 1-2 min |
| Multi-backend assessment | ~$0.15-0.20 | 3-5 min |
| Fixup (if needed) | ~$0.03-0.05 | 30 sec |

**Use Claude Code CLI for zero marginal cost** if you have unlimited usage.

---

## Operational Commands

### Check Assessment Status

```bash
# Find unassessed transcripts
python3 -c "
import json
from pathlib import Path

transcripts = set(f.stem for f in Path('data/raw/transcripts').rglob('transcript_*.json')
                 if '_backup' not in str(f))

assessed = set()
for cp in Path('data/assessments').glob('*.jsonl'):
    for line in cp.read_text().strip().split('\n'):
        if line:
            assessed.add(json.loads(line).get('conversation_id', ''))

unassessed = sorted(transcripts - assessed)
print(f'Total: {len(transcripts)}, Assessed: {len(assessed)}, Unassessed: {len(unassessed)}')
"
```

### View Assessment Results

```bash
# Last 5 assessments
tail -5 data/assessments/checkpoint.jsonl | jq '{id: .conversation_id, pass: .pass, score: .score}'

# All failed assessments
cat data/assessments/*.jsonl | jq 'select(.pass == false) | {id: .conversation_id, score: .score, failed: .failed_checks}'
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limits | Use `concurrency=1`, exponential backoff |
| Corrupt checkpoint | Filter valid lines: `cat checkpoint.jsonl \| jq -c '.' > fixed.jsonl` |
| Missing API key | Set in `.env`: `GOOGLE_API_KEY=...` or `OPENAI_API_KEY=...` |
