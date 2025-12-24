# Evaluation Integration

The evaluation rubric is the quality gate for training data. Every generated example must pass the rubric before inclusion.

## Rubric Requirements

Your evaluation rubric must provide:

| Component | Purpose |
|-----------|---------|
| **Criteria list** | What to check (binary YES/NO/NA questions) |
| **Evaluate function** | Score a single response against all criteria |
| **Score function** | Aggregate criterion answers to pass/fail |
| **Safety gate** | Hard fail on critical violations (if applicable) |

## Core Pattern

```python
async def filter_example(
    input_text: str,
    output_text: str,
    threshold: float = 0.80,
) -> tuple[bool, dict]:
    """
    Evaluate and filter a single example.

    Returns:
        (keep: bool, details: dict)
    """
    # Run your evaluation (however implemented)
    answers = await evaluate(input_text, output_text)
    result = score(answers)

    # Check safety gate first (if you have safety criteria)
    if result.get("safety_failure"):
        return False, {"reason": "safety", "failed": result["failed_safety"]}

    # Check threshold
    keep = result["pass"] and result["score"] >= threshold

    return keep, {
        "score": result["score"],
        "category_scores": result.get("category_scores", {}),
        "failed_checks": result.get("failed_checks", []),
    }
```

## Multi-Turn Conversations

For conversation data, evaluate at two levels:

```python
async def filter_conversation(
    conversation: list[tuple[str, str]],  # (input, output) pairs
    turn_threshold: float = 0.80,
    conv_threshold: float = 0.66,
) -> tuple[bool, dict]:
    """
    All turns must pass AND conversation-level criteria must pass.
    """
    # Evaluate each turn
    turn_results = []
    for input_text, output_text in conversation:
        keep, details = await filter_example(input_text, output_text, turn_threshold)
        turn_results.append({"keep": keep, **details})

    # If any turn fails, reject conversation
    if not all(t["keep"] for t in turn_results):
        failed_turns = [i for i, t in enumerate(turn_results) if not t["keep"]]
        return False, {"reason": "turn_failure", "failed_turns": failed_turns}

    # Evaluate conversation-level criteria (variety, arc, coherence, etc.)
    conv_result = await evaluate_conversation(conversation)

    if not conv_result["pass"]:
        return False, {
            "reason": "conversation_failure",
            "failed_checks": conv_result["failed_checks"],
        }

    return True, {
        "turn_scores": [t["score"] for t in turn_results],
        "conversation_score": conv_result["score"],
    }
```

## Batch Processing

Use async gathering with controlled concurrency:

```python
import asyncio

async def filter_batch(
    examples: list[tuple[str, str]],
    threshold: float = 0.80,
    concurrency: int = 50,
):
    """Filter examples with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(idx, input_text, output_text):
        async with semaphore:
            keep, details = await filter_example(input_text, output_text, threshold)
            return idx, keep, details

    tasks = [
        process_one(i, inp, out)
        for i, (inp, out) in enumerate(examples)
    ]

    for coro in asyncio.as_completed(tasks):
        yield await coro
```

## Cost Optimization

**Use batch API** — 50% cheaper, latency doesn't matter for data generation:

```python
async def evaluate_with_batch_api(examples: list[tuple[str, str]]) -> list[dict]:
    """Submit evaluation jobs to batch API."""
    # Create batch file — one request per (example, criterion) pair
    batch_requests = []
    for i, (input_text, output_text) in enumerate(examples):
        for criterion in CRITERIA:
            batch_requests.append({
                "custom_id": f"{i}_{criterion.id}",
                "method": "POST",
                "url": "/v1/responses",  # or /v1/chat/completions
                "body": make_eval_request(criterion, input_text, output_text),
            })

    # Submit batch (OpenAI example)
    batch = await client.batches.create(
        input_file=upload_batch_file(batch_requests),
        endpoint="/v1/responses",
        completion_window="24h",
    )

    # Poll for completion
    while batch.status != "completed":
        await asyncio.sleep(60)
        batch = await client.batches.retrieve(batch.id)

    return parse_batch_results(batch.output_file)
```

## Threshold Calibration

Run pilot evaluation to find the right threshold:

```python
async def calibrate_threshold(
    pilot_examples: list[tuple[str, str]],
    target_pass_rate: float = 0.50,
) -> float:
    """Find threshold that yields target pass rate."""
    scores = []
    for input_text, output_text in pilot_examples:
        answers = await evaluate(input_text, output_text)
        result = score(answers)
        if not result.get("safety_failure"):
            scores.append(result["score"])

    scores.sort(reverse=True)
    threshold_idx = int(len(scores) * target_pass_rate)
    return scores[threshold_idx]
```

**Threshold guidance:**
- 0.70 — Minimum viable quality
- 0.80 — Recommended for training data
- 0.85+ — Premium quality, lower volume

## Quality Monitoring

Track metrics during generation runs:

```python
from dataclasses import dataclass, field

@dataclass
class GenerationStats:
    total_generated: int = 0
    passed_filter: int = 0
    failed_safety: int = 0
    failed_threshold: int = 0
    category_failures: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_generated == 0:
            return 0
        return self.passed_filter / self.total_generated

    def record(self, keep: bool, details: dict):
        self.total_generated += 1
        if keep:
            self.passed_filter += 1
        elif details.get("reason") == "safety":
            self.failed_safety += 1
        else:
            self.failed_threshold += 1
            for check in details.get("failed_checks", []):
                # Extract category from criterion ID (e.g., "CP1" -> "CP")
                category = check[:2] if len(check) >= 2 else check
                self.category_failures[category] = self.category_failures.get(category, 0) + 1

    def report(self) -> str:
        sorted_cats = sorted(self.category_failures.items(), key=lambda x: -x[1])
        top_failures = "\n    ".join(f"{cat}: {count}" for cat, count in sorted_cats[:5])
        return f"""
Generation Report:
  Total: {self.total_generated}
  Passed: {self.passed_filter} ({self.pass_rate:.1%})
  Failed safety: {self.failed_safety}
  Failed threshold: {self.failed_threshold}

  Top failure categories:
    {top_failures}
"""
```

## Debugging Low Pass Rates

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| <30% pass overall | Fundamental prompt issues | Review failed examples, compare to passed |
| One category dominates failures | Generation prompt missing that aspect | Add explicit instructions for that category |
| High variance in scores | Inconsistent generation | Lower temperature, add constraints |
| Safety failures | Missing guardrails | Add explicit safety instructions to prompt |

**Diagnosis pattern:**

```python
def diagnose_failures(failed_examples: list[dict]) -> dict:
    """Identify patterns in failures."""
    by_category = {}
    by_criterion = {}

    for ex in failed_examples:
        for check in ex.get("failed_checks", []):
            category = check[:2] if len(check) >= 2 else check
            by_category[category] = by_category.get(category, 0) + 1
            by_criterion[check] = by_criterion.get(check, 0) + 1

    return {
        "by_category": sorted(by_category.items(), key=lambda x: -x[1]),
        "by_criterion": sorted(by_criterion.items(), key=lambda x: -x[1]),
        "sample_failures": failed_examples[:5],  # For manual review
    }
```

**After diagnosis**, return to Phase 5 (Improve Generation Prompts) in the main loop.
