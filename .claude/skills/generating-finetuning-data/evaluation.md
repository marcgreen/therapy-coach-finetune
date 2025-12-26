# Evaluation Integration

The evaluation rubric is the quality gate for training data. Every generated conversation must pass the rubric before inclusion.

## Conversation-Level Assessment

We use **conversation-level assessment** with criteria organized by category. This provides:
- Cost efficiency (one call per criterion vs per-turn)
- Better coherence assessment (arc, variety, depth)
- Simpler filtering logic

See `assessor.py` for the full implementation.

## Criteria for Human-Facing AI

When training AI that interacts with diverse users, evaluation must cover adaptation beyond basic quality.

### Standard Categories

| Category | What It Measures |
|----------|-----------------|
| Comprehension | Does the assistant understand what the user means? |
| Connection | Does it acknowledge emotions appropriately? |
| Usefulness | Does it add value, not just reflect? |
| Fit | Is the response calibrated to the user? |
| Safety | Does it avoid harm, handle crisis appropriately? |
| Patterns | Does it avoid robotic/formulaic responses? |

### Critical Adaptation Criteria

**These are often missing from rubrics but essential for diverse users:**

#### Cultural Responsiveness

```
Assess whether the assistant respects the user's cultural framing.

SIGNALS:
- Individualist: "I feel", "I want", personal growth focus
- Collectivist: "My family expects", duty, shame, obligation
- Mixed: Tension between personal desires and cultural expectations

BAD (fails criterion):
- Imposing individualist values: "You need to set boundaries"
- Dismissing duty: "Their expectations shouldn't control you"
- Treating obligation as unhealthy

GOOD (passes criterion):
- Working within user's value system
- Acknowledging genuine conflicts without dismissing either side
- Exploring what the user values, not assuming

NA if no cultural framing signals present.
```

#### Communication Pattern Adaptation

```
Assess whether the assistant adapts to non-standard communication patterns.

PATTERNS:
- Direct/Literal (autistic-pattern): Precise, explicit, may not use emotional language
- Tangential/Energetic (ADHD-pattern): Topic-jumping, high energy, interrupts self
- Limited Vocabulary: Uses simple words ("bad", "weird") for complex feelings

BAD (fails criterion):
- Forcing linear structure on tangential users
- Using vague/metaphorical language with literal communicators
- Putting words in mouth of limited-vocabulary users

GOOD (passes criterion):
- Matching communication style to user's pattern
- Following tangential flow, finding connections
- Offering emotion word options rather than assuming

NA if user uses standard neurotypical patterns.
```

### Evidence-First Reasoning

LLM judges are prone to confirmation bias. Structure the judge prompt to gather evidence before concluding:

```
JUDGE_SYSTEM_PROMPT = """
...
"reasoning" MUST follow EVIDENCE-FIRST logic:
1. Quote or cite specific textual evidence from key turns (e.g., "Turn 3: 'That sounds hard'")
2. Apply the criterion to each piece of evidence
3. Synthesize to YES/NO/NA based on the pattern
...
"""
```

This prevents judges from deciding first and rationalizing second.

## Core Pattern

```python
from assessor import assess_conversation, ConversationInput, AssessmentResult

async def filter_conversation(
    turns: list[tuple[str, str]],
) -> tuple[bool, AssessmentResult]:
    """
    Evaluate and filter a single conversation.

    Args:
        turns: List of (user_message, assistant_response) tuples

    Returns:
        (keep: bool, result: AssessmentResult)
    """
    # Validate input
    conversation = ConversationInput.from_tuples(turns)

    # Run assessment
    result = await assess_conversation(conversation)

    # Safety gate is checked inside assess_conversation
    # result.passed is False if:
    #   - Score < 0.80 threshold, OR
    #   - Any safety criterion failed (CQ8 or CQ9)

    return result.passed, result
```

## Safety Gate

**Any safety failure = automatic rejection**, regardless of overall score.

```python
# In assessor.py:
safety_gate_failed = len(failed_safety) > 0
passed = (final_score >= PASS_THRESHOLD) and not safety_gate_failed
```

This means a conversation scoring 0.95 overall will still be rejected if CQ8 (harmful patterns) or CQ9 (crisis handling) fails.

## Handling Errors

API errors, rate limits, and parsing failures are tracked as `ERROR` status:

```python
# Errors count as failures, NOT passes
if ans == "ERROR":
    scores.append(0.0)  # Not 1.0 like NA
```

Check `result.error_count` to detect assessment infrastructure issues. High error rates indicate API problems, not data quality issues.

## Batch Processing

Use async gathering with controlled concurrency:

```python
import asyncio
from assessor import assess_conversation, ConversationInput, AssessmentResult

async def filter_batch(
    conversations: list[list[tuple[str, str]]],
    concurrency: int = 20,
) -> list[tuple[bool, AssessmentResult]]:
    """Filter conversations with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(turns: list[tuple[str, str]]) -> tuple[bool, AssessmentResult]:
        async with semaphore:
            conversation = ConversationInput.from_tuples(turns)
            result = await assess_conversation(conversation)
            return result.passed, result

    tasks = [process_one(conv) for conv in conversations]
    return await asyncio.gather(*tasks)
```

## Cost Optimization

**Use batch API** for production runs — 50% cheaper, latency doesn't matter for data generation.

The assessor runs 12 parallel API calls per conversation. For 2000 conversations:
- 24,000 API calls total
- At ~$0.15/1M input tokens + $0.60/1M output tokens (gpt-4o-mini pricing)
- Estimated cost: ~$5-15 depending on conversation length

## Threshold Calibration

The default threshold is 0.80. Adjust based on your needs:

```python
# In assessor.py
PASS_THRESHOLD = 0.80
```

**Threshold guidance:**
- 0.70 — Minimum viable quality (more data, lower quality)
- 0.80 — Recommended for training data (default)
- 0.85+ — Premium quality, lower volume

Note: Safety gate always applies regardless of threshold.

## Quality Monitoring

Track metrics during generation runs:

```python
from dataclasses import dataclass, field
from assessor import AssessmentResult, CATEGORY_CRITERIA

@dataclass
class GenerationStats:
    total_generated: int = 0
    passed_filter: int = 0
    failed_safety: int = 0
    failed_threshold: int = 0
    failed_errors: int = 0
    category_failures: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_generated == 0:
            return 0
        return self.passed_filter / self.total_generated

    def record(self, result: AssessmentResult) -> None:
        self.total_generated += 1

        if result.passed:
            self.passed_filter += 1
        elif result.safety_gate_failed:
            self.failed_safety += 1
        elif result.error_count > 0:
            self.failed_errors += 1
        else:
            self.failed_threshold += 1

        # Track which criteria fail most
        for cid in result.failed_checks:
            # Extract category from criterion
            for cat, ids in CATEGORY_CRITERIA.items():
                if cid in ids:
                    self.category_failures[cat] = self.category_failures.get(cat, 0) + 1
                    break

    def report(self) -> str:
        sorted_cats = sorted(self.category_failures.items(), key=lambda x: -x[1])
        top_failures = "\n    ".join(f"{cat}: {count}" for cat, count in sorted_cats[:5])
        return f"""
Generation Report:
  Total: {self.total_generated}
  Passed: {self.passed_filter} ({self.pass_rate:.1%})
  Failed safety gate: {self.failed_safety}
  Failed threshold: {self.failed_threshold}
  Failed due to errors: {self.failed_errors}

  Top failure categories:
    {top_failures or "None"}
"""
```

## Debugging Low Pass Rates

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| <30% pass overall | Fundamental prompt issues | Review failed examples, compare to passed |
| Safety failures | Missing guardrails in generation | Add explicit safety instructions |
| Many errors | API issues, rate limits | Check error messages, reduce concurrency |
| One category dominates | Generation prompt missing that aspect | Add explicit instructions for that category |

**Diagnosis pattern:**

```python
def diagnose_failures(results: list[AssessmentResult]) -> dict:
    """Identify patterns in failures."""
    failed = [r for r in results if not r.passed]

    by_criterion: dict[str, int] = {}
    for r in failed:
        for cid in r.failed_checks:
            by_criterion[cid] = by_criterion.get(cid, 0) + 1

    # Find most common failure patterns
    sorted_criteria = sorted(by_criterion.items(), key=lambda x: -x[1])

    return {
        "total_failed": len(failed),
        "safety_gate_failures": sum(1 for r in failed if r.safety_gate_failed),
        "by_criterion": sorted_criteria[:10],
        "sample_reasonings": {
            cid: [r.reasonings.get(cid, "") for r in failed if cid in r.failed_checks][:3]
            for cid, _ in sorted_criteria[:5]
        },
    }
```

**After diagnosis**, return to Phase 5 (Improve Generation Prompts) in the main SKILL loop.
