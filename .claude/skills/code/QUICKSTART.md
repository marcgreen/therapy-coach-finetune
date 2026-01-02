# Quickstart: Using the Infrastructure

How to integrate `infrastructure.py` into your fine-tuning project.

---

## 1. Project Structure

Create this structure (adapt names for your domain):

```
my-finetune-project/
├── config/
│   ├── taxonomy.yaml           # Input distribution (from taxonomy-guide)
│   ├── rubric.yaml             # Evaluation criteria (from rubric-guide)
│   ├── persona-template.yaml   # User diversity (from persona-guide)
│   └── prompts/
│       ├── user_sim.md         # User simulator prompt
│       ├── assistant.md        # Assistant generation prompt
│       └── system.md           # System prompt for training data
├── data/
│   ├── raw/
│   │   └── transcripts/        # Generated conversations
│   ├── assessments/            # Assessment checkpoints
│   └── processed/
│       └── training_data.jsonl # Final training examples
├── scripts/
│   ├── generate.py             # Two-agent generation
│   ├── assess.py               # Multi-backend assessment
│   └── slice.py                # Create training examples
├── infrastructure.py           # Copy from skills/code/
└── criteria.py                 # YOUR domain-specific criteria
```

---

## 2. Copy Infrastructure

```bash
cp .claude/skills/code/infrastructure.py ./infrastructure.py
```

---

## 3. Define Your Criteria

Create `criteria.py` with your domain-specific evaluation criteria:

```python
# criteria.py
from pydantic import BaseModel
from typing import Literal

# Your categories and weights (must sum to 1.0)
CATEGORIES = {
    "accuracy": {"weight": 0.25, "criteria": ["C1", "C2"]},
    "helpfulness": {"weight": 0.35, "criteria": ["C3", "C4", "C5"]},
    "safety": {"weight": 0.20, "criteria": ["C6"]},
    "tone": {"weight": 0.20, "criteria": ["C7", "C8"]},
}

# Safety gates - any NO = auto-reject
SAFETY_GATES = ["C6"]

# Criteria where NA is not allowed
NA_INVALID = ["C1", "C6"]

# Pass threshold
PASS_THRESHOLD = 0.80

# Pydantic model for structured assessment
class CriterionResult(BaseModel):
    answer: Literal["YES", "NO", "NA"]
    reasoning: str

class AssessmentOutput(BaseModel):
    C1: CriterionResult
    C2: CriterionResult
    C3: CriterionResult
    C4: CriterionResult
    C5: CriterionResult
    C6: CriterionResult
    C7: CriterionResult
    C8: CriterionResult
```

---

## 4. Generate Conversations

```python
# scripts/generate.py
import asyncio
from pathlib import Path
from infrastructure import get_backend, append_checkpoint, load_checkpoint

# Configuration
TRANSCRIPTS_DIR = Path("data/raw/transcripts")
CHECKPOINT_PATH = Path("data/raw/generation_checkpoint.jsonl")
TARGET_TRANSCRIPTS = 100
TARGET_TURNS = 25

async def generate_transcript(persona: dict, backend) -> dict:
    """Generate one multi-turn conversation."""
    exchanges = []

    for turn in range(TARGET_TURNS):
        # User simulator
        user_prompt = build_user_prompt(persona, exchanges, turn)
        user_result = await backend.complete(user_prompt)
        user_msg = user_result.content

        # Assistant response
        assistant_prompt = build_assistant_prompt(exchanges, user_msg)
        assistant_result = await backend.complete(
            assistant_prompt,
            system=load_system_prompt()
        )
        assistant_msg = assistant_result.content

        exchanges.append({"user": user_msg, "assistant": assistant_msg})

    return {
        "id": f"transcript_{persona['id']}",
        "persona": persona,
        "exchanges": exchanges,
    }

async def main():
    backend = get_backend("openai", model="gpt-4o")
    completed = load_checkpoint(CHECKPOINT_PATH)
    personas = load_personas()  # Your persona generation

    for persona in personas:
        if persona["id"] in completed:
            continue

        transcript = await generate_transcript(persona, backend)

        # Save transcript
        transcript_path = TRANSCRIPTS_DIR / f"{transcript['id']}.json"
        transcript_path.write_text(json.dumps(transcript, indent=2))

        # Checkpoint
        append_checkpoint(CHECKPOINT_PATH, {"id": transcript["id"]})
        print(f"Generated {transcript['id']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Assess Conversations

```python
# scripts/assess.py
import asyncio
from pathlib import Path
from infrastructure import (
    get_backend,
    append_checkpoint,
    load_checkpoint,
    compute_category_score,
    compute_weighted_score,
    compute_length_stats,
    AssessmentResult,
)
from criteria import (
    CATEGORIES,
    SAFETY_GATES,
    NA_INVALID,
    PASS_THRESHOLD,
    AssessmentOutput,
)

CHECKPOINT_PATH = Path("data/assessments/checkpoint.jsonl")

async def assess_transcript(transcript: dict, backend) -> AssessmentResult:
    """Assess one transcript."""
    # Pre-compute length stats (LLMs can't count)
    turns = [(ex["user"], ex["assistant"]) for ex in transcript["exchanges"]]
    length_stats = compute_length_stats(turns)

    # Build assessment prompt
    prompt = build_assessment_prompt(transcript, length_stats)

    # Get structured assessment
    output = await backend.complete_structured(
        prompt,
        response_model=AssessmentOutput,
        system="You are an expert evaluator..."
    )

    # Convert to answers dict
    answers = {
        cid: getattr(output, cid).answer
        for cid in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    }
    reasonings = {
        cid: getattr(output, cid).reasoning
        for cid in answers
    }

    # Compute scores
    category_scores = {}
    for cat_name, cat_config in CATEGORIES.items():
        category_scores[cat_name] = compute_category_score(
            answers,
            cat_config["criteria"],
            set(NA_INVALID)
        )

    weights = {cat: cfg["weight"] for cat, cfg in CATEGORIES.items()}
    score = compute_weighted_score(category_scores, weights)

    # Check safety gates
    safety_failed = any(answers.get(g) == "NO" for g in SAFETY_GATES)

    # Determine pass/fail
    passed = score >= PASS_THRESHOLD and not safety_failed
    failed_checks = [cid for cid, ans in answers.items() if ans == "NO"]

    return AssessmentResult(
        id=transcript["id"],
        passed=passed,
        score=score,
        threshold=PASS_THRESHOLD,
        category_scores=category_scores,
        answers=answers,
        reasonings=reasonings,
        failed_checks=failed_checks,
        safety_gate_failed=safety_failed,
        error_count=sum(1 for a in answers.values() if a == "ERROR"),
    )

async def main():
    # Use multiple backends
    backends = [
        get_backend("openai", model="gpt-4o"),
        get_backend("google", model="gemini-2.0-flash"),
    ]

    completed = load_checkpoint(CHECKPOINT_PATH)
    transcripts = load_transcripts()  # Your loading logic

    for transcript in transcripts:
        if transcript["id"] in completed:
            continue

        # Assess with all backends
        results = []
        for backend in backends:
            result = await assess_transcript(transcript, backend)
            results.append(result)

        # Take strictest result
        final_result = min(results, key=lambda r: r.score)

        # Checkpoint
        append_checkpoint(CHECKPOINT_PATH, final_result.to_dict())
        status = "PASS" if final_result.passed else "FAIL"
        print(f"{transcript['id']}: {status} ({final_result.score:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Slice for Training

```python
# scripts/slice.py
import json
from pathlib import Path
from infrastructure import get_slice_points, count_messages_tokens

MAX_TOKENS = 16_000
OUTPUT_PATH = Path("data/processed/training_data.jsonl")

def slice_transcript(transcript: dict, system_prompt: str) -> list[dict]:
    """Create training examples from one transcript."""
    exchanges = transcript["exchanges"]
    slice_points = get_slice_points(len(exchanges), transcript["id"])

    examples = []
    for point in slice_points:
        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        for ex in exchanges[:point]:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})

        # Check token limit
        if count_messages_tokens(messages) > MAX_TOKENS:
            continue

        examples.append({"messages": messages})

    return examples

def main():
    system_prompt = Path("config/prompts/system.md").read_text()
    assessments = load_assessments()  # Load from checkpoint

    with open(OUTPUT_PATH, "w") as f:
        for assessment in assessments:
            if not assessment["pass"]:
                continue

            transcript = load_transcript(assessment["id"])
            examples = slice_transcript(transcript, system_prompt)

            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    print(f"Created {count_lines(OUTPUT_PATH)} training examples")

if __name__ == "__main__":
    main()
```

---

## 7. Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "tenacity==8.2.3",
    "pydantic==2.5.0",
    "tiktoken==0.5.2",
    "openai==1.12.0",        # If using OpenAI
    "google-genai==0.3.0",   # If using Google
]
```

---

## Workflow Summary

```
1. Design phase (finetune-design skill)
   ├── Define taxonomy, rubric, personas
   └── Create prompts

2. Generate phase
   ├── Run scripts/generate.py
   ├── Review 5 transcripts (human-in-loop)
   ├── Iterate prompts
   └── Scale when ≥70% pass rate

3. Assess phase
   ├── Run scripts/assess.py with multiple backends
   ├── Review disagreements
   └── Add calibration examples as needed

4. Slice phase
   └── Run scripts/slice.py to create training_data.jsonl

5. Train phase (finetune-train skill)
   └── Submit to HuggingFace Jobs or train locally
```

---

## Common Patterns

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### Running with uv

```bash
uv run python scripts/generate.py
uv run python scripts/assess.py
uv run python scripts/slice.py
```

### Resuming After Crash

The checkpoint pattern handles this automatically:
- `load_checkpoint()` returns completed IDs
- Skip already-processed items
- `append_checkpoint()` writes immediately after each item
