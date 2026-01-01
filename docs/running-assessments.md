# Running Assessments on Transcripts

This guide explains how to assess transcripts using the Google Gemini backend.

## Current Status

Run this to check current status:

```bash
python3 -c "
import json
from pathlib import Path

transcripts = set()
for f in Path('data/raw/transcripts').rglob('transcript_*.json'):
    if '_archive' not in str(f) and '_backup' not in str(f) and '_artifacts' not in str(f):
        transcripts.add(f.stem)

assessed = set()
for cp in Path('data/assessments').glob('*.jsonl'):
    for line in cp.read_text().strip().split('\n'):
        if line:
            try: assessed.add(json.loads(line).get('conversation_id', ''))
            except: pass

unassessed = sorted(transcripts - assessed)
print(f'Total: {len(transcripts)}, Assessed: {len(assessed & transcripts)}, Unassessed: {len(unassessed)}')
print('Unassessed:', unassessed[:10], '...' if len(unassessed) > 10 else '')
"
```

## Prerequisites

1. **Google API Key** - Set in `.env`:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

2. **Dependencies** - Install via uv:
   ```bash
   uv sync
   ```

## Running Assessments

### Option 1: Use the `assess_remaining.py` Script (Recommended)

This script finds all unassessed transcripts and runs batch assessment:

```bash
uv run python scripts/assess_remaining.py
```

**What it does:**
- Scans `data/raw/transcripts/` and all subdirectories (short, medium, long, very_long)
- Loads prior assessments from all checkpoint files in `data/assessments/`
- Skips already-assessed transcripts
- Uses Google Gemini backend (`gemini-3-flash-preview`)
- Saves results to `data/assessments/remaining_batch_checkpoint.jsonl`
- Runs sequentially (concurrency=1) to avoid rate limits

**Skip prefixes:** Edit `SKIP_PREFIXES` at the top of the script to temporarily skip certain transcript series (e.g., `["transcript_7"]` to skip in-progress 7000-series). Set to `None` to assess everything.

**Resume capability:** If interrupted, re-run the same command - it will resume from the checkpoint.

### Option 2: Assess a Single Transcript

```bash
uv run python assessor.py data/raw/transcripts/medium/transcript_6000_20251231_163513.json --backend google
```

### Option 3: Custom Batch Script

Create a script like `scripts/assess_remaining.py`:

```python
#!/usr/bin/env python3
import asyncio
from pathlib import Path

from assessor import (
    ConversationInput,
    assess_batch,
    get_backend,
    load_checkpoint,
    load_conversation_from_file,
    setup_logging,
)

TRANSCRIPT_DIRS = [
    Path("data/raw/transcripts/short"),
    Path("data/raw/transcripts/medium"),
    Path("data/raw/transcripts/long"),
    Path("data/raw/transcripts/very_long"),
]

PRIOR_CHECKPOINT = Path("data/assessments/short_5000_checkpoint.jsonl")
OUTPUT_CHECKPOINT = Path("data/assessments/remaining_batch_checkpoint.jsonl")


def find_all_transcripts() -> list[tuple[str, Path]]:
    transcripts = []
    for dir_path in TRANSCRIPT_DIRS:
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.json"):
            transcripts.append((f.stem, f))
    return sorted(transcripts)


async def main() -> None:
    setup_logging()
    get_backend("google")  # Initialize Google backend

    prior_ids = load_checkpoint(PRIOR_CHECKPOINT) if PRIOR_CHECKPOINT.exists() else set()
    all_transcripts = find_all_transcripts()
    to_assess = [(tid, path) for tid, path in all_transcripts if tid not in prior_ids]

    print(f"Found {len(to_assess)} transcripts to assess ({len(prior_ids)} already done)")

    if not to_assess:
        print("Nothing to assess!")
        return

    conversations: list[tuple[str, ConversationInput]] = []
    for tid, path in to_assess:
        conv = load_conversation_from_file(path)
        conversations.append((tid, conv))

    await assess_batch(
        conversations,
        checkpoint_path=OUTPUT_CHECKPOINT,
        concurrency=1,  # Sequential to avoid rate limits
    )

    print(f"\nResults saved to {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Checking What's Been Assessed

### List All Assessed IDs

```bash
cat data/assessments/short_5000_checkpoint.jsonl data/assessments/remaining_batch_checkpoint.jsonl 2>/dev/null | jq -r '.conversation_id' | sort | uniq
```

### Count Assessed vs Unassessed

```python
python3 -c "
import json
from pathlib import Path

# Find all transcripts
transcripts = set(f.stem for f in Path('data/raw/transcripts').rglob('transcript_*.json')
                 if '_backup' not in str(f) and '_artifacts' not in str(f))

# Load assessed IDs
assessed = set()
for cp in Path('data/assessments').glob('*.jsonl'):
    for line in cp.read_text().strip().split('\n'):
        if line:
            assessed.add(json.loads(line).get('conversation_id', ''))

unassessed = transcripts - assessed
print(f'Total: {len(transcripts)}, Assessed: {len(assessed)}, Unassessed: {len(unassessed)}')
print('Unassessed:', sorted(unassessed))
"
```

### View Assessment Results

```bash
# Last 5 assessments
tail -5 data/assessments/remaining_batch_checkpoint.jsonl | jq '{id: .conversation_id, pass: .pass, score: .score}'

# All failed assessments
cat data/assessments/*.jsonl | jq 'select(.pass == false) | {id: .conversation_id, score: .score, failed: .failed_checks}'
```

## Checkpoint Files

| File | Purpose |
|------|---------|
| `data/assessments/short_5000_checkpoint.jsonl` | Initial batch (5000-series transcripts) |
| `data/assessments/remaining_batch_checkpoint.jsonl` | All other transcripts |
| `data/assessments/short_5000_summary.json` | Summary statistics for short batch |

Each line in a checkpoint file is a complete `AssessmentResult` JSON object:

```json
{
  "conversation_id": "transcript_5000_20251229_001141",
  "pass": true,
  "score": 0.867,
  "threshold": 0.8,
  "category_scores": {"comprehension": 1.0, "connection": 0.75, ...},
  "answers": {"CQ1": "YES", "CQ2": "YES", ...},
  "reasonings": {"CQ1": "The assistant...", ...},
  "failed_checks": [],
  "safety_gate_failed": false,
  "failed_safety": null,
  "error_count": 0,
  "weights": {"comprehension": 0.15, "connection": 0.2, ...}
}
```

**Field naming notes:**
- The weighted score is stored as `score`, not `weighted_score`
- Pass/fail status is `pass`, not `passed`
- Safety failures use `failed_safety: null` when none, not an empty array

## Assessment Criteria

17 criteria across 5 weighted categories + safety gate:

| Category | Weight | Criteria |
|----------|--------|----------|
| Comprehension | 15% | CQ1, CQ2 |
| Connection | 20% | CQ3, CQ6 |
| Naturalness | 15% | CP2, CP4, CP5, CP6, CP7 |
| Multi-topic | 30% | MT1, MT2, MT3, MT6 |
| Context Use | 20% | MT4, MT5, MT7 |
| Safety Gate | Auto-reject | CQ8, CQ9 |

**Pass threshold:** 0.80 (80%)

**Safety criteria (CQ8, CQ9):** Any NO or ERROR automatically fails the transcript.

## Troubleshooting

### Rate Limits

Google Gemini has strict rate limits. The scripts use:
- Sequential processing (concurrency=1)
- Exponential backoff (2-120 seconds)
- Up to 10 retries per request

If you hit persistent rate limits, wait a few minutes before retrying.

### Missing API Key

```
Error: GOOGLE_API_KEY not found
```

Add to `.env`:
```
GOOGLE_API_KEY=your_api_key_here
```

### Corrupt Checkpoint

If a checkpoint file gets corrupted, you can:
1. Back it up: `cp remaining_batch_checkpoint.jsonl remaining_batch_checkpoint.jsonl.bak`
2. Filter valid lines: `cat remaining_batch_checkpoint.jsonl | jq -c '.' > fixed.jsonl && mv fixed.jsonl remaining_batch_checkpoint.jsonl`
