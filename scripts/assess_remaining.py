#!/usr/bin/env python3
"""Assess all remaining unassessed transcripts."""

import asyncio
import json
from pathlib import Path

from assessor import (
    assess_conversation,
    get_backend,
    load_conversation_from_file,
    setup_logging,
)

TRANSCRIPT_DIRS = [
    ("short", Path("data/raw/transcripts/short")),
    ("medium", Path("data/raw/transcripts/medium")),
    ("long", Path("data/raw/transcripts/long")),
    ("very_long", Path("data/raw/transcripts/very_long")),
]

# Already assessed (from short_5000 batch)
ASSESSED_CHECKPOINT = Path("data/assessments/short_5000_checkpoint.jsonl")
OUTPUT_CHECKPOINT = Path("data/assessments/remaining_batch_checkpoint.jsonl")


def load_assessed_ids() -> set[str]:
    """Load IDs of already-assessed transcripts."""
    assessed = set()
    if ASSESSED_CHECKPOINT.exists():
        with open(ASSESSED_CHECKPOINT) as f:
            for line in f:
                data = json.loads(line)
                assessed.add(data["conversation_id"])
    if OUTPUT_CHECKPOINT.exists():
        with open(OUTPUT_CHECKPOINT) as f:
            for line in f:
                data = json.loads(line)
                assessed.add(data["conversation_id"])
    return assessed


def find_unassessed_transcripts() -> list[Path]:
    """Find all transcripts not yet assessed."""
    assessed_ids = load_assessed_ids()
    unassessed = []

    for _category, dir_path in TRANSCRIPT_DIRS:
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.json"):
            transcript_id = f.stem
            if transcript_id not in assessed_ids:
                unassessed.append(f)

    return sorted(unassessed)


async def main() -> None:
    setup_logging()
    get_backend("google")  # Use Google backend like existing assessment

    unassessed = find_unassessed_transcripts()
    print(f"Found {len(unassessed)} unassessed transcripts")

    for i, path in enumerate(unassessed):
        print(f"\n[{i + 1}/{len(unassessed)}] Assessing {path.name}...")

        # Load transcript using assessor's loader (handles "exchanges" format)
        conversation = load_conversation_from_file(path)

        # Get transcript ID from the JSON file
        with open(path) as f:
            transcript_data = json.load(f)
        transcript_id = transcript_data.get("id", path.stem)

        result = await assess_conversation(conversation, conversation_id=transcript_id)

        # Append to checkpoint
        with open(OUTPUT_CHECKPOINT, "a") as f:
            f.write(
                json.dumps(
                    {
                        "conversation_id": transcript_id,
                        "pass": result.passed,
                        "score": result.score,
                        "safety_gate_failed": result.safety_gate_failed,
                        "category_scores": result.category_scores,
                        "failed_checks": result.failed_checks,
                        "source_file": str(path),
                    }
                )
                + "\n"
            )

        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} (score={result.score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
