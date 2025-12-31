#!/usr/bin/env python3
"""Gather paths to all passing transcripts."""

import json
from pathlib import Path


CHECKPOINTS = [
    Path("data/assessments/short_5000_checkpoint.jsonl"),
    Path("data/assessments/remaining_batch_checkpoint.jsonl"),
]

TRANSCRIPT_DIRS = [
    Path("data/raw/transcripts/short"),
    Path("data/raw/transcripts/medium"),
    Path("data/raw/transcripts/long"),
    Path("data/raw/transcripts/very_long"),
]

OUTPUT = Path("data/processed/passing_transcripts.json")


def find_transcript_path(transcript_id: str) -> Path | None:
    """Find transcript file by ID across all directories."""
    for dir_path in TRANSCRIPT_DIRS:
        if not dir_path.exists():
            continue
        candidate = dir_path / f"{transcript_id}.json"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    # Collect all assessments, keeping latest per ID
    assessments: dict[str, dict] = {}

    for checkpoint in CHECKPOINTS:
        if not checkpoint.exists():
            print(f"Warning: {checkpoint} not found, skipping")
            continue

        with open(checkpoint) as f:
            for line in f:
                data = json.loads(line)
                conv_id = data["conversation_id"]
                # Later entries overwrite earlier ones (latest wins)
                assessments[conv_id] = data

    print(f"Total unique assessments: {len(assessments)}")

    # Filter to passing only
    passing = []
    for conv_id, data in assessments.items():
        if not data.get("pass", False):
            continue

        # Find the transcript file path
        path = find_transcript_path(conv_id)
        if path is None:
            print(f"Warning: Could not find transcript file for {conv_id}")
            continue

        passing.append(
            {
                "id": conv_id,
                "source_file": str(path),
                "score": data.get("score"),
            }
        )

    # Sort by ID for reproducibility
    passing.sort(key=lambda x: x["id"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"count": len(passing), "transcripts": passing}, f, indent=2)

    print(f"Found {len(passing)} passing transcripts")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
