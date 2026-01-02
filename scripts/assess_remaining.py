#!/usr/bin/env python3
"""Assess all remaining unassessed transcripts using assess_batch."""

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

# Set to None to assess all transcripts, or a list of prefixes to skip
# (e.g., ["transcript_7"] to skip in-progress 7000-series long transcripts)
SKIP_PREFIXES: list[str] | None = None

TRANSCRIPT_DIRS = [
    Path("data/raw/transcripts"),  # Root directory (includes 1000-series)
    Path("data/raw/transcripts/short"),
    Path("data/raw/transcripts/medium"),
    Path("data/raw/transcripts/long"),
    Path("data/raw/transcripts/very_long"),
]

# Checkpoints to load (skip already-assessed)
PRIOR_CHECKPOINTS = [
    Path("data/assessments/short_5000_checkpoint.jsonl"),
    Path("data/assessments/remaining_batch_checkpoint.jsonl"),
]
# Output for this batch
OUTPUT_CHECKPOINT = Path("data/assessments/remaining_batch_checkpoint.jsonl")


def find_all_transcripts() -> list[tuple[str, Path]]:
    """Find all transcripts across all directories (excluding archives/backups)."""
    transcripts = []
    for dir_path in TRANSCRIPT_DIRS:
        if not dir_path.exists():
            continue
        for f in dir_path.glob("transcript_*.json"):
            # Skip archive and backup directories
            if "_archive" in str(f) or "_backup" in str(f) or "_artifacts" in str(f):
                continue
            # Skip prefixes if configured
            if SKIP_PREFIXES and any(f.stem.startswith(p) for p in SKIP_PREFIXES):
                continue
            transcripts.append((f.stem, f))
    return sorted(transcripts)


async def main() -> None:
    setup_logging()
    get_backend("openai", model="gpt-5-mini")

    # Load all prior checkpoints to skip already-assessed
    prior_ids: set[str] = set()
    for cp in PRIOR_CHECKPOINTS:
        if cp.exists():
            prior_ids |= load_checkpoint(cp)

    # Find all transcripts and filter out prior assessments
    all_transcripts = find_all_transcripts()
    to_assess = [(tid, path) for tid, path in all_transcripts if tid not in prior_ids]

    print(
        f"Found {len(to_assess)} transcripts to assess ({len(prior_ids)} already done)"
    )

    if not to_assess:
        print("Nothing to assess!")
        return

    # Load conversations
    conversations: list[tuple[str, ConversationInput]] = []
    for tid, path in to_assess:
        conv = load_conversation_from_file(path)
        conversations.append((tid, conv))

    # assess_batch handles checkpointing, resume, errors, and progress
    await assess_batch(
        conversations,
        checkpoint_path=OUTPUT_CHECKPOINT,
        concurrency=1,  # Sequential to avoid rate limits
    )

    print(f"\nResults saved to {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    asyncio.run(main())
