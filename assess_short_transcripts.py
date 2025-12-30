#!/usr/bin/env python3
"""
Batch assessment script for short transcripts (5000-series).

Usage:
    uv run python assess_short_transcripts.py

Features:
    - Uses Google backend (Gemini)
    - Checkpointing: resumes from where it left off
    - Appends results to existing checkpoint file
    - Prints summary when complete

Rerun after new transcripts are generated to assess only new ones.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from assessor import (
    assess_batch,
    get_backend,
    load_checkpoint_results,
    load_conversation_from_file,
    setup_logging,
)


TRANSCRIPT_DIR = Path("data/raw/transcripts/short")
CHECKPOINT_PATH = Path("data/assessments/short_5000_checkpoint.jsonl")
SUMMARY_PATH = Path("data/assessments/short_5000_summary.json")

# Match transcripts with ID pattern 5XXX
TRANSCRIPT_PATTERN = "transcript_5*.json"


async def main() -> None:
    setup_logging()

    # Initialize Google backend
    get_backend(backend_type="google")

    # Find all matching transcripts
    transcript_files = sorted(TRANSCRIPT_DIR.glob(TRANSCRIPT_PATTERN))

    print(f"Found {len(transcript_files)} transcripts matching '{TRANSCRIPT_PATTERN}'")

    # Load conversations
    conversations: list[tuple[str, Any]] = []
    for f in transcript_files:
        conv_id = f.stem  # e.g., "transcript_5000_20251229_001141"
        conv = load_conversation_from_file(f)
        conversations.append((conv_id, conv))

    # Ensure checkpoint directory exists
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Run batch assessment with checkpointing
    await assess_batch(
        conversations,
        checkpoint_path=CHECKPOINT_PATH,
        concurrency=1,  # Sequential to avoid rate limits
        log_interval=1,
    )

    # Load all results and print summary
    results = load_checkpoint_results(CHECKPOINT_PATH)
    print_summary(results)
    save_summary(results)


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    """Print assessment summary to console."""
    print(f"\n{'=' * 60}")
    print("ASSESSMENT SUMMARY")
    print(f"{'=' * 60}")

    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    passed = sum(1 for r in results.values() if r.get("pass"))
    failed = total - passed

    print(f"Total assessed: {total}")
    print(f"Passed: {passed} ({100 * passed / total:.1f}%)")
    print(f"Failed: {failed} ({100 * failed / total:.1f}%)")

    # Category score averages
    category_totals: dict[str, list[float]] = {}
    for r in results.values():
        for cat, score in r.get("category_scores", {}).items():
            if cat not in category_totals:
                category_totals[cat] = []
            category_totals[cat].append(score)

    print("\nCategory Score Averages:")
    for cat, scores in sorted(category_totals.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat}: {avg:.3f}")

    # Failed criteria breakdown
    failed_counts: dict[str, int] = {}
    for r in results.values():
        for cid in r.get("failed_checks", []):
            failed_counts[cid] = failed_counts.get(cid, 0) + 1

    if failed_counts:
        print("\nMost Common Failures:")
        for cid, count in sorted(failed_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cid}: {count} failures")

    # Safety gate failures
    safety_failures = [
        r["conversation_id"] for r in results.values() if r.get("safety_gate_failed")
    ]
    if safety_failures:
        print(f"\nSafety Gate Failures ({len(safety_failures)}):")
        for cid in safety_failures:
            print(f"  - {cid}")


def save_summary(results: dict[str, dict[str, Any]]) -> None:
    """Save summary to JSON file."""
    total = len(results)
    if total == 0:
        return

    passed = sum(1 for r in results.values() if r.get("pass"))

    # Category averages
    category_totals: dict[str, list[float]] = {}
    for r in results.values():
        for cat, score in r.get("category_scores", {}).items():
            if cat not in category_totals:
                category_totals[cat] = []
            category_totals[cat].append(score)

    # Failed criteria counts
    failed_counts: dict[str, int] = {}
    for r in results.values():
        for cid in r.get("failed_checks", []):
            failed_counts[cid] = failed_counts.get(cid, 0) + 1

    # Safety failures
    safety_failures = [
        r["conversation_id"] for r in results.values() if r.get("safety_gate_failed")
    ]

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total,
        "category_averages": {
            cat: sum(scores) / len(scores) for cat, scores in category_totals.items()
        },
        "failed_criteria_counts": failed_counts,
        "safety_failures": safety_failures,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
