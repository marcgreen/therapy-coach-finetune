"""Batch assessment script for 5000-series transcripts."""

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


async def main() -> None:
    setup_logging()

    # Initialize Google backend
    get_backend(backend_type="google")

    # Find all 5000-series transcripts
    transcript_dir = Path("data/raw/transcripts/short")
    transcript_files = sorted(transcript_dir.glob("transcript_5*.json"))

    print(f"Found {len(transcript_files)} transcripts to assess")

    # Load conversations
    conversations: list[tuple[str, Any]] = []
    for f in transcript_files:
        conv_id = f.stem  # e.g., "transcript_5000_20251229_001141"
        conv = load_conversation_from_file(f)
        conversations.append((conv_id, conv))

    # Run batch assessment with checkpointing
    checkpoint_path = Path("data/assessments/short_5000_checkpoint.jsonl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    await assess_batch(
        conversations,
        checkpoint_path=checkpoint_path,
        concurrency=1,  # 1 at a time for API rate limits
        log_interval=1,
    )

    # Load all results and summarize
    results = load_checkpoint_results(checkpoint_path)

    print(f"\n{'=' * 60}")
    print("ASSESSMENT SUMMARY")
    print(f"{'=' * 60}")

    total = len(results)
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

    # Save summary to file
    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "category_averages": {
            cat: sum(scores) / len(scores) for cat, scores in category_totals.items()
        },
        "failed_criteria_counts": failed_counts,
        "safety_failures": safety_failures,
    }

    summary_path = Path("data/assessments/short_5000_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
