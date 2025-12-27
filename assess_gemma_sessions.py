"""Batch assess all Gemma session transcripts."""

import asyncio
import json
from pathlib import Path

from assessor import (
    ConversationInput,
    assess_batch,
    load_conversation_from_file,
    setup_logging,
)


async def main() -> None:
    setup_logging()

    # Find all Gemma session files
    transcript_dir = Path("data/raw/transcripts")
    gemma_files = sorted(transcript_dir.glob("gemma_session_*.json"))

    print(f"Found {len(gemma_files)} Gemma session files")

    # Load conversations
    conversations: list[tuple[str, ConversationInput]] = []
    for file_path in gemma_files:
        session_id = file_path.stem  # e.g., gemma_session_1000_20251227_132426
        conv = load_conversation_from_file(file_path)
        conversations.append((session_id, conv))

    # Run batch assessment with checkpointing
    checkpoint_path = Path("output/gemma_assessments_checkpoint.jsonl")
    checkpoint_path.parent.mkdir(exist_ok=True)

    print(f"Running assessments (checkpoint: {checkpoint_path})...")
    await assess_batch(
        conversations,
        checkpoint_path=checkpoint_path,
        concurrency=3,  # Claude CLI is sequential, but some parallelism helps
        log_interval=5,
    )

    # Load all results (including any from checkpoint)
    from assessor import load_checkpoint_results

    all_results = load_checkpoint_results(checkpoint_path)

    # Calculate summary stats
    output_path = Path("output/gemma_assessments_summary.json")
    total = len(gemma_files)
    passed = sum(1 for r in all_results.values() if r.get("pass", False))
    failed = sum(1 for r in all_results.values() if not r.get("pass", False))
    avg_score = (
        sum(r.get("score", 0) for r in all_results.values()) / len(all_results)
        if all_results
        else 0.0
    )

    # Build results list sorted by session ID
    results_list: list[dict[str, object]] = []
    for session_id in sorted(all_results.keys()):
        result = all_results[session_id]
        results_list.append(
            {
                "session_id": session_id,
                "passed": result.get("pass", False),
                "score": result.get("score", 0),
                "category_scores": result.get("category_scores", {}),
                "failed_checks": result.get("failed_checks", []),
                "reasonings": result.get("reasonings", {}),
            }
        )

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "avg_score": avg_score,
        "results": results_list,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("GEMMA SESSION ASSESSMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total sessions: {total}")
    print(f"Passed: {passed}/{total} ({100 * passed / total:.1f}%)")
    print(f"Average score: {avg_score:.3f}")

    print(f"\n{'─' * 60}")
    print("PER-SESSION RESULTS")
    print(f"{'─' * 60}")
    for r in results_list:
        r_passed = r["passed"]
        r_score = r["score"]
        r_failed = r["failed_checks"]
        r_session = r["session_id"]
        status = "PASS" if r_passed else "FAIL"
        failed_str = ", ".join(r_failed) if r_failed else "none"  # type: ignore[arg-type]
        print(f"  {r_session}: {status} ({r_score:.3f}) - Failed: {failed_str}")  # type: ignore[str-format]

    print(f"\nFull results saved to: {output_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    asyncio.run(main())
