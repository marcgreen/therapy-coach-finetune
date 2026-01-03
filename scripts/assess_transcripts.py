"""
Assess transcripts using the 17-criteria rubric.

Runs assessment on transcripts in a directory, saving results alongside.
Supports resume - skips already-assessed transcripts.

Usage:
    # Assess all transcripts in a directory
    uv run python scripts/assess_transcripts.py data/eval/comparison/transcripts

    # Assess specific files
    uv run python scripts/assess_transcripts.py data/eval/comparison/transcripts/finetune_*.json

    # Force re-assessment (ignore existing)
    uv run python scripts/assess_transcripts.py data/eval/comparison/transcripts --force
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from assessor import (
    ConversationInput,
    assess_conversation,
    get_backend,
    setup_logging as setup_assessor_logging,
)


async def assess_transcript_file(
    transcript_path: Path,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict | None:
    """Assess a single transcript file.

    Returns assessment dict, or None if skipped.
    """
    # Determine output path
    if output_dir:
        assessment_path = output_dir / f"{transcript_path.stem}.json"
    else:
        # Put in assessments/ sibling to transcripts/
        assessment_path = (
            transcript_path.parent.parent
            / "assessments"
            / f"{transcript_path.stem}.json"
        )

    # Skip if exists (unless force)
    if assessment_path.exists() and not force:
        return None

    # Load transcript
    with open(transcript_path) as f:
        data = json.load(f)

    # Convert to messages format, checking for empty content
    messages = []
    for i, ex in enumerate(data.get("exchanges", [])):
        user_content = ex.get("user", "").strip()
        assistant_content = ex.get("assistant", "").strip()
        if not user_content or not assistant_content:
            print(f"  [SKIP] Empty message at exchange {i} in {transcript_path.name}")
            return None
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    if not messages:
        print(f"  [SKIP] No exchanges in {transcript_path.name}")
        return None

    # Assess
    conversation = ConversationInput.from_messages(messages)
    assessment = await assess_conversation(
        conversation,
        conversation_id=transcript_path.stem,
    )

    # Save
    assessment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(assessment_path, "w") as f:
        json.dump(assessment.to_dict(), f, indent=2)

    return assessment.to_dict()


async def assess_directory(
    input_path: Path,
    output_dir: Path | None = None,
    force: bool = False,
    pattern: str = "*.json",
) -> None:
    """Assess all transcripts in a directory."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob(pattern))

    if not files:
        print(f"No files found matching {input_path / pattern}")
        return

    # Initialize assessment backend
    get_backend(backend_type="google", model="gemini-3-flash-preview")

    total = len(files)
    assessed = 0
    skipped = 0
    scores = []

    print(f"\nAssessing {total} transcripts...")
    print(f"Output: {output_dir or 'assessments/ sibling directory'}\n")

    for i, path in enumerate(files, 1):
        result = await assess_transcript_file(path, output_dir, force)

        if result is None:
            skipped += 1
            print(f"[{i}/{total}] {path.name} -> SKIP (exists)")
        else:
            assessed += 1
            scores.append(result["score"])
            status = "PASS" if result["pass"] else "FAIL"
            safety = "SAFETY_FAIL" if result["safety_gate_failed"] else ""
            print(
                f"[{i}/{total}] {path.name} -> {result['score']:.3f} {status} {safety}"
            )

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Assessed: {assessed}, Skipped: {skipped}")
    if scores:
        import numpy as np

        print(f"Mean score: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
        print(f"Pass rate: {sum(1 for s in scores if s >= 0.8) / len(scores):.1%}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess transcripts using 17-criteria rubric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Assess all in directory
    uv run python scripts/assess_transcripts.py data/eval/comparison/transcripts

    # Assess with glob pattern
    uv run python scripts/assess_transcripts.py data/eval/comparison/transcripts --pattern "finetune_*.json"

    # Force re-assessment
    uv run python scripts/assess_transcripts.py data/eval/comparison/transcripts --force
""",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Directory containing transcripts, or single transcript file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for assessments (default: assessments/ sibling)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for transcript files (default: *.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-assess even if assessment exists",
    )

    args = parser.parse_args()

    import logging

    setup_assessor_logging(logging.INFO)

    asyncio.run(
        assess_directory(
            input_path=args.input,
            output_dir=args.output_dir,
            force=args.force,
            pattern=args.pattern,
        )
    )


if __name__ == "__main__":
    main()
