#!/usr/bin/env python3
"""Slice transcripts into training examples with random slice points."""

import hashlib
import json
import random
from pathlib import Path

import tiktoken

# Tokenizer for accurate counting
_tokenizer = tiktoken.get_encoding("cl100k_base")

PASSING_MANIFEST = Path("data/processed/passing_transcripts.json")
SYSTEM_PROMPT_FILE = Path("config/system-prompt.md")
OUTPUT = Path("data/processed/training_examples.jsonl")

# Token limit (with buffer for 128K context window)
MAX_TOKENS = 120_000

# Slicing bounds
MIN_CONTEXT = 3  # First slice at exchange 3 minimum
MIN_GAP = 2  # At least 2 exchanges between slices
MAX_GAP = 5  # At most 5 exchanges between slices


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(_tokenizer.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """Count tokens in a messages array."""
    # Include role tokens and formatting overhead (~4 tokens per message)
    total = 0
    for msg in messages:
        total += count_tokens(msg["content"]) + 4
    return total


def get_system_prompt() -> str:
    """Extract system prompt from config file."""
    content = SYSTEM_PROMPT_FILE.read_text()
    lines = content.split("\n")
    in_prompt = False
    prompt_lines = []
    for line in lines:
        if line.strip() == "```" and not in_prompt:
            in_prompt = True
            continue
        if line.strip() == "```" and in_prompt:
            break
        if in_prompt:
            prompt_lines.append(line)
    return "\n".join(prompt_lines).strip()


def get_slice_points(total_turns: int, transcript_id: str) -> list[int]:
    """Generate random slice points with min/max gap constraints.

    Uses uniform random gaps between MIN_GAP and MAX_GAP.
    Seeded by transcript ID for reproducibility.
    """
    # Stable seed from transcript ID
    seed = int(hashlib.sha256(transcript_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    points = []
    current = MIN_CONTEXT  # Start at minimum context

    while current <= total_turns:
        points.append(current)
        # Random gap between min and max
        gap = rng.randint(MIN_GAP, MAX_GAP)
        current += gap

    # Always include final turn
    if points[-1] != total_turns:
        points.append(total_turns)

    return points


def find_max_exchanges_under_limit(
    exchanges: list[dict],
    system_prompt: str,
    max_tokens: int,
) -> int:
    """Find maximum number of recent exchanges that fit under token limit.

    Returns the count of exchanges (from the end) that fit.
    Used for truncating long transcripts.
    """
    system_tokens = count_tokens(system_prompt) + 4

    # Start from the end and work backwards
    total_tokens = system_tokens
    count = 0

    for ex in reversed(exchanges):
        ex_tokens = count_tokens(ex["user"]) + count_tokens(ex["assistant"]) + 8
        if total_tokens + ex_tokens > max_tokens:
            break
        total_tokens += ex_tokens
        count += 1

    return count


def slice_transcript(
    transcript: dict,
    system_prompt: str,
    stats: dict,
) -> list[dict]:
    """Slice a transcript into multiple training examples.

    For transcripts exceeding MAX_TOKENS, truncates to most recent
    exchanges that fit within the limit.
    """
    exchanges = transcript.get("exchanges", transcript.get("conversations", []))
    total_turns = len(exchanges)
    transcript_id = transcript.get("id", "unknown")

    if total_turns < MIN_CONTEXT:
        return []  # Too short to slice

    points = get_slice_points(total_turns, transcript_id)
    examples = []

    for point in points:
        # Build messages for this slice point
        messages = [{"role": "system", "content": system_prompt}]

        for ex in exchanges[:point]:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})

        # Check token count
        tokens = count_messages_tokens(messages)

        if tokens > MAX_TOKENS:
            # Truncate: find how many recent exchanges fit
            max_ex = find_max_exchanges_under_limit(
                exchanges[:point], system_prompt, MAX_TOKENS
            )

            if max_ex < MIN_CONTEXT:
                stats["skipped_too_long"] += 1
                continue  # Can't fit minimum context

            # Rebuild with truncated context (most recent exchanges)
            start_idx = point - max_ex
            messages = [{"role": "system", "content": system_prompt}]
            for ex in exchanges[start_idx:point]:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})

            tokens = count_messages_tokens(messages)
            stats["truncated"] += 1
            stats["truncation_log"].append(
                {
                    "transcript": transcript_id,
                    "slice_point": point,
                    "original_exchanges": point,
                    "truncated_to": max_ex,
                    "tokens": tokens,
                }
            )

        examples.append(
            {
                "messages": messages,
                "source_transcript": transcript_id,
                "slice_point": point,
                "tokens": tokens,
            }
        )

    return examples


def main() -> None:
    system_prompt = get_system_prompt()
    system_tokens = count_tokens(system_prompt)
    print(f"System prompt: {len(system_prompt)} chars, {system_tokens} tokens")

    with open(PASSING_MANIFEST) as f:
        manifest = json.load(f)

    stats: dict = {
        "skipped_too_long": 0,
        "truncated": 0,
        "truncation_log": [],
    }
    all_examples: list[dict] = []

    for item in manifest["transcripts"]:
        # Find transcript file
        source = item.get("source_file")
        if source:
            path = Path(source)
        else:
            print(f"Warning: No source_file for {item['id']}")
            continue

        if not path.exists():
            print(f"Warning: {path} not found")
            continue

        with open(path) as f:
            transcript = json.load(f)

        examples = slice_transcript(transcript, system_prompt, stats)
        before = len(all_examples)
        all_examples.extend(examples)
        added = len(all_examples) - before

        exchanges = transcript.get("exchanges", [])
        print(f"  {item['id']}: {len(exchanges)} turns -> {added} examples")

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for ex in all_examples:
            # Remove tokens field before writing (metadata only)
            output = {k: v for k, v in ex.items() if k != "tokens"}
            f.write(json.dumps(output) + "\n")

    # Summary
    print(f"\n{'=' * 60}")
    print("SLICING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total training examples: {len(all_examples)}")
    print(f"Skipped (too long even after truncation): {stats['skipped_too_long']}")
    print(f"Truncated (fit after removing early context): {stats['truncated']}")

    if stats["truncation_log"]:
        print("\nTruncation details:")
        for entry in stats["truncation_log"]:
            print(
                f"  {entry['transcript']} slice {entry['slice_point']}: "
                f"{entry['original_exchanges']} -> {entry['truncated_to']} exchanges "
                f"({entry['tokens']:,} tokens)"
            )

    # Token distribution
    token_counts = [ex["tokens"] for ex in all_examples]
    print("\nToken distribution:")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Avg: {sum(token_counts) // len(token_counts):,}")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
