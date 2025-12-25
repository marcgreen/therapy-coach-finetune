#!/usr/bin/env python3
"""Read and display a conversation from generated_conversations.jsonl."""

import json
import sys
from pathlib import Path


def read_conversation(index: int, path: Path) -> None:
    """Print a conversation in readable format."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i == index:
                convo = json.loads(line)
                print(f"=== Conversation {convo.get('id', index)} ===\n")
                for msg in convo["messages"]:
                    role = msg["role"].upper()
                    content = msg["content"]
                    print(f"--- {role} ---")
                    print(content)
                    print()
                return
    print(f"Error: No conversation at index {index}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/read_convo.py <index> [path]")
        print("  index: 0-based conversation number")
        print("  path:  optional, defaults to output/generated_conversations.jsonl")
        sys.exit(1)

    idx = int(sys.argv[1])
    file_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("output/generated_conversations.jsonl")
    )
    read_conversation(idx, file_path)
