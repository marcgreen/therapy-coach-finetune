#!/usr/bin/env python3
"""Prepare training data for MLX-LM fine-tuning.

MLX-LM expects a directory with train.jsonl, valid.jsonl, test.jsonl files.
Our data is already in the correct chat format: {"messages": [...]}
"""

import json
import random
from pathlib import Path


INPUT_FILE = Path("data/processed/training_examples.jsonl")
OUTPUT_DIR = Path("data/processed/mlx_training")

# For validation run: use ALL data for training
# No valid/test split needed - we evaluate on new personas, not held-out data

# Reproducibility
SEED = 42


def main() -> None:
    random.seed(SEED)

    # Load all examples
    examples = []
    with open(INPUT_FILE) as f:
        for line in f:
            data = json.loads(line)
            # Keep only 'messages' field (MLX format)
            examples.append({"messages": data["messages"]})

    print(f"Loaded {len(examples)} examples")

    # Shuffle for reproducibility (affects order during training)
    random.shuffle(examples)

    print(f"Using all {len(examples)} examples for training")

    # Write files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "train.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {output_file}")

    print(f"\nData ready for MLX training at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
