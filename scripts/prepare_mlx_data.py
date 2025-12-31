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

# Split ratios
TRAIN_RATIO = 0.9
VALID_RATIO = 0.05
TEST_RATIO = 0.05

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

    # Shuffle for random split
    random.shuffle(examples)

    # Calculate split indices
    n = len(examples)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    train_data = examples[:train_end]
    valid_data = examples[train_end:valid_end]
    test_data = examples[valid_end:]

    print(
        f"Split: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test"
    )

    # Write files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("train", train_data),
        ("valid", valid_data),
        ("test", test_data),
    ]:
        output_file = OUTPUT_DIR / f"{name}.jsonl"
        with open(output_file, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {output_file}")

    print(f"\nData ready for MLX training at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
