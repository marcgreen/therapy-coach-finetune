#!/usr/bin/env python3
"""Local QLoRA fine-tuning using MLX (Apple Silicon).

Usage:
    uv run python scripts/train_local_mlx.py

Prerequisites:
    uv add mlx-lm

This script converts our training data to MLX format and runs LoRA fine-tuning.
"""

import json
import subprocess
from pathlib import Path

TRAINING_DATA = Path("data/processed/training_examples.jsonl")
MLX_DATA_DIR = Path("data/processed/mlx_format")
ADAPTER_OUTPUT = Path("adapters/therapeutic-gemma")

# Training config
MODEL = "mlx-community/gemma-2-9b-it-4bit"  # Use 4-bit quantized for memory
ITERS = 500  # Training iterations
BATCH_SIZE = 1
LORA_LAYERS = 16  # Number of layers to apply LoRA to


def convert_to_mlx_format() -> None:
    """Convert our JSONL to MLX's expected format.

    MLX expects either:
    1. A directory with train.jsonl, valid.jsonl, test.jsonl
    2. Each line: {"text": "full conversation as text"}

    For chat format, we need to convert messages to text.
    """
    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(TRAINING_DATA) as f:
        for line in f:
            data = json.loads(line)
            messages = data["messages"]

            # Convert to text format that MLX expects
            # Using Gemma's chat template
            text_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                elif role == "user":
                    text_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                elif role == "assistant":
                    text_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

            full_text = "\n".join(text_parts)
            examples.append({"text": full_text})

    # Split into train/valid (90/10)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    # Write files
    with open(MLX_DATA_DIR / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(MLX_DATA_DIR / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    print(
        f"Converted {len(train_examples)} train, {len(valid_examples)} valid examples"
    )
    print(f"Saved to {MLX_DATA_DIR}")


def run_training() -> None:
    """Run MLX LoRA fine-tuning."""
    ADAPTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "--model",
        MODEL,
        "--data",
        str(MLX_DATA_DIR),
        "--train",
        "--iters",
        str(ITERS),
        "--batch-size",
        str(BATCH_SIZE),
        "--lora-layers",
        str(LORA_LAYERS),
        "--adapter-path",
        str(ADAPTER_OUTPUT),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    print("Step 1: Converting data to MLX format...")
    convert_to_mlx_format()

    print("\nStep 2: Running LoRA fine-tuning...")
    print(f"Model: {MODEL}")
    print(f"Iterations: {ITERS}")
    print(f"Output: {ADAPTER_OUTPUT}")
    run_training()

    print(f"\nDone! Adapter saved to {ADAPTER_OUTPUT}")
    print("\nTo test the model:")
    print(
        f"  python -m mlx_lm.generate --model {MODEL} --adapter-path {ADAPTER_OUTPUT}"
    )


if __name__ == "__main__":
    main()
