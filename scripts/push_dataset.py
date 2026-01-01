#!/usr/bin/env python3
"""Push training data to HuggingFace Hub."""

from pathlib import Path

from datasets import Dataset


TRAINING_DATA = Path("data/processed/training_examples.jsonl")
REPO_ID = "marcgreen/therapeutic-coaching-sft"


def main() -> None:
    # Load dataset
    print(f"Loading {TRAINING_DATA}...")
    dataset = Dataset.from_json(str(TRAINING_DATA))
    print(f"Loaded {len(dataset)} examples")

    # Remove metadata columns (keep only 'messages' for training)
    columns_to_remove = [c for c in dataset.column_names if c != "messages"]
    if columns_to_remove:
        print(f"Removing metadata columns: {columns_to_remove}")
        dataset = dataset.remove_columns(columns_to_remove)

    print(f"Columns: {dataset.column_names}")
    print(f"Sample messages count: {len(dataset[0]['messages'])}")

    # Push to Hub
    print(f"\nPushing to {REPO_ID}...")
    dataset.push_to_hub(
        REPO_ID,
        private=True,
        commit_message=f"Upload therapeutic coaching SFT training data ({len(dataset)} examples)",
    )

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
