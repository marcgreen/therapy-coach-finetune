"""Push training data to HuggingFace Hub."""

from datasets import Dataset
from pathlib import Path
import json


def load_training_data(path: Path) -> list[dict]:
    """Load JSONL training data."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main() -> None:
    data_path = Path("data/processed/training_examples.jsonl")
    repo_id = "marcgreen/therapeutic-coaching-v1"

    print(f"Loading training data from {data_path}...")
    examples = load_training_data(data_path)
    print(f"Loaded {len(examples)} examples")

    # Only keep 'messages' field for training
    # SFTTrainer expects just the messages
    training_data = [{"messages": ex["messages"]} for ex in examples]

    dataset = Dataset.from_list(training_data)
    print(f"Created dataset with {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")

    print(f"\nPushing to {repo_id}...")
    dataset.push_to_hub(repo_id, private=True)
    print("Done!")


if __name__ == "__main__":
    main()
