# E2E Fine-tuning Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the complete fine-tuning pipeline end-to-end using existing transcripts, from assessment through GGUF export and evaluation.

**Architecture:** Multi-phase pipeline: (1) assess remaining transcripts, (2) slice passing transcripts into training examples, (3) push to HuggingFace Hub, (4) train with TRL on HF Jobs, (5) convert to GGUF, (6) evaluate on new personas. All scripts are reusable for full-scale training.

**Tech Stack:** Python 3.12, uv, existing assessor.py, HuggingFace datasets/hub, TRL via HF Jobs (model-trainer skill), llama.cpp for GGUF

---

## Current State

| Category | Count | Assessed | Passing |
|----------|-------|----------|---------|
| Short (5000-series) | 29 | 29 | 24 |
| Short (0000-series) | 5 | 0 | ? |
| Medium | 3 | 0 | ? |
| Long | 1 | 0 | ? |
| Very Long | 3 | 0 | ? |
| **Total** | **41** | **29** | **24+** |

**Target:** ~200-300 training examples from ~30 passing transcripts

---

## Task 1: Assess Remaining Transcripts

**Files:**
- Create: `scripts/assess_remaining.py` ✅ DONE
- Output: `data/assessments/remaining_batch_checkpoint.jsonl`

**Status:** Script created, needs to be run.

### Implementation

Uses existing `assess_batch()` from `assessor.py` for proper checkpointing/error handling:

**File:** `scripts/assess_remaining.py`

```python
#!/usr/bin/env python3
"""Assess all remaining unassessed transcripts using assess_batch."""

import asyncio
from pathlib import Path

from assessor import (
    ConversationInput,
    assess_batch,
    get_backend,
    load_checkpoint,
    load_conversation_from_file,
    setup_logging,
)

TRANSCRIPT_DIRS = [
    Path("data/raw/transcripts/short"),
    Path("data/raw/transcripts/medium"),
    Path("data/raw/transcripts/long"),
    Path("data/raw/transcripts/very_long"),
]

PRIOR_CHECKPOINT = Path("data/assessments/short_5000_checkpoint.jsonl")
OUTPUT_CHECKPOINT = Path("data/assessments/remaining_batch_checkpoint.jsonl")


def find_all_transcripts() -> list[tuple[str, Path]]:
    """Find all transcripts across all directories."""
    transcripts = []
    for dir_path in TRANSCRIPT_DIRS:
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.json"):
            transcripts.append((f.stem, f))
    return sorted(transcripts)


async def main() -> None:
    setup_logging()
    get_backend("google")

    prior_ids = load_checkpoint(PRIOR_CHECKPOINT) if PRIOR_CHECKPOINT.exists() else set()
    all_transcripts = find_all_transcripts()
    to_assess = [(tid, path) for tid, path in all_transcripts if tid not in prior_ids]

    print(f"Found {len(to_assess)} transcripts to assess ({len(prior_ids)} already done)")

    if not to_assess:
        print("Nothing to assess!")
        return

    conversations: list[tuple[str, ConversationInput]] = []
    for tid, path in to_assess:
        conv = load_conversation_from_file(path)
        conversations.append((tid, conv))

    await assess_batch(
        conversations,
        checkpoint_path=OUTPUT_CHECKPOINT,
        concurrency=1,
    )

    print(f"\nResults saved to {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Step: Run the assessment

```bash
uv run python scripts/assess_remaining.py
```

Expected: Assesses ~12 transcripts (5 short + 3 medium + 1 long + 3 very_long)

### Step: Review results

```bash
cat data/assessments/remaining_batch_checkpoint.jsonl | jq -s '{
  total: length,
  passed: [.[] | select(.pass)] | length,
  failed: [.[] | select(.pass | not)] | length
}'
```

---

## Task 2: Gather Passing Transcripts

**Files:**
- Create: `scripts/gather_passing.py`
- Output: `data/processed/passing_transcripts.json` (manifest file)

### Step 1: Create script to gather all passing transcript paths

**File:** `scripts/gather_passing.py`

```python
#!/usr/bin/env python3
"""Gather paths to all passing transcripts."""

import json
from pathlib import Path


CHECKPOINTS = [
    Path("data/assessments/short_5000_checkpoint.jsonl"),
    Path("data/assessments/remaining_batch_checkpoint.jsonl"),
]

OUTPUT = Path("data/processed/passing_transcripts.json")


def main():
    passing = []
    seen_ids = set()

    for checkpoint in CHECKPOINTS:
        if not checkpoint.exists():
            print(f"Warning: {checkpoint} not found, skipping")
            continue

        with open(checkpoint) as f:
            for line in f:
                data = json.loads(line)
                conv_id = data["conversation_id"]

                # Skip duplicates
                if conv_id in seen_ids:
                    continue
                seen_ids.add(conv_id)

                if data.get("pass", False):
                    passing.append({
                        "id": conv_id,
                        "source_file": data.get("source_file"),
                        "score": data.get("score"),
                    })

    # Sort by ID for reproducibility
    passing.sort(key=lambda x: x["id"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({
            "count": len(passing),
            "transcripts": passing,
        }, f, indent=2)

    print(f"Found {len(passing)} passing transcripts")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
```

### Step 2: Run and verify

```bash
uv run python scripts/gather_passing.py
cat data/processed/passing_transcripts.json | jq '.count'
```

Expected: 30+ passing transcripts

### Step 3: Commit

```bash
git add scripts/gather_passing.py data/processed/passing_transcripts.json
git commit -m "feat: gather passing transcripts for training"
```

---

## Task 3: Slice Transcripts into Training Examples

**Files:**
- Create: `scripts/slice_transcripts.py`
- Output: `data/processed/training_examples.jsonl`

### Slicing Strategy

**Random density with bounds** (not dense-at-end):
- `min_context = 3`: First slice at exchange 3 minimum
- `min_gap = 2`: At least 2 exchanges between slices
- `max_gap = 5`: At most 5 exchanges between slices (ensures coverage)
- `include_final = True`: Always include last exchange
- Seeded by transcript ID for reproducibility

**Rationale:** Random provides uniform coverage of early and late conversation dynamics. Dense-at-end would bias toward late-conversation style. With only ~24 transcripts, we need balanced representation.

**Token handling:**
- Use tiktoken (cl100k_base) for accurate token counts
- Max 120K tokens per training example (8K buffer for 128K context)
- For transcripts exceeding limit: truncate to most recent N exchanges that fit

### Step 1: Create slicing script

**File:** `scripts/slice_transcripts.py`

```python
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
MIN_GAP = 2      # At least 2 exchanges between slices
MAX_GAP = 5      # At most 5 exchanges between slices


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
    transcript_id = transcript.get("transcript_id", transcript.get("id", "unknown"))

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
            stats["truncation_log"].append({
                "transcript": transcript_id,
                "slice_point": point,
                "original_exchanges": point,
                "truncated_to": max_ex,
                "tokens": tokens,
            })

        examples.append({
            "messages": messages,
            "source_transcript": transcript_id,
            "slice_point": point,
            "tokens": tokens,
        })

    return examples


def main():
    system_prompt = get_system_prompt()
    system_tokens = count_tokens(system_prompt)
    print(f"System prompt: {len(system_prompt)} chars, {system_tokens} tokens")

    with open(PASSING_MANIFEST) as f:
        manifest = json.load(f)

    stats = {
        "skipped_too_long": 0,
        "truncated": 0,
        "truncation_log": [],
    }
    all_examples = []

    for item in manifest["transcripts"]:
        # Find transcript file
        source = item.get("source_file")
        if source:
            path = Path(source)
        else:
            # Try to find by ID
            for subdir in ["short", "medium", "long", "very_long"]:
                candidate = Path(f"data/raw/transcripts/{subdir}/{item['id']}.json")
                if candidate.exists():
                    path = candidate
                    break
            else:
                print(f"Warning: Could not find transcript {item['id']}")
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

        exchanges = transcript.get("exchanges", transcript.get("conversations", []))
        print(f"  {item['id']}: {len(exchanges)} turns -> {added} examples")

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for ex in all_examples:
            # Remove tokens field before writing (metadata only)
            output = {k: v for k, v in ex.items() if k != "tokens"}
            f.write(json.dumps(output) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("SLICING SUMMARY")
    print(f"{'='*60}")
    print(f"Total training examples: {len(all_examples)}")
    print(f"Skipped (too long even after truncation): {stats['skipped_too_long']}")
    print(f"Truncated (fit after removing early context): {stats['truncated']}")

    if stats["truncation_log"]:
        print(f"\nTruncation details:")
        for entry in stats["truncation_log"]:
            print(f"  {entry['transcript']} slice {entry['slice_point']}: "
                  f"{entry['original_exchanges']} -> {entry['truncated_to']} exchanges "
                  f"({entry['tokens']:,} tokens)")

    # Token distribution
    token_counts = [ex["tokens"] for ex in all_examples]
    print(f"\nToken distribution:")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Avg: {sum(token_counts)//len(token_counts):,}")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
```

### Step 2: Run slicing

```bash
uv run python scripts/slice_transcripts.py
```

Expected output:
- ~200-250 training examples (24 transcripts × ~8-10 slices each)
- Token distribution: 5K-40K tokens per example
- Truncation log if any transcripts exceed 120K at late slice points

### Step 3: Verify output format

```bash
head -1 data/processed/training_examples.jsonl | jq '.messages | length'
head -1 data/processed/training_examples.jsonl | jq '.messages[0]'
wc -l data/processed/training_examples.jsonl
```

Expected: Messages array with system/user/assistant roles

### Step 4: Commit

```bash
git add scripts/slice_transcripts.py data/processed/training_examples.jsonl
git commit -m "feat: slice transcripts into training examples"
```

---

## Task 4: Push Dataset to HuggingFace Hub

**Files:**
- Create: `scripts/push_dataset.py`

### Step 1: Create push script

**File:** `scripts/push_dataset.py`

```python
#!/usr/bin/env python3
"""Push training data to HuggingFace Hub."""

import os
from pathlib import Path

from datasets import Dataset


TRAINING_DATA = Path("data/processed/training_examples.jsonl")
REPO_ID = "marcgreen/therapeutic-coaching-sft"  # Change to your username


def main():
    # Verify HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set")
        print("Run: export HF_TOKEN=your_token")
        return

    # Load dataset
    print(f"Loading {TRAINING_DATA}...")
    dataset = Dataset.from_json(str(TRAINING_DATA))
    print(f"Loaded {len(dataset)} examples")

    # Remove metadata columns (keep only 'messages' for training)
    columns_to_remove = [c for c in dataset.column_names if c != "messages"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    print(f"Columns: {dataset.column_names}")
    print(f"Sample: {dataset[0]}")

    # Push to Hub
    print(f"\nPushing to {REPO_ID}...")
    dataset.push_to_hub(
        REPO_ID,
        private=True,
        commit_message="Upload therapeutic coaching SFT training data",
    )

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
```

### Step 2: Set HF_TOKEN and run

```bash
export HF_TOKEN=your_write_token_here
uv run python scripts/push_dataset.py
```

Expected: Dataset pushed to HuggingFace Hub

### Step 3: Verify on Hub

Visit: `https://huggingface.co/datasets/marcgreen/therapeutic-coaching-sft`

### Step 4: Commit

```bash
git add scripts/push_dataset.py
git commit -m "feat: add script to push dataset to HuggingFace Hub"
```

---

## Task 5: Train Model with HuggingFace Jobs

**Files:**
- Create: `scripts/train_therapeutic_model.py` (training script for HF Jobs)
- Create: `scripts/submit_training_job.py` (job submission helper)

### Step 1: Create training script

This script will be submitted to HF Jobs. It uses TRL's SFTTrainer with LoRA.

**File:** `scripts/train_therapeutic_model.py`

```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "datasets>=2.14.0",
#     "trackio",
#     "bitsandbytes>=0.41.0",
#     "accelerate>=0.25.0",
# ]
# ///
"""
Therapeutic coaching model fine-tuning with QLoRA.

Submitted via HuggingFace Jobs (model-trainer skill).
"""

import os

import trackio
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


# Configuration
BASE_MODEL = "google/gemma-2-9b-it"  # Using Gemma 2 9B (more accessible than 12B)
DATASET_ID = "marcgreen/therapeutic-coaching-sft"  # Your dataset
OUTPUT_REPO = "marcgreen/therapeutic-gemma-9b"  # Output model

# LoRA configuration
LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Training configuration
TRAINING_CONFIG = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_private_repo=True,
    report_to="trackio",
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=8192,  # Reasonable context for training
)


def main():
    # Initialize tracking
    trackio.init(
        project="therapeutic-coaching",
        run_name="e2e-test-run",
        config={
            "base_model": BASE_MODEL,
            "dataset": DATASET_ID,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "epochs": TRAINING_CONFIG.num_train_epochs,
            "learning_rate": TRAINING_CONFIG.learning_rate,
        },
    )

    # Load dataset
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # Load tokenizer (needed for chat template)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create trainer
    print(f"Initializing trainer with base model: {BASE_MODEL}")
    trainer = SFTTrainer(
        model=BASE_MODEL,
        train_dataset=dataset,
        peft_config=LORA_CONFIG,
        args=TRAINING_CONFIG,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save and push
    print("Saving model...")
    trainer.save_model()
    trainer.push_to_hub()

    trackio.finish()
    print(f"Training complete! Model saved to: {OUTPUT_REPO}")


if __name__ == "__main__":
    main()
```

### Step 2: Submit training job

Use the model-trainer skill's `hf_jobs()` MCP tool. The script content is passed inline:

```python
# In Claude Code, invoke:
# hf_jobs("uv", {
#     "script": <contents of train_therapeutic_model.py>,
#     "flavor": "a10g-large",
#     "timeout": "4h",
#     "secrets": {"HF_TOKEN": "$HF_TOKEN"}
# })
```

For manual submission, read the script and call:

```bash
# The model-trainer skill handles this via hf_jobs() MCP tool
# See Task 5 Step 3 for the actual submission
```

### Step 3: Submit via model-trainer skill

When ready to train, invoke in Claude Code:

```
Use the model-trainer skill to submit a training job:
- Script: contents of scripts/train_therapeutic_model.py
- Hardware: a10g-large
- Timeout: 4h
- Push to Hub: enabled
```

Expected output:
- Job ID and monitoring URL
- Training runs for ~2-3 hours
- Model pushed to `marcgreen/therapeutic-gemma-9b`

### Step 4: Monitor training

Check Trackio dashboard for loss curves. Training loss should:
- Start high (~2-3)
- Decrease over epochs
- Stabilize around epoch 2-3

### Step 5: Commit training script

```bash
git add scripts/train_therapeutic_model.py
git commit -m "feat: add therapeutic model training script for HF Jobs"
```

---

## Task 6: Convert to GGUF

**Files:**
- Create: `scripts/convert_to_gguf.py`

### Step 1: Create GGUF conversion script

This script is submitted as a follow-up HF Job after training completes.

**File:** `scripts/convert_to_gguf.py`

```python
# /// script
# dependencies = [
#     "huggingface_hub",
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "llama-cpp-python",
# ]
# ///
"""
Convert fine-tuned model to GGUF format.

Submitted via HuggingFace Jobs after training completes.
"""

import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configuration - set via environment variables
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "marcgreen/therapeutic-gemma-9b")
BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-2-9b-it")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "marcgreen/therapeutic-gemma-9b-gguf")
QUANTIZATION = os.environ.get("QUANTIZATION", "q4_k_m")


def main():
    work_dir = Path("./gguf_work")
    work_dir.mkdir(exist_ok=True)

    merged_dir = work_dir / "merged"
    gguf_dir = work_dir / "gguf"

    # Step 1: Download and merge adapter
    print(f"Downloading adapter from {ADAPTER_MODEL}...")
    adapter_path = snapshot_download(ADAPTER_MODEL)

    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    # Step 2: Convert to GGUF using llama.cpp
    print("Converting to GGUF...")
    gguf_dir.mkdir(exist_ok=True)

    # Clone llama.cpp for conversion script
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/ggerganov/llama.cpp.git",
        str(work_dir / "llama.cpp")
    ], check=True)

    # Run conversion
    output_file = gguf_dir / f"therapeutic-gemma-9b-{QUANTIZATION}.gguf"
    subprocess.run([
        "python", str(work_dir / "llama.cpp/convert_hf_to_gguf.py"),
        str(merged_dir),
        "--outfile", str(output_file),
        "--outtype", QUANTIZATION,
    ], check=True)

    print(f"GGUF file created: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1e9:.2f} GB")

    # Step 3: Upload to Hub
    print(f"Uploading to {OUTPUT_REPO}...")
    api = HfApi()
    api.create_repo(OUTPUT_REPO, repo_type="model", exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=str(output_file),
        path_in_repo=output_file.name,
        repo_id=OUTPUT_REPO,
    )

    print(f"Done! GGUF available at: https://huggingface.co/{OUTPUT_REPO}")


if __name__ == "__main__":
    main()
```

### Step 2: Submit GGUF conversion job

After training completes, submit via model-trainer skill:

```
Use the model-trainer skill to submit a GGUF conversion job:
- Script: contents of scripts/convert_to_gguf.py
- Hardware: a10g-large (needs memory for model loading)
- Timeout: 1h
- Environment variables:
  - ADAPTER_MODEL: marcgreen/therapeutic-gemma-9b
  - BASE_MODEL: google/gemma-2-9b-it
  - OUTPUT_REPO: marcgreen/therapeutic-gemma-9b-gguf
```

### Step 3: Download GGUF for local testing

```bash
huggingface-cli download marcgreen/therapeutic-gemma-9b-gguf \
    therapeutic-gemma-9b-q4_k_m.gguf \
    --local-dir ~/models/
```

### Step 4: Commit

```bash
git add scripts/convert_to_gguf.py
git commit -m "feat: add GGUF conversion script"
```

---

## Task 7: Evaluate Fine-tuned Model

**Files:**
- Create: `scripts/generate_eval_personas.py`
- Create: `scripts/run_evaluation.py`

### Step 1: Create evaluation persona generator

**File:** `scripts/generate_eval_personas.py`

```python
#!/usr/bin/env python3
"""Generate new personas for evaluation (not used in training)."""

import json
from pathlib import Path


OUTPUT = Path("data/evaluation/eval_personas.json")

# 3 diverse personas for e2e testing
EVAL_PERSONAS = [
    {
        "id": "eval_persona_001",
        "name": "Maya",
        "age_range": "28-35",
        "personality_traits": ["analytical", "reserved", "perfectionist"],
        "communication_style": "detailed",
        "topic_seeds": [
            {"category": "anxiety", "subtopic": "work_stress", "complexity": "medium"},
            {"category": "self_worth", "subtopic": "imposter", "complexity": "hard"},
        ],
        "trajectory": "gradual_opening",
    },
    {
        "id": "eval_persona_002",
        "name": "James",
        "age_range": "45-55",
        "personality_traits": ["pragmatic", "stoic", "caring"],
        "communication_style": "terse",
        "topic_seeds": [
            {"category": "relationships", "subtopic": "family", "complexity": "medium"},
            {"category": "life_transitions", "subtopic": "career_change", "complexity": "medium"},
        ],
        "trajectory": "resistant_then_engaged",
    },
    {
        "id": "eval_persona_003",
        "name": "Priya",
        "age_range": "22-28",
        "personality_traits": ["expressive", "anxious", "creative"],
        "communication_style": "emotional",
        "topic_seeds": [
            {"category": "emotional_regulation", "subtopic": "overwhelm", "complexity": "hard"},
            {"category": "relationships", "subtopic": "romantic", "complexity": "medium"},
        ],
        "trajectory": "volatile",
    },
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"personas": EVAL_PERSONAS}, f, indent=2)

    print(f"Created {len(EVAL_PERSONAS)} evaluation personas")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
```

### Step 2: Create evaluation script

**File:** `scripts/run_evaluation.py`

```python
#!/usr/bin/env python3
"""
Run full-conversation evaluation comparing base vs fine-tuned model.

Generates conversations with both models on the same personas,
assesses with rubric, computes statistical comparison.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from assessor import assess_transcript, setup_logging, get_backend
from llm_backend import LlamaServerBackend


EVAL_PERSONAS = Path("data/evaluation/eval_personas.json")
SYSTEM_PROMPT = Path("config/system-prompt.md")
OUTPUT_DIR = Path("data/evaluation/results")

# Model paths
BASE_MODEL_PATH = Path.home() / "models/gemma-3-12b-it-q4_0.gguf"
FINETUNED_MODEL_PATH = Path.home() / "models/therapeutic-gemma-9b-q4_k_m.gguf"

# Evaluation config
TRIALS_PER_PERSONA = 2
TARGET_TURNS = 20


def get_system_prompt() -> str:
    """Extract system prompt from config."""
    content = SYSTEM_PROMPT.read_text()
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


async def generate_conversation(
    model_backend,
    persona: dict,
    system_prompt: str,
    target_turns: int,
) -> dict:
    """Generate a full conversation with the given model."""
    # This is a simplified version - full implementation would use
    # the user simulator from transcript_generator.py

    exchanges = []
    history = []

    # For e2e test, use simple user messages based on persona
    user_messages = [
        f"Hi, I've been dealing with some {persona['topic_seeds'][0]['subtopic'].replace('_', ' ')} lately.",
    ]

    # Generate remaining user messages dynamically would require user simulator
    # For e2e, we'll do a simplified version

    for turn in range(min(target_turns, 5)):  # Limit for e2e
        if turn < len(user_messages):
            user_msg = user_messages[turn]
        else:
            user_msg = "Can you tell me more about that?"

        # Build context
        messages = [{"role": "system", "content": system_prompt}]
        for ex in exchanges:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_msg})

        # Generate response
        response = await model_backend.generate(messages)

        exchanges.append({
            "exchange_number": turn + 1,
            "user": user_msg,
            "assistant": response,
        })

    return {
        "id": f"eval_{persona['id']}",
        "persona": persona,
        "exchanges": exchanges,
    }


async def evaluate_model(
    model_path: Path,
    model_name: str,
    personas: list[dict],
    system_prompt: str,
) -> list[float]:
    """Evaluate a model on all personas."""
    print(f"\nEvaluating {model_name}...")

    # Start llama-server for this model
    # Note: In practice, you'd need to manage the server lifecycle
    backend = LlamaServerBackend(model_path=str(model_path))

    scores = []
    for persona in personas:
        for trial in range(TRIALS_PER_PERSONA):
            print(f"  {persona['name']} trial {trial + 1}...")

            transcript = await generate_conversation(
                backend, persona, system_prompt, TARGET_TURNS
            )

            # Assess
            result = await assess_transcript(transcript)
            scores.append(result.score)

            # Save transcript
            output_file = OUTPUT_DIR / f"{model_name}_{persona['id']}_trial{trial}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "transcript": transcript,
                    "assessment": {
                        "score": result.score,
                        "passed": result.passed,
                        "category_scores": result.category_scores,
                    },
                }, f, indent=2)

    return scores


def compare_models(base_scores: list[float], ft_scores: list[float]) -> dict:
    """Statistical comparison of model scores."""
    base_mean = np.mean(base_scores)
    ft_mean = np.mean(ft_scores)

    t_stat, p_value = ttest_rel(base_scores, ft_scores)

    return {
        "base_model": {
            "mean": float(base_mean),
            "std": float(np.std(base_scores)),
            "scores": base_scores,
        },
        "finetuned_model": {
            "mean": float(ft_mean),
            "std": float(np.std(ft_scores)),
            "scores": ft_scores,
        },
        "improvement": float(ft_mean - base_mean),
        "improvement_pct": float((ft_mean - base_mean) / base_mean * 100),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


async def main():
    setup_logging()
    get_backend("google")  # For assessment

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load personas
    with open(EVAL_PERSONAS) as f:
        personas = json.load(f)["personas"]

    system_prompt = get_system_prompt()

    # Evaluate both models
    base_scores = await evaluate_model(
        BASE_MODEL_PATH, "base", personas, system_prompt
    )
    ft_scores = await evaluate_model(
        FINETUNED_MODEL_PATH, "finetuned", personas, system_prompt
    )

    # Compare
    comparison = compare_models(base_scores, ft_scores)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nBase Model:       {comparison['base_model']['mean']:.3f} ± {comparison['base_model']['std']:.3f}")
    print(f"Fine-tuned Model: {comparison['finetuned_model']['mean']:.3f} ± {comparison['finetuned_model']['std']:.3f}")
    print(f"\nImprovement:      {comparison['improvement']:+.3f} ({comparison['improvement_pct']:+.1f}%)")
    print(f"p-value:          {comparison['p_value']:.4f}")
    print(f"Significant:      {'YES' if comparison['significant'] else 'NO'}")

    # Save results
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Run evaluation

```bash
# First, generate eval personas
uv run python scripts/generate_eval_personas.py

# Then run evaluation (requires both models available locally)
uv run python scripts/run_evaluation.py
```

### Step 4: Review results

```bash
cat data/evaluation/results/comparison.json | jq
```

### Step 5: Commit

```bash
git add scripts/generate_eval_personas.py scripts/run_evaluation.py
git add data/evaluation/
git commit -m "feat: add evaluation scripts for base vs fine-tuned comparison"
```

---

## Summary

### Artifacts Created

| Script | Purpose | Reusable? |
|--------|---------|-----------|
| `scripts/assess_remaining.py` | Assess unassessed transcripts | ✅ Yes |
| `scripts/gather_passing.py` | Collect passing transcript paths | ✅ Yes |
| `scripts/slice_transcripts.py` | Slice into training examples | ✅ Yes |
| `scripts/push_dataset.py` | Push to HuggingFace Hub | ✅ Yes |
| `scripts/train_therapeutic_model.py` | HF Jobs training script | ✅ Yes |
| `scripts/convert_to_gguf.py` | GGUF conversion | ✅ Yes |
| `scripts/generate_eval_personas.py` | Create eval personas | ✅ Yes |
| `scripts/run_evaluation.py` | Full-conversation evaluation | ✅ Yes |

### Data Flow

```
data/raw/transcripts/{short,medium,long,very_long}/
    ↓ (assess_remaining.py)
data/assessments/*_checkpoint.jsonl
    ↓ (gather_passing.py)
data/processed/passing_transcripts.json
    ↓ (slice_transcripts.py)
data/processed/training_examples.jsonl
    ↓ (push_dataset.py)
HuggingFace Hub: marcgreen/therapeutic-coaching-sft
    ↓ (train_therapeutic_model.py via HF Jobs)
HuggingFace Hub: marcgreen/therapeutic-gemma-9b
    ↓ (convert_to_gguf.py via HF Jobs)
HuggingFace Hub: marcgreen/therapeutic-gemma-9b-gguf
    ↓ (download to local)
~/models/therapeutic-gemma-9b-q4_k_m.gguf
    ↓ (run_evaluation.py)
data/evaluation/results/comparison.json
```

### Success Criteria

| Checkpoint | Criteria |
|------------|----------|
| Assessment | All transcripts assessed, ~30+ passing |
| Slicing | ~200-300 training examples generated |
| Dataset upload | Dataset visible on HuggingFace Hub |
| Training | Loss decreases and converges |
| GGUF export | File downloadable, loads in llama.cpp |
| Evaluation | Pipeline completes, comparison generated |

---

## Notes for Full-Scale Training

When scaling from e2e to full 155 transcripts:

1. **Assessment** — Same scripts work, just more transcripts
2. **Slicing** — Same script, expect ~1,200-1,500 examples (155 × ~8-10 slices)
3. **Training** — May want longer training (more epochs) or larger batch
4. **Evaluation** — Use 10 personas × 3 trials for statistical power
5. **Filtering pipeline** — Add Claude fixup for failing transcripts (not needed for e2e)

## Slicing Configuration Reference

```python
# Current bounds (can be tuned based on results)
MIN_CONTEXT = 3    # First slice at exchange 3 minimum
MIN_GAP = 2        # At least 2 exchanges between slices
MAX_GAP = 5        # At most 5 exchanges between slices
MAX_TOKENS = 120_000  # Token limit per training example
```

**Tuning guidance:**
- Increase MIN_GAP to reduce redundancy (fewer examples, more diverse)
- Decrease MAX_GAP to increase coverage (more examples, higher redundancy)
- For 155 transcripts, consider MIN_GAP=3 to keep examples under ~1,200
