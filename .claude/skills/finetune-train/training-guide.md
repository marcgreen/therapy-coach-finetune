# Training Guide

How to run fine-tuning and evaluate improvement.

---

## Training Options

| Option | Best For | Cost | Setup |
|--------|----------|------|-------|
| **HuggingFace Jobs** | Fast iteration, serverless | ~$5-15/1K examples | Minimal |
| **MLX Local** | Apple Silicon, free | Free (time) | Moderate |
| **Cloud GPU** | Full control, large jobs | Varies | Complex |

**Recommendation:** Start with HuggingFace Jobs for speed. Switch to MLX for iteration once you're confident in data quality.

---

## HuggingFace Jobs

### Prerequisites

1. **HuggingFace Pro subscription** ($9/month) or Team/Enterprise
2. **Prepaid credits** for GPU time
3. **Logged in:** `hf auth login`
4. **Gated model access** (accept license for Gemma, Llama, etc.)

### GPU Selection Based on Context Length

**Critical: Vocabulary size dominates memory at long contexts.**

Logits are computed in FP32 regardless of quantization:
```
memory = vocab_size × sequence_length × 4 bytes
```

| Model | Vocab Size | Logits @ 2k | Logits @ 8k | Logits @ 16k |
|-------|-----------|-------------|-------------|--------------|
| **Gemma 3** | 262K | 2.1 GB | 8.6 GB | **17.2 GB** |
| **Qwen3** | 152K | 1.2 GB | 5.0 GB | **10.0 GB** |
| **Llama 3** | 128K | 1.0 GB | 4.2 GB | 8.4 GB |

**GPU Selection:**

| Context Length | Gemma 3 (262K vocab) | Qwen3/Llama (128-152K vocab) |
|---------------|----------------------|------------------------------|
| ≤2048 tokens | A10G (24GB) ~$1.50/hr | A10G (24GB) |
| ≤8192 tokens | **A100 (80GB)** ~$4/hr | A10G (24GB) |
| ≤16384 tokens | **A100 (80GB)** | **A100 (80GB)** |

**Rule of thumb:** Gemma 3 with 8k+ context → A100. Others can use A10G up to 8k.

### Known Issues

**Issue 1: CLI Syntax**

Flags MUST come BEFORE the script path:

```bash
# ✅ CORRECT
hf jobs uv run --flavor a100-large --secrets HF_TOKEN train.py

# ❌ WRONG: flags after script (will be ignored!)
hf jobs uv run train.py --flavor a100-large

# ❌ WRONG: --secret (singular)
hf jobs uv run --secret HF_TOKEN train.py

# ❌ WRONG: command order
hf jobs run uv train.py  # Should be "uv run"
```

**Issue 2: Token Placeholder**

Use `--secrets HF_TOKEN` (without value) to pass your logged-in token:

```bash
# ✅ CORRECT: passes your logged-in token
hf jobs uv run --secrets HF_TOKEN train.py

# ❌ WRONG: placeholder syntax (old docs)
secrets={"HF_TOKEN": "$HF_TOKEN"}
```

**Issue 3: Gemma 3 OOM (Large Vocabulary)**

Gemma 3 has 262K vocabulary. With default settings, you'll OOM.

**Solution:** Use appropriate GPU for your context length (see table above).

**Issue 4: SFTTrainer API**

`model_init_kwargs` goes in `SFTConfig`, not `SFTTrainer`:

```python
# ✅ CORRECT
config = SFTConfig(
    model_init_kwargs={
        "load_in_4bit": True,
        ...
    }
)
trainer = SFTTrainer(model="model-name", args=config, ...)
```

Pass model name string, not loaded model. Tokenizer auto-loads.

**Issue 5: Auto-Login Breaks Token Detection**

Don't call `login()`. The library auto-detects `HF_TOKEN` from environment:

```python
# ❌ WRONG
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

# ✅ CORRECT: Just use the library
dataset = load_dataset("username/private-dataset")
```

**Issue 6: Flash Attention Build Failure**

`flash-attn` requires torch during compilation. With `uv`, this fails.

**Solution:** Don't use flash-attn with HF Jobs. Standard attention works fine on A10G/A100.

---

## Trackio Integration

**Trackio** provides real-time monitoring for training on HF Jobs. It syncs metrics to a HuggingFace Space for visualization.

### Setup

1. **Add trackio dependency:**
```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "trackio",
# ]
# ///
```

2. **Initialize Trackio:**
```python
import trackio

trackio.init(
    project="my-project",
    name="descriptive-run-name",
    space_id="username/trackio",  # Auto-creates if doesn't exist
    config={
        "model": "google/gemma-3-12b-it",
        "dataset": "username/dataset",
        "max_length": 16384,
        "epochs": 3,
    },
)
```

3. **Configure TRL:**
```python
config = SFTConfig(
    report_to="trackio",
    # ... other config
)
```

4. **Finish tracking:**
```python
trainer.train()
trackio.finish()  # Ensures final metrics synced
```

### What Trackio Tracks

- Training loss
- Evaluation loss (if eval_dataset provided)
- Learning rate
- GPU utilization
- Memory usage

### Dashboard

View at: `https://huggingface.co/spaces/username/trackio`

---

## Working Script Template

Complete template with Trackio:

```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.51.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.45.0",
#     "datasets>=3.0.0",
#     "trackio",
#     "pillow",  # Required for Gemma 3
# ]
# ///
"""Training script for HuggingFace Jobs."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import trackio
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Config
MODEL_ID = "google/gemma-3-12b-it"
DATASET_ID = "username/dataset-name"
OUTPUT_REPO = "username/model-name"
MAX_LENGTH = 16384  # A100 required for Gemma 3 at this length

# Initialize Trackio
trackio.init(
    project="my-project",
    name="gemma3-12b-sft",
    space_id="username/trackio",
    config={
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "max_length": MAX_LENGTH,
        "epochs": 3,
    },
)

# Load dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Loaded {len(dataset)} examples")

# QLoRA config
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Training config
config = SFTConfig(
    output_dir="model-name",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="every_save",

    model_init_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "device_map": "auto",
    },

    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_length=MAX_LENGTH,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",

    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    report_to="trackio",
)

# Train
trainer = SFTTrainer(
    model=MODEL_ID,
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

print("Starting training...")
trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()

trackio.finish()
print(f"Complete! Model at: https://huggingface.co/{OUTPUT_REPO}")
```

### Job Submission

```bash
# CRITICAL: flags BEFORE script path
hf jobs uv run \
    --flavor a100-large \
    --timeout 6h \
    --secrets HF_TOKEN \
    scripts/train_model.py
```

### Monitoring

```bash
hf jobs ps                    # List jobs
hf jobs logs <job_id>         # View logs
hf jobs inspect <job_id>      # Job details
hf jobs cancel <job_id>       # Cancel job
```

**Expected loss progression:**
- Start: ~2-3
- Epoch 1: ~1.5-2
- Epoch 2: ~1.0-1.5
- Epoch 3: ~0.8-1.2 (stabilizing)

---

## Model Naming Conventions

Different model families use different naming for instruction-tuned variants:

| Family | Base Model | Instruction-Tuned | Use For SFT |
|--------|-----------|-------------------|-------------|
| **Gemma 3** | `google/gemma-3-12b` | `google/gemma-3-12b-it` | `-it` version |
| **Qwen3** | `Qwen/Qwen3-14B-Base` | `Qwen/Qwen3-14B` | Without `-Base` |
| **Llama 3** | `meta-llama/Llama-3-8B` | `meta-llama/Llama-3-8B-Instruct` | `-Instruct` version |

**Always use instruction-tuned for SFT** to preserve instruction-following capabilities.

---

## MLX Local Training

For Apple Silicon Macs — free but slower.

### Setup

```bash
pip install mlx-lm
```

### Training Command

```bash
python -m mlx_lm.lora \
    --model mlx-community/gemma-3-12b-it-8bit \
    --train \
    --data data/processed/mlx_training \
    --iters 800 \
    --batch-size 1 \
    --lora-layers 16
```

### Key Lessons

1. **mask_prompt=true is essential** — Only compute loss on assistant turns
2. **8-bit base model works** — Uses less memory
3. **Adapter format differs** — Need to convert for GGUF export
4. **Slower but free** — ~4-6 hours for 1K examples on M3 Max

---

## GGUF Conversion

Convert fine-tuned adapter to GGUF for local inference.

### Step 1: Download Adapter

```python
from huggingface_hub import snapshot_download
adapter_path = snapshot_download("username/therapeutic-model")
```

### Step 2: Merge Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    torch_dtype="auto",
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
merged = model.merge_and_unload()

merged.save_pretrained("./merged")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
tokenizer.save_pretrained("./merged")
```

### Step 3: Convert to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp

# Convert
python llama.cpp/convert_hf_to_gguf.py ./merged --outtype q4_k_m
```

### Step 4: Test Locally

```bash
# Run
llama-server -m merged-q4_k_m.gguf --port 8080 -ngl 99
```

---

## Evaluation Methodology

### Why Full-Conversation Evaluation?

| Method | What It Measures | Limitation |
|--------|-----------------|------------|
| Perplexity | Did training work? | Low perplexity ≠ good conversations |
| Single-turn | Response quality | Misses multi-turn dynamics |
| Benchmarks | General capability | Doesn't test your domain |
| **Full-conversation** | **Actual use case** | **Most rigorous** |

### Protocol

1. **Generate NEW personas** (10-15, not used in training)
2. **For each persona:** Generate 3 conversations with BOTH models
3. **Same user simulator** for both (controlled comparison)
4. **Assess all conversations** with your rubric
5. **Statistical comparison**

### Statistical Test

```python
from scipy import stats
import numpy as np

base_scores = [...]
finetuned_scores = [...]

t_stat, p_value = stats.ttest_rel(finetuned_scores, base_scores)

improvement = np.mean(finetuned_scores) - np.mean(base_scores)
improvement_pct = improvement / np.mean(base_scores) * 100

print(f"Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
print(f"p-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Absolute improvement | ≥0.10 (10 points) |
| Statistical significance | p < 0.05 |
| Safety regressions | None |

---

## Troubleshooting

```
Job fails immediately?
├── "No module named X" → Check dependencies in PEP 723 header
├── "403 Forbidden" → Accept gated model license, check token
└── Timeout before start → Increase timeout, check script syntax

Job fails during training?
├── "CUDA out of memory" → Use larger GPU or reduce max_length (see GPU table)
├── "TypeError: unexpected argument" → model_init_kwargs goes in SFTConfig
└── "KeyError" during login → Remove login() call

Training loss doesn't decrease?
├── Learning rate too high → Try 1e-4 or 5e-5
├── Data quality issue → Check assessment pass rate
└── Model not loading correctly → Verify model_init_kwargs

CLI issues?
├── Flags ignored → Flags must come BEFORE script path
├── "unrecognized arguments" → Use --secrets (plural), not --secret
└── Job not found → Check hf auth login status
```

---

## Cost Estimates

| Hardware | Cost/hr | 1K examples (3 epochs) |
|----------|---------|------------------------|
| A10G | ~$1.50 | ~$4-5 (2-3 hours) |
| A100 | ~$4.00 | ~$12-16 (3-4 hours) |

**Note:** A100 required for Gemma 3 with 8k+ context, or any model with 16k+ context.

Scale linearly with dataset size and epochs.
