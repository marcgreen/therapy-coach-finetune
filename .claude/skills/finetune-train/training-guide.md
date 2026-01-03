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

**Issue 7: Trackio Dual Initialization (Concurrent Jobs)**

When using `trackio.init()` + `report_to="trackio"`, you see TWO runs per job:

```
* Trackio project initialized: therapeutic-coaching  ← Your trackio.init()
* Created new run: gemma3-12b-sft
...
* Trackio project initialized: huggingface          ← TRL's default
* Created new run: marcgreen-1767396634
```

**Cause:** TRL's `report_to="trackio"` creates its own run instead of reusing your init.

**Workaround:**
- Same Trackio Space handles multiple runs fine (they're separated by name)
- Add timestamp to run names to avoid collisions: `f"gemma3-12b-{datetime.now().strftime('%Y%m%d-%H%M')}"`
- Your named run gets config metadata; TRL's default run gets actual metrics

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
from datetime import datetime

# Unique run name with timestamp (important for concurrent jobs)
RUN_NAME = f"gemma3-12b-{datetime.now().strftime('%Y%m%d-%H%M')}"

trackio.init(
    project="my-project",
    name=RUN_NAME,
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
from datetime import datetime

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

# Unique run name with timestamp (for concurrent jobs)
RUN_NAME = f"gemma3-12b-{datetime.now().strftime('%Y%m%d-%H%M')}"

# Initialize Trackio
# NOTE: TRL's report_to="trackio" may create a second run with defaults.
# This is a known issue - our explicit init ensures we have a named run.
trackio.init(
    project="my-project",
    name=RUN_NAME,
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

### Memory Requirements

**Critical:** Merging requires loading the full model in memory.

| Model | float32 (CPU default) | bfloat16 (recommended) |
|-------|----------------------|------------------------|
| Qwen3 14B | ~56GB RAM | ~28GB RAM |
| Gemma 3 12B | ~48GB RAM | ~24GB RAM |
| Llama 3 8B | ~32GB RAM | ~16GB RAM |

**Always use bfloat16** to reduce memory by half. The scripts below default to bfloat16.

### Known Issues

**Issue: Homebrew llama.cpp version mismatch**

Homebrew's `llama.cpp` has a version mismatch with the `gguf` Python package:
```
ImportError: cannot import name 'MistralTokenizerType' from 'gguf.vocab'
```

**Solution:** Clone fresh from GitHub instead of using Homebrew:
```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
pip install -r ~/llama.cpp/requirements.txt
```

The Homebrew `llama-quantize` binary still works fine for quantization.

### Manual Process (Recommended)

The manual process gives you more control and works reliably on macOS.

**Step 1: Download Base Model (Resumable)**
```bash
# Use hf download for resumable downloads (important for 28GB models)
hf download Qwen/Qwen3-14B --local-dir ~/models/qwen3-14b-base

# Or for Gemma
hf download google/gemma-3-12b-it --local-dir ~/models/gemma3-12b-base
```

**Step 2: Download Adapter**
```bash
# Download adapter from HuggingFace
hf download username/therapeutic-qwen3-14b --local-dir ./models/qwen3-therapeutic/adapter
```

**Step 3: Merge Adapter (bfloat16)**

Use `scripts/merge_lora_adapter.py`:
```bash
uv run python scripts/merge_lora_adapter.py \
    --base-model ~/models/qwen3-14b-base \
    --adapter-path ./models/qwen3-therapeutic/adapter \
    --output-dir ./models/qwen3-therapeutic/merged
```

This loads in bfloat16 (~28GB RAM for 14B model) and saves the merged model.

**Step 4: Convert to GGUF**
```bash
# IMPORTANT: Use cloned llama.cpp, NOT Homebrew version
uv run python ~/llama.cpp/convert_hf_to_gguf.py \
    --outtype bf16 \
    --outfile ./models/qwen3-therapeutic/therapeutic-qwen3-14b-bf16.gguf \
    ./models/qwen3-therapeutic/merged
```

**Step 5: Quantize**
```bash
# Homebrew llama-quantize works fine for this step
llama-quantize \
    ./models/qwen3-therapeutic/therapeutic-qwen3-14b-bf16.gguf \
    ./models/qwen3-therapeutic/therapeutic-qwen3-14b-q4_k_m.gguf \
    Q4_K_M
```

Quantization reduces file size: 28GB (bf16) → 8.4GB (Q4_K_M).

**Step 6: Test Locally**
```bash
llama-server -m ./models/qwen3-therapeutic/therapeutic-qwen3-14b-q4_k_m.gguf --port 8080 -ngl 99
```

**Step 7: Upload to Hub (Optional)**
```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo("username/model-gguf", exist_ok=True)
api.upload_file(
    path_or_fileobj="therapeutic-qwen3-14b-q4_k_m.gguf",
    path_in_repo="therapeutic-qwen3-14b-q4_k_m.gguf",
    repo_id="username/model-gguf",
)
```

### Automated Script

Use `scripts/convert_to_gguf.py` for end-to-end conversion:

```bash
# Prerequisites: clone llama.cpp (not Homebrew!)
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp

# Convert Qwen adapter to GGUF
uv run python scripts/convert_to_gguf.py \
    --adapter-repo username/therapeutic-qwen3-14b \
    --base-model Qwen/Qwen3-14B \
    --output-dir ./models/qwen3-therapeutic

# Convert AND upload to HuggingFace Hub
uv run python scripts/convert_to_gguf.py \
    --adapter-repo username/therapeutic-gemma3-12b \
    --base-model google/gemma-3-12b-it \
    --output-dir ./models/gemma3-therapeutic \
    --upload
```

**Script options:**
- `--quant-type`: Quantization type (default: `q4_k_m`)
- `--cpu`: Use CPU for merging with bfloat16 (default, ~28GB RAM for 14B)
- `--skip-download`: Use existing adapter in output-dir
- `--skip-merge`: Use existing merged model
- `--upload`: Upload GGUF to HuggingFace Hub
- `--gguf-repo`: Custom repo name for upload

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
2. **For each persona:** Generate 3 conversations with EACH model
3. **Same user simulator** for all (controlled comparison)
4. **Assess all conversations** with your rubric
5. **Statistical comparison** (paired t-test)

### Multi-Model Comparison Workflow

When comparing multiple fine-tuned models (e.g., Gemma vs Qwen):

**Step 1: Generate evaluation personas**
```bash
# Uses seeds 9000+ to avoid overlap with training (0-4999)
uv run python scripts/generate_eval_personas.py --count 15
# Output: data/eval/personas.json
```

**Step 2: Start model servers on different ports**
```bash
# Terminal 1: Baseline
llama-server -m gemma-3-12b-it.gguf --port 8080 -ngl 99

# Terminal 2: Fine-tuned Gemma
llama-server -m therapeutic-gemma.gguf --port 8081 -ngl 99

# Terminal 3: Fine-tuned Qwen
llama-server -m therapeutic-qwen.gguf --port 8082 -ngl 99
```

**Step 3: Run 3-way comparison**
```bash
uv run python scripts/run_model_evaluation.py \
    --personas data/eval/personas.json \
    --output-dir data/eval/results \
    --baseline-port 8080 \
    --gemma-port 8081 \
    --qwen-port 8082
```

**Step 4: Review report**
Report saved to `data/eval/results/evaluation_report.md` with:
- Per-model statistics (mean, std, pass rate)
- Pairwise comparisons with p-values
- Category breakdown

### Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_eval_personas.py` | Generate NEW personas for evaluation |
| `scripts/run_model_evaluation.py` | Run full 3-way comparison |
| `scripts/merge_lora_adapter.py` | Merge LoRA adapter with base model (bfloat16, ~28GB RAM) |
| `scripts/convert_to_gguf.py` | End-to-end: download, merge, convert, optionally upload |

### Statistical Test

```python
from scipy import stats
import numpy as np

base_scores = [...]
finetuned_scores = [...]

# Paired t-test (same personas, same user simulator)
t_stat, p_value = stats.ttest_rel(finetuned_scores, base_scores)

improvement = np.mean(finetuned_scores) - np.mean(base_scores)
improvement_pct = improvement / np.mean(base_scores) * 100

print(f"Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
print(f"p-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

**Why paired t-test:** Each persona generates conversations with both models using the same user simulator, making the samples paired. This increases statistical power.

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Absolute improvement | ≥0.10 (10 points) |
| Statistical significance | p < 0.05 |
| Safety regressions | None |

### Recommended Sample Size

- **15 personas × 3 conversations = 45 samples per model**
- This provides statistical power to detect ~10% improvement at p < 0.05
- More personas = more power, but diminishing returns after ~20

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
├── Job not found → Check hf auth login status
└── TTY/socket errors with hf jobs ps → Use Python API fallback (see below)

Job shows ERROR but adapter exists?
├── Calculate expected steps: (examples ÷ batch_size) × epochs
├── Check adapter repo for checkpoint matching final step
└── Job may have completed but timed out on push → Safe to use adapter
```

### HF Jobs API Fallback

When `hf jobs ps` fails with TTY/socket errors, use the Python API directly:

```python
import requests

# Get your jobs
resp = requests.get(
    "https://huggingface.co/api/jobs/YOUR_USERNAME",
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
)

for job in resp.json()["jobs"]:
    print(f"{job['metadata']['jobId']}: {job['status']} - step {job.get('step', 'N/A')}")
```

### Training Completion Detection

HF Jobs can timeout after training completes but before final push. To verify:

1. **Calculate expected steps:**
   ```
   steps = (num_examples ÷ batch_size) × num_epochs
   Example: (1294 ÷ 8) × 3 = 486 steps
   ```

2. **Check adapter repo for final checkpoint:**
   - Look for `adapter_model.safetensors` commit message like "step 486"
   - If step matches expected total, training completed successfully

3. **Download and use the adapter:**
   ```bash
   hf download username/model-adapter --local-dir ./adapter
   ```

---

## Cost Estimates

| Hardware | Cost/hr | 1K examples (3 epochs) |
|----------|---------|------------------------|
| A10G | ~$1.50 | ~$4-5 (2-3 hours) |
| A100 | ~$4.00 | ~$12-16 (3-4 hours) |

**Note:** A100 required for Gemma 3 with 8k+ context, or any model with 16k+ context.

Scale linearly with dataset size and epochs.
