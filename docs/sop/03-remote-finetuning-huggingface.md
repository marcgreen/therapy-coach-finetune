# SOP 3: Remote Fine-tuning via HuggingFace

> **Lessons learned from the Therapeutic Coaching Fine-tuning Project**

This SOP documents how to run QLoRA fine-tuning on HuggingFace Jobs infrastructure, including the nine critical issues we encountered and their solutions.

---

## Overview

HuggingFace Jobs provides serverless GPU access for training. The workflow:

```
Local: Prepare dataset → Push to Hub
  ↓
HF Jobs: Submit training script → GPU runs → Model pushed to Hub
  ↓
Local: Download GGUF → Run locally
```

**Key insight:** HuggingFace Jobs has subtle permission and configuration issues that cause silent failures. Every issue in this SOP cost us 1-4 hours of debugging.

---

## Prerequisites

Before submitting any job:

1. **HuggingFace Pro subscription** ($9/month) - Required for Jobs access
2. **Prepaid credits** - GPU time billed from credit balance
3. **Token with ALL write permissions** - Including Spaces management
4. **Gated model access** - Accept license for models like Gemma

### Critical Pre-flight: Create Output Repos Locally

Your CLI has full permissions. The Jobs environment has limited permissions. Create repos before submitting:

```bash
# Model output repo (REQUIRED)
huggingface-cli repo create username/model-name --type model --private

# Dataset (if not already created)
huggingface-cli repo create username/dataset-name --type dataset --private
```

---

## Issue 1: Token Permission Mismatch (Most Common)

### The Problem

Multiple permission-related errors:
- "403 Forbidden: You don't have the rights to create a model"
- "Cannot access content at: .../info/lfs/objects/batch"
- "You have read access but not the required permissions"

### Root Cause

The `$HF_TOKEN` placeholder in HF Jobs resolves to a **session OAuth token** (`hf_oauth_...`), NOT your personal access token. OAuth tokens have limited permissions.

```
$HF_TOKEN → hf_oauth_xxxxx (limited session token)
Your actual token → hf_xxxxx (full access)
```

### The Solution

**Option A (Recommended):** Pass your actual token directly:
```python
# Get your token
cat ~/.cache/huggingface/token
# Output: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Pass directly in secrets
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
})
```

**Option B:** Create repos locally first, then job only needs push (lower permission):
```bash
huggingface-cli repo create username/model-name --type model --private
```

### Lesson Learned

Running `huggingface-cli login` locally does NOT update what Jobs sees. The token on HuggingFace's servers is what matters.

---

## Issue 2: Hidden Vocabulary Cost (Gemma 3 OOM)

### The Problem

Gemma 3 12B crashes with OOM despite:
- 4-bit quantization enabled
- Gradient checkpointing enabled
- Batch size = 1

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB.
```

### Root Cause

Gemma 3 has a **262,144 token vocabulary** (for vision-language support). This is 8x larger than Llama/Mistral.

**Logits are always computed in full precision**, regardless of model quantization:

```
logits_size = seq_length x vocab_size x 4 bytes (float32)
```

| Model | Vocab Size | Logits @ 4096 tokens | Logits @ 8192 tokens |
|-------|------------|----------------------|----------------------|
| **Gemma 3** | **262,144** | **4.3 GB** | **8.6 GB** |
| Llama 3 | 128,256 | 2.1 GB | 4.2 GB |
| Mistral | 32,768 | 0.5 GB | 1.1 GB |

### The Solution

Reduce `max_length` based on your GPU:

| GPU | VRAM | Max Sequence (Gemma 3 12B 4-bit) |
|-----|------|----------------------------------|
| A10G | 24 GB | ~2048 tokens |
| A100 | 80 GB | ~16384 tokens |

```python
config = SFTConfig(
    max_length=2048,  # For A10G with Gemma 3
    ...
)
```

### Lesson Learned

Model weights are quantized. Logits are not. Large vocabularies dominate memory at inference/loss computation.

---

## Issue 3: SFTTrainer API Confusion

### The Problem

Multiple API errors:
- `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'model_init_kwargs'`
- `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`

### Root Cause

TRL's SFTTrainer API changed. Documentation and examples are often outdated.

### The Solution

**Rule 1:** `model_init_kwargs` goes in `SFTConfig`, not `SFTTrainer`:

```python
# WRONG
trainer = SFTTrainer(model=..., model_init_kwargs={...})

# CORRECT
config = SFTConfig(
    model_init_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        ...
    },
    ...
)
trainer = SFTTrainer(model="model-name", args=config, ...)
```

**Rule 2:** Pass model name string, not loaded model. Tokenizer is auto-loaded:

```python
# WRONG
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)

# CORRECT
trainer = SFTTrainer(
    model="google/gemma-3-12b-it",  # String, not loaded model
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)
```

### Lesson Learned

TRL API is a moving target. Always test locally with a tiny dataset before submitting to Jobs.

---

## Issue 4: Auto-Login Breaks Token Detection

### The Problem

Explicit login call causes KeyError:

```python
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))  # KeyError during token parsing
```

### The Solution

Don't call `login()`. The `huggingface_hub` library auto-detects `HF_TOKEN` from environment:

```python
# Just use the library directly - it finds HF_TOKEN automatically
dataset = load_dataset("username/private-dataset", split="train")
```

### Lesson Learned

HuggingFace Jobs sets `HF_TOKEN` as an environment variable when you pass `secrets={"HF_TOKEN": "..."}`. Manual login interferes with this.

---

## Issue 5: Flash Attention Build Failure

### The Problem

```
ModuleNotFoundError: No module named 'torch'
Failed to build flash-attn==2.8.3
```

### Root Cause

`flash-attn` requires torch during compilation. With `uv` parallel dependency resolution, the build starts before torch is available.

### The Solution

Don't use flash-attn with HF Jobs uv scripts. Standard attention works fine:

```python
# Don't include in dependencies
# dependencies = [
#     "flash-attn>=2.8.0",  # REMOVE THIS
# ]
```

If you need flash attention:
- Use A100 (has enough memory without it)
- Or use a pre-built Docker image

### Lesson Learned

Build-time dependencies are tricky in serverless environments. Stick to pure Python packages.

---

## Issue 6: Trackio Space Variables Permission

### The Problem

```
403 Forbidden: Cannot access content at: .../spaces/username/trackio/variables
EOFError  # getpass() trying to prompt in non-interactive container
```

### Root Cause

Trackio needs to configure your Space, but the Jobs token lacks "Spaces management" permissions.

### The Solution

**Simplest:** Skip Trackio entirely for initial runs:

```python
config = SFTConfig(
    report_to="none",  # Training progress in job logs
    logging_steps=10,
    ...
)
```

Monitor via job logs instead. Add Trackio later when you have full permissions sorted.

### Lesson Learned

Trackio integration is fragile. Get training working first, add monitoring later.

---

## Issue 7: Trackio Dashboard 404s

### The Problem

Dashboard at `https://username-trackio.hf.space/` returns 404 even though Space exists.

### Root Cause

You manually created the Space (to test permissions), but it's empty. Trackio finds existing Space and doesn't deploy its app.

### The Solution

Delete both Space AND dataset, restart job:

```python
from huggingface_hub import HfApi
api = HfApi()
api.delete_repo('username/trackio', repo_type='space')
api.delete_repo('username/trackio-dataset', repo_type='dataset')
# Trackio will create fresh repos on next run
```

### Lesson Learned

Don't pre-create Trackio repos. Let it create them automatically.

---

## Issue 8: Deprecated Parameters Warning

### The Warning

```
The load_in_4bit and load_in_8bit arguments are deprecated
```

### Status

Warning only - still works. Future-proof version uses `BitsAndBytesConfig`:

```python
from transformers import BitsAndBytesConfig

config = SFTConfig(
    model_init_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        "device_map": "auto",
    },
    ...
)
```

---

## Issue 9: Dataset Access Timeout

### The Problem

```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='huggingface.co'...
```

### Root Cause

Private dataset requires authentication, but token wasn't being used.

### The Solution

Pass secrets in job submission:

```python
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "hf_xxxxx"}  # Your actual token
})
```

The library auto-detects `HF_TOKEN` from environment.

---

## Working Script Template

Based on all lessons learned:

```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.25.0",
#     "bitsandbytes>=0.41.0",
#     "pillow"  # Required for Gemma 3 (VLM architecture)
# ]
# ///

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# HF_TOKEN auto-detected from environment - DON'T call login()
dataset = load_dataset("username/dataset", split="train")

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

config = SFTConfig(
    output_dir="model-name",
    push_to_hub=True,
    hub_model_id="username/model-name",  # Must exist already!
    hub_strategy="every_save",

    # Quantization config in SFTConfig, not SFTTrainer
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
    max_length=2048,  # Critical for Gemma 3 on A10G
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    report_to="none",  # Skip Trackio for simplicity
    logging_steps=10,
)

# Pass model name string, not loaded model
trainer = SFTTrainer(
    model="google/gemma-3-12b-it",
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()
```

---

## Job Submission Checklist

Before submitting:

- [ ] Output model repo created locally: `huggingface-cli repo create username/model-name --type model --private`
- [ ] Dataset uploaded to Hub
- [ ] Gated model access accepted (e.g., Gemma license)
- [ ] Local test with tiny dataset passed
- [ ] `max_length` set appropriately for GPU + model vocab size

Submit:

```python
hf_jobs("uv", {
    "script": "<script content>",
    "flavor": "a10g-large",  # or "a100-large" for more memory
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "hf_xxxxx"}  # Your actual token, not $HF_TOKEN
})
```

---

## GPU Selection Guide

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| **A10G** | 24 GB | ~$1.50 | 7B models, 12B with short context |
| **A100** | 80 GB | ~$4.00 | 12B+ models, longer context |

### Gemma 3 12B Specifically

Due to 262K vocabulary:
- **A10G:** max_length=2048 (safe), max_length=3072 (risky)
- **A100:** max_length=16384 (safe)

---

## GGUF Conversion

After training completes, convert to GGUF for local inference:

```python
# Submit as separate HF Job after training
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download adapter
adapter_path = snapshot_download("username/therapeutic-model")

# Load base model + merge adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged")
tokenizer.save_pretrained("./merged")

# Convert to GGUF using llama.cpp
# (clone llama.cpp, run convert_hf_to_gguf.py)
```

Download locally:
```bash
huggingface-cli download username/model-gguf \
    model-q4_k_m.gguf \
    --local-dir ~/models/
```

Run with llama-server:
```bash
llama-server -m ~/models/model-q4_k_m.gguf --port 8080 -ngl 99
```

---

## Monitoring Training

Without Trackio, monitor via job logs:

1. Go to HuggingFace Jobs dashboard
2. Click on your running job
3. View logs for:
   - Step number and total steps
   - Training loss (should decrease over time)
   - Learning rate schedule
   - GPU memory usage

Expected loss progression:
- Start: ~2-3
- Epoch 1: ~1.5-2
- Epoch 2: ~1.0-1.5
- Epoch 3: ~0.8-1.2 (should stabilize)

---

## Cost Estimates

Based on our project:

| Phase | Time | Cost (A10G) |
|-------|------|-------------|
| Training (1000 examples, 3 epochs) | ~2-3 hours | ~$4-5 |
| GGUF conversion | ~30 min | ~$0.75 |
| **Total** | ~3-4 hours | ~$5-6 |

Scale linearly with dataset size and epochs.

---

## Alternative: MLX Local Training (Apple Silicon)

If you have an Apple Silicon Mac, you can train locally without HuggingFace Jobs:

```bash
# Install MLX
pip install mlx-lm

# Prepare data in MLX format
python scripts/prepare_mlx_data.py

# Train locally
python -m mlx_lm.lora \
    --model mlx-community/gemma-3-12b-it-8bit \
    --train \
    --data data/processed/mlx_training \
    --iters 800 \
    --batch-size 1 \
    --lora-layers 16
```

### MLX Config Example

```yaml
# config/mlx_lora_config.yaml
model: mlx-community/gemma-3-12b-it-8bit
train: true
data: data/processed/mlx_training
iters: 800
learning_rate: 1e-5
batch_size: 1
grad_accum_steps: 8
lora_layers: 16
mask_prompt: true  # Only compute loss on assistant turns
```

### Lessons Learned

1. **MLX is slower but free.** ~4-6 hours for 1000 examples on M3 Max vs 2-3 hours on A10G.

2. **8-bit quantized base model works.** `mlx-community/gemma-3-12b-it-8bit` trains fine.

3. **mask_prompt=true is essential.** Only compute loss on assistant turns, not user turns or system prompt.

4. **MLX adapter format differs.** Need to convert to HuggingFace format before merging with base model for GGUF export.

---

## Additional Dependencies Note

Gemma 3 is a vision-language model architecture. Even for text-only fine-tuning, you need:

```python
# /// script
# dependencies = [
#     ...
#     "pillow"  # Required for Gemma 3 even without images
# ]
# ///
```

Without Pillow, you get cryptic import errors about image processing.

---

## Test Locally Before Submitting

**Critical:** Always test your training script locally with a tiny dataset before submitting to HF Jobs. Each failed job costs time and money.

```python
# Create a tiny test dataset (10 examples)
tiny_dataset = dataset.select(range(10))

# Test locally with CPU (slow but validates code)
trainer = SFTTrainer(
    model="google/gemma-3-12b-it",
    train_dataset=tiny_dataset,
    args=SFTConfig(
        max_steps=2,  # Just 2 steps to verify setup
        ...
    ),
    ...
)

# If this runs without errors, submit to HF Jobs
```

### What Local Testing Catches

1. **Import errors** - Missing dependencies
2. **API mismatches** - Wrong argument names
3. **Data format issues** - Messages not in expected format
4. **Token errors** - Auth problems with private repos

### What Local Testing Doesn't Catch

1. **GPU memory issues** - Only appear on actual GPU
2. **Long training failures** - Convergence, learning rate issues
3. **Hub push permissions** - Jobs token differs from local

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| Using `$HF_TOKEN` placeholder | Resolves to limited OAuth token | Pass actual token directly |
| Not pre-creating output repo | Jobs can't create repos | Create locally first |
| Calling `huggingface_hub.login()` | Interferes with auto-detection | Let library auto-detect |
| High `max_length` with Gemma | Vocabulary cost causes OOM | Use 2048 on A10G |
| Including flash-attn | Build fails with uv | Use standard attention |
| Pre-creating Trackio repos | Dashboard 404s | Let Trackio create them |
| Passing loaded model to SFTTrainer | API expects string | Pass model name string |

---

## Troubleshooting Decision Tree

```
Job fails immediately?
  ├── "No module named X" → Check dependencies list
  ├── "403 Forbidden" → Token permissions (Issue 1)
  └── "Cannot create repo" → Pre-create repo locally (Issue 3)

Job fails during training?
  ├── "CUDA out of memory" → Reduce max_length (Issue 2)
  ├── "TypeError: unexpected argument" → Check SFTTrainer API (Issue 3)
  └── "KeyError" during login → Remove login() call (Issue 4)

Job completes but upload fails?
  ├── "403 on LFS" → Use actual token, not $HF_TOKEN (Issue 1)
  └── "Cannot access batch" → Pre-create repo locally

Training loss doesn't decrease?
  ├── Learning rate too high → Try 1e-4 or 5e-5
  ├── Data quality issue → Check assessment pass rate
  └── Model not loading correctly → Verify model_init_kwargs
```

---

*Last updated: January 2026*
*Based on therapeutic coaching fine-tuning project*
