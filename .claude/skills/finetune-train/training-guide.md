# Training Guide

How to run fine-tuning and evaluate improvement.

---

## Training Options

| Option | Best For | Cost | Setup |
|--------|----------|------|-------|
| **HuggingFace Jobs** | Fast iteration, serverless | ~$5-10/1K examples | Minimal |
| **MLX Local** | Apple Silicon, free | Free (time) | Moderate |
| **Cloud GPU** | Full control, large jobs | Varies | Complex |

**Recommendation:** Start with HuggingFace Jobs for speed. Switch to MLX for iteration once you're confident in data quality.

---

## HuggingFace Jobs

### Prerequisites

1. **HuggingFace Pro subscription** ($9/month)
2. **Prepaid credits** for GPU time
3. **Token with ALL write permissions** (including Spaces management)
4. **Gated model access** (accept license for Gemma, Llama, etc.)

### Critical Pre-flight

**Create output repos locally before submitting:**

```bash
huggingface-cli repo create username/model-name --type model --private
huggingface-cli repo create username/dataset-name --type dataset --private
```

Jobs environment has limited permissions. Create repos with your full-permission CLI first.

### Known Issues

**Issue 1: Token Permission Mismatch**

The `$HF_TOKEN` placeholder resolves to a limited OAuth token, not your personal token.

```python
# WRONG: Uses limited session token
secrets={"HF_TOKEN": "$HF_TOKEN"}

# CORRECT: Your actual token
secrets={"HF_TOKEN": "hf_xxxxxxxxxxxxxxxx"}
```

**Issue 2: Gemma 3 OOM (Large Vocabulary)**

Gemma 3 has 262K vocabulary (for vision-language). Logits are computed in full precision regardless of quantization.

| Model | Vocab | Logits @ 4096 tokens |
|-------|-------|----------------------|
| Gemma 3 | 262K | 4.3 GB |
| Llama 3 | 128K | 2.1 GB |
| Mistral | 32K | 0.5 GB |

**Solution:** Reduce `max_length`:
- A10G (24GB): max_length=2048
- A100 (80GB): max_length=16384

**Issue 3: SFTTrainer API**

`model_init_kwargs` goes in `SFTConfig`, not `SFTTrainer`:

```python
# CORRECT
config = SFTConfig(
    model_init_kwargs={
        "load_in_4bit": True,
        ...
    }
)
trainer = SFTTrainer(model="model-name", args=config, ...)
```

Pass model name string, not loaded model. Tokenizer auto-loads.

**Issue 4: Auto-Login Breaks Token Detection**

Don't call `login()`. The library auto-detects `HF_TOKEN` from environment:

```python
# WRONG
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

# CORRECT: Just use the library
dataset = load_dataset("username/private-dataset")
```

**Issue 5: Flash Attention Build Failure**

`flash-attn` requires torch during compilation. With `uv`, this fails.

**Solution:** Don't use flash-attn with HF Jobs. Standard attention works fine on A10G/A100.

### Working Script Template

```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.25.0",
#     "bitsandbytes>=0.41.0",
#     "pillow"  # Required for Gemma 3 even without images
# ]
# ///

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# HF_TOKEN auto-detected - DON'T call login()
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
    hub_model_id="username/model-name",

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
    report_to="none",
    logging_steps=10,
)

trainer = SFTTrainer(
    model="google/gemma-3-12b-it",  # String, not loaded model
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()
```

### Job Submission

```python
hf_jobs("uv", {
    "script": "<script content>",
    "flavor": "a10g-large",  # or "a100-large"
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "hf_xxxxxxxx"}  # Your actual token
})
```

### GPU Selection

| GPU | VRAM | Cost/hr | Max Length (Gemma 3 12B) |
|-----|------|---------|--------------------------|
| A10G | 24GB | ~$1.50 | 2048 (safe) |
| A100 | 80GB | ~$4.00 | 16384 (safe) |

### Monitoring

Without Trackio, monitor via job logs:
- Step number / total steps
- Training loss (should decrease)
- Learning rate schedule

**Expected loss progression:**
- Start: ~2-3
- Epoch 1: ~1.5-2
- Epoch 2: ~1.0-1.5
- Epoch 3: ~0.8-1.2 (stabilizing)

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

### Configuration

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
# Download
huggingface-cli download username/model-gguf \
    model-q4_k_m.gguf \
    --local-dir ~/models/

# Run
llama-server -m ~/models/model-q4_k_m.gguf --port 8080 -ngl 99
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

Full-conversation evaluation:
- Tests multi-turn consistency
- Captures context utilization
- Uses your actual rubric

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

base_scores = [...]      # From base model conversations
finetuned_scores = [...]  # From fine-tuned conversations

# Paired t-test (same personas)
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

### Sanity Checks

| Check | Purpose |
|-------|---------|
| Perplexity on held-out | Did training actually work? |
| Human eval (5-10 convos) | Does LLM judge agree with humans? |
| Capability regression | Didn't break general abilities |
| Safety audit | No new harmful patterns |

---

## Troubleshooting

```
Job fails immediately?
├── "No module named X" → Check dependencies list
├── "403 Forbidden" → Token permissions
└── "Cannot create repo" → Pre-create repo locally

Job fails during training?
├── "CUDA out of memory" → Reduce max_length
├── "TypeError: unexpected argument" → Check SFTTrainer API
└── "KeyError" during login → Remove login() call

Training loss doesn't decrease?
├── Learning rate too high → Try 1e-4 or 5e-5
├── Data quality issue → Check assessment pass rate
└── Model not loading correctly → Verify model_init_kwargs
```

---

## Cost Estimates

| Phase | Time | Cost (A10G) |
|-------|------|-------------|
| Training (1K examples, 3 epochs) | ~2-3 hours | ~$4-5 |
| GGUF conversion | ~30 min | ~$0.75 |
| **Total** | ~3-4 hours | ~$5-6 |

Scale linearly with dataset size and epochs.
