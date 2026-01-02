# HuggingFace Jobs Troubleshooting

Issues encountered while setting up SFT training on HuggingFace Jobs infrastructure.

## Prerequisites

Before submitting a job, ensure:

1. **HuggingFace Pro subscription** ($9/month) - Required for Jobs access
2. **Prepaid credits** - GPU time is billed from your credit balance
3. **Token with write permissions** - For pushing models to Hub
4. **Gated model access** - Accept license for models like Gemma (google/gemma-3-12b-it)

## Pre-flight Checklist

Before submitting, create these repos locally (your CLI has write permissions, Jobs token may not):

```bash
# Model output repo
huggingface-cli repo create username/model-name --type model --private

# Trackio monitoring (if using report_to="trackio")
python -c "from huggingface_hub import create_repo; create_repo('username/trackio', repo_type='space', space_sdk='gradio', private=True, exist_ok=True)"
huggingface-cli repo create username/trackio-dataset --type dataset --private
```

## Issues & Fixes

### 1. Private Dataset Access Timeout

**Error:**
```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='huggingface.co'...
```

**Cause:** Private dataset requires authentication, but `load_dataset()` wasn't using the token.

**Fix:** HuggingFace Jobs automatically sets `HF_TOKEN` as an environment variable when you pass `secrets={"HF_TOKEN": "$HF_TOKEN"}`. The `huggingface_hub` library auto-detects this - no explicit `login()` call needed.

```python
# Don't do this - causes KeyError with token parsing
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))  # Breaks

# Just let huggingface_hub auto-detect HF_TOKEN env var
dataset = load_dataset("username/private-dataset", split="train")  # Works
```

### 2. SFTTrainer API Errors

**Error 1:** `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'model_init_kwargs'`

**Fix:** `model_init_kwargs` goes in `SFTConfig`, not `SFTTrainer`:

```python
# Wrong
trainer = SFTTrainer(model=..., model_init_kwargs={...})

# Correct
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

**Error 2:** `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`

**Fix:** Don't pass tokenizer separately. When you pass a model name string, SFTTrainer loads the tokenizer automatically:

```python
# Wrong
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)

# Correct - pass model name, let SFTTrainer handle loading
trainer = SFTTrainer(
    model="google/gemma-3-12b-it",  # String, not loaded model
    train_dataset=dataset,
    args=config,  # config has model_init_kwargs for quantization
    peft_config=peft_config,
)
```

### 3. Cannot Create Model Repository

**Error:**
```
403 Forbidden: You don't have the rights to create a model under the namespace "username"
```

**Cause:** The `$HF_TOKEN` in Jobs environment may have limited permissions, even if your local token has full access.

**Fix:** Create the output repository locally before submitting the job:

```bash
huggingface-cli repo create username/model-name --type model --private
```

Then the job only needs to push to an existing repo (lower permission requirement).

### 4. Trackio Space Variables Permission Error

**Error:**
```
403 Forbidden: You have read access but not the required permissions for this operation.
Cannot access content at: https://huggingface.co/api/spaces/username/trackio/variables
```

Followed by:
```
EOFError  # From getpass() trying to prompt for token in non-interactive container
```

**Cause:** Trackio needs to add variables to your Space, but the Jobs token doesn't have "Spaces management" permissions. Even if you create the Space locally, the job still can't configure it.

**Key insight:** `$HF_TOKEN` in Jobs references your token stored on **HuggingFace's servers**, not your local `~/.huggingface/token`. Running `huggingface-cli login` locally doesn't update what Jobs sees.

**Fix options:**

1. **Update your HF account token** with full permissions:
   - Go to https://huggingface.co/settings/tokens
   - Delete old tokens
   - Create new token with **all** write permissions (including Spaces)
   - This becomes the default token Jobs uses

2. **Skip Trackio** - Use `report_to="none"` and monitor via job logs:
   ```python
   config = SFTConfig(
       report_to="none",  # Training progress in job logs
       logging_steps=10,   # Log every 10 steps
       ...
   )
   ```

**Recommendation:** Skip Trackio for initial runs. The job logs show training progress. Add Trackio later when token permissions are sorted out.

### 5. Deprecated Parameters

**Warning:** `The load_in_4bit and load_in_8bit arguments are deprecated`

**Status:** Warning only, still works. Future-proof version would use `BitsAndBytesConfig` in `model_init_kwargs`:

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

### 6. Gemma 3 OOM Despite 4-bit Quantization (Hidden Vocabulary Cost)

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB.
```

Occurs at loss computation (`loss = loss_fct(flat_logits, flat_labels)`) even with:
- 4-bit quantization
- Gradient checkpointing
- Batch size 1

**Cause:** Gemma 3 has a **262,144 token vocabulary** (for vision-language support). This is 8x larger than Llama/Mistral (32k vocab). The logits tensor size at loss computation:

```
logits_size = seq_length × vocab_size × 4 bytes (float32 for loss)
```

| Model | Vocab Size | Logits at 4096 tokens | Logits at 8192 tokens |
|-------|------------|----------------------|----------------------|
| Gemma 3 | 262,144 | **4.3 GB** | **8.6 GB** |
| Llama 3 | 128,256 | 2.1 GB | 4.2 GB |
| Qwen 2.5 | 151,936 | 2.5 GB | 5.0 GB |
| Mistral | 32,768 | 0.5 GB | 1.1 GB |

Even with model weights quantized to 4-bit, **logits are always computed in full precision**.

**Fix:** Reduce `max_length` based on your GPU:

| GPU | VRAM | Max Sequence (Gemma 3 12B 4-bit) |
|-----|------|----------------------------------|
| A10G | 24 GB | ~2048 tokens |
| A100 | 80 GB | ~16384 tokens |

```python
config = SFTConfig(
    max_length=2048,  # For A10G with Gemma 3
    # ...
)
```

**Alternative:** Use a model with smaller vocabulary for memory-constrained training:
- Mistral 7B (32k vocab) - same A10G can handle 16k+ tokens
- Llama 3 8B (128k vocab) - can handle ~4k tokens on A10G

### 7. LFS Push Permission Error (Training Completes, Upload Fails)

**Error:**
```
403 Forbidden: You have read access but not the required permissions for this operation.
Cannot access content at: https://huggingface.co/username/model.git/info/lfs/objects/batch
```

**Cause:** The `$HF_TOKEN` placeholder in HF Jobs resolves to a **session OAuth token** (`hf_oauth_...`), NOT your personal access token. OAuth tokens have limited permissions and cannot push to repos.

**Fix:** Pass your actual token value directly instead of using `$HF_TOKEN`:

```python
# Get your token
cat ~/.cache/huggingface/token
# Output: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Pass it directly in secrets (NOT using $HF_TOKEN)
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # Your actual token
})
```

**Why `$HF_TOKEN` doesn't work:**
- `$HF_TOKEN` → `hf_oauth_...` (limited session token)
- Your token → `hf_...` (full access token with your permissions)

**Note:** Your token appears briefly in job logs, but logs are private to your account.

### 8. Trackio Dashboard 404s

**Error:** Dashboard at `https://username-trackio.hf.space/` returns 404.

**Cause:** You manually created the Space (e.g., to test permissions), but it's empty - no app files. Trackio finds the existing Space and doesn't deploy its app.

**Fix:** Delete both the Space AND dataset, then restart the job:

```python
from huggingface_hub import HfApi
api = HfApi()
api.delete_repo('username/trackio', repo_type='space')
api.delete_repo('username/trackio-dataset', repo_type='dataset')
# Then restart the training job - Trackio will create fresh repos
```

**Prevention:** Don't pre-create Trackio repos. Let Trackio create them automatically on first run.

### 9. Flash Attention Build Failure

**Error:**
```
ModuleNotFoundError: No module named 'torch'
Failed to build flash-attn==2.8.3
```

**Cause:** flash-attn requires torch during compilation, but uv installs dependencies in parallel. The build starts before torch is available.

**Fix:** Don't use flash-attn with HF Jobs uv scripts. The standard attention implementation works fine, just uses more memory. If you need flash attention:
- Use A100 (has enough memory without it)
- Or pre-build a Docker image with flash-attn installed

## Working Script Template

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

# HF_TOKEN is auto-detected from environment
dataset = load_dataset("username/dataset", split="train")

peft_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

config = SFTConfig(
    output_dir="model-name",
    push_to_hub=True,
    hub_model_id="username/model-name",  # Must exist already
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
    max_length=2048,  # See issue #6 for Gemma 3 memory limits
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model="google/gemma-3-12b-it",  # Pass string, not loaded model
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()
```

## Job Submission

```python
hf_jobs("uv", {
    "script": "<script content>",
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```
