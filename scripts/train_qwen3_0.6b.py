# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.51.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.45.0",
#     "datasets>=3.0.0",
#     "trackio",
# ]
# ///
# ty: ignore  # UV script - deps installed at runtime on HF Jobs
"""
Qwen3 0.6B fine-tuning for therapeutic coaching.

GPU: A10G (24GB) - 8k context max (16k OOMs due to logits memory)
Expected training time: ~1-2 hours for 1294 examples, 3 epochs

This is an experimental smaller model for faster iteration and testing.
Tradeoff: Less capacity for nuanced therapeutic responses vs 14B.
"""

import os
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import trackio  # ty: ignore[unresolved-import]
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig  # ty: ignore[unresolved-import]

# Config
MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "marcgreen/therapeutic-coaching-v1"
OUTPUT_REPO = "marcgreen/therapeutic-qwen3-0.6b"
MAX_LENGTH = 8192  # 16k OOMs on A10G due to logits memory (vocab × seq)

# Unique run name with timestamp to avoid collisions
RUN_NAME = f"qwen3-0.6b-{datetime.now().strftime('%Y%m%d-%H%M')}"

# Initialize Trackio BEFORE trainer to set project/run name
# NOTE: TRL's report_to="trackio" may create a second run with defaults.
# This is a known issue - our explicit init ensures we have a named run.
trackio.init(
    project="therapeutic-coaching",
    name=RUN_NAME,
    space_id="marcgreen/trackio",
    config={
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "max_length": MAX_LENGTH,
        "epochs": 3,
        "learning_rate": 2e-4,
    },
)

# Load dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Loaded {len(dataset)} training examples")  # ty: ignore[invalid-argument-type]

# QLoRA config - higher rank for better expressivity
peft_config = LoraConfig(
    r=32,  # Higher than remote's r=8 for more capacity
    lora_alpha=64,  # 2:1 ratio with r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Training config - larger batches possible with small model
config = SFTConfig(
    output_dir="therapeutic-qwen3-0.6b",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="every_save",
    # Quantization - 4bit for efficiency
    model_init_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "device_map": "auto",
    },
    # Training hyperparameters - batch=1 due to logits memory (vocab × seq × batch)
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Keep small for 16k context + 152K vocab
    gradient_accumulation_steps=16,  # Effective batch = 16 (same quality)
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # Memory optimization
    max_length=MAX_LENGTH,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    # Logging & checkpointing
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    # Trackio monitoring
    report_to="trackio",
)

# Create trainer
print(f"Initializing trainer for {MODEL_ID}...")
trainer = SFTTrainer(
    model=MODEL_ID,
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

print("Starting training...")
print(f"Max length: {MAX_LENGTH}")
print(f"Epochs: {config.num_train_epochs}")
print(
    f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}"
)

trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()

# Finish Trackio
trackio.finish()

print(f"Complete! Model at: https://huggingface.co/{OUTPUT_REPO}")
print("View metrics at: https://huggingface.co/spaces/marcgreen/trackio")
