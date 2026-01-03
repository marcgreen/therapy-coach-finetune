# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.51.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.45.0",
#     "datasets>=3.0.0",
#     "trackio",
#     "pillow",
# ]
# ///
# ty: ignore  # UV script - deps installed at runtime on HF Jobs
"""
Gemma 3 12B fine-tuning for therapeutic coaching.

GPU: A100 (80GB) - required for 16k context with 262K vocabulary
Expected training time: ~3-4 hours for 1294 examples, 3 epochs
"""

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
DATASET_ID = "marcgreen/therapeutic-coaching-v1"
OUTPUT_REPO = "marcgreen/therapeutic-gemma3-12b"
MAX_LENGTH = 16384  # A100 can handle this with Gemma 3

# Unique run name with timestamp to avoid collisions
RUN_NAME = f"gemma3-12b-{datetime.now().strftime('%Y%m%d-%H%M')}"

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
print(f"Loaded {len(dataset)} training examples")

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
    output_dir="therapeutic-gemma3-12b",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="every_save",
    # Quantization
    model_init_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "device_map": "auto",
    },
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
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

trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()

# Finish Trackio
trackio.finish()

print(f"Complete! Model at: https://huggingface.co/{OUTPUT_REPO}")
print("View metrics at: https://huggingface.co/spaces/marcgreen/trackio")
