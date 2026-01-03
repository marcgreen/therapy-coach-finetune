#!/usr/bin/env python3
"""Merge LoRA adapter into base model and save for GGUF conversion."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
) -> None:
    """Merge LoRA adapter into base model and save."""
    print(f"Loading base model from {base_model_path}...")

    # Load base model in bf16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("Done! Merged model saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model directory",
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged model",
    )

    args = parser.parse_args()
    merge_lora(args.base_model, args.lora, args.output)


if __name__ == "__main__":
    main()
