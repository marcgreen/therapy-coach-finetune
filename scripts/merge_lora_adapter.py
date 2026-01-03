"""Merge LoRA adapter with base model for GGUF conversion.

Usage:
    # After downloading base model with: hf download Qwen/Qwen3-14B
    uv run python scripts/merge_lora_adapter.py \
        --base-model Qwen/Qwen3-14B \
        --adapter-path ./models/qwen3-therapeutic/adapter \
        --output-dir ./models/qwen3-therapeutic/merged
"""

import argparse
import os
from pathlib import Path

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--adapter-path", required=True, type=Path, help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("LoRA Adapter Merge")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Output: {args.output_dir}")
    print(f"Dtype: {args.dtype}")
    print("=" * 60)

    # Step 1: Load base model
    print("\n[1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"      Model loaded. Parameters: {model.num_parameters():,}")

    # Step 2: Load adapter
    print(f"\n[2/4] Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    print("      Adapter loaded.")

    # Step 3: Merge
    print("\n[3/4] Merging adapter into base model...")
    merged = model.merge_and_unload()
    print("      Merge complete.")

    # Step 4: Save
    print(f"\n[4/4] Saving merged model to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"Merged model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
