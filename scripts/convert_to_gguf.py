"""
Convert fine-tuned LoRA adapter to GGUF for local inference.

This script:
1. Downloads the adapter from HuggingFace Hub
2. Merges the adapter with the base model
3. Saves the merged model
4. Converts to GGUF format using llama.cpp

Usage:
    # Convert Gemma 3 12B adapter
    uv run python scripts/convert_to_gguf.py \
        --adapter-repo marcgreen/therapeutic-gemma3-12b \
        --base-model google/gemma-3-12b-it \
        --output-dir ./models/gemma3-therapeutic

    # Convert Qwen3 14B adapter
    uv run python scripts/convert_to_gguf.py \
        --adapter-repo marcgreen/therapeutic-qwen3-14b \
        --base-model Qwen/Qwen3-14B \
        --output-dir ./models/qwen3-therapeutic
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_llama_cpp() -> Path | None:
    """Check if llama.cpp is available and return its path."""
    # Common locations
    possible_paths = [
        Path.home() / "llama.cpp",
        Path("./llama.cpp"),
        Path("/opt/llama.cpp"),
    ]

    for path in possible_paths:
        convert_script = path / "convert_hf_to_gguf.py"
        if convert_script.exists():
            return path

    return None


def download_adapter(adapter_repo: str, output_dir: Path) -> Path:
    """Download adapter from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading adapter from {adapter_repo}...")
    adapter_path = snapshot_download(
        adapter_repo,
        local_dir=output_dir / "adapter",
    )
    print(f"Adapter downloaded to {adapter_path}")
    return Path(adapter_path)


def merge_adapter(
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    use_cpu: bool = False,
) -> Path:
    """Merge LoRA adapter with base model."""
    import torch  # ty: ignore[unresolved-import]
    from peft import PeftModel  # ty: ignore[unresolved-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # ty: ignore[unresolved-import]

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {base_model}...")

    # Choose device based on available hardware
    # Always use bfloat16 to reduce memory: 14B model = ~28GB (bf16) vs ~56GB (f32)
    if use_cpu:
        device_map = "cpu"
        dtype = torch.bfloat16
    else:
        device_map = "auto"
        dtype = torch.bfloat16

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter with base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    merged.save_pretrained(merged_dir)

    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    print(f"Merged model saved to {merged_dir}")
    return merged_dir


def convert_to_gguf(
    merged_dir: Path,
    output_dir: Path,
    llama_cpp_path: Path,
    quant_type: str = "q4_k_m",
) -> Path:
    """Convert merged model to GGUF format."""
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    output_file = output_dir / f"model-{quant_type}.gguf"

    print(f"Converting to GGUF ({quant_type})...")
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outtype",
        quant_type,
        "--outfile",
        str(output_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"GGUF conversion failed: {result.stderr}")
        raise RuntimeError("GGUF conversion failed")

    print(f"GGUF saved to {output_file}")
    return output_file


def upload_gguf(
    gguf_path: Path,
    repo_id: str,
    quant_type: str = "q4_k_m",
) -> str:
    """Upload GGUF file to HuggingFace Hub.

    Creates a new repo (or uses existing) with GGUF file and model card.
    Returns the URL to the uploaded model.
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist (repo_type="model" is default)
    print(f"Creating/verifying repo: {repo_id}")
    try:
        create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload the GGUF file
    gguf_filename = f"{repo_id.split('/')[-1]}-{quant_type}.gguf"
    print(f"Uploading {gguf_path.name} as {gguf_filename}...")

    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_filename,
        repo_id=repo_id,
        commit_message=f"Add {quant_type} GGUF quantization",
    )

    # Create a simple model card if none exists
    readme_content = f"""---
license: apache-2.0
tags:
  - gguf
  - therapeutic
  - coaching
  - fine-tuned
---

# {repo_id.split("/")[-1]}

GGUF quantized version for local inference.

## Files

- `{gguf_filename}` - {quant_type.upper()} quantization

## Usage

```bash
# With llama.cpp
llama-server -m {gguf_filename} --port 8080 -ngl 99

# With Ollama
ollama create {repo_id.split("/")[-1]} -f Modelfile
```

## Modelfile (for Ollama)

```
FROM ./{gguf_filename}
PARAMETER temperature 0.7
SYSTEM You are a supportive therapeutic coach...
```
"""

    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card",
        )
    except Exception:
        # README might already exist from adapter upload
        pass

    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to: {url}")
    return url


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned LoRA adapter to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Gemma 3 12B
    uv run python scripts/convert_to_gguf.py \\
        --adapter-repo marcgreen/therapeutic-gemma3-12b \\
        --base-model google/gemma-3-12b-it \\
        --output-dir ./models/gemma3-therapeutic

    # Qwen3 14B
    uv run python scripts/convert_to_gguf.py \\
        --adapter-repo marcgreen/therapeutic-qwen3-14b \\
        --base-model Qwen/Qwen3-14B \\
        --output-dir ./models/qwen3-therapeutic

Note: Requires llama.cpp to be cloned. If not found, clone it:
    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
""",
    )
    parser.add_argument(
        "--adapter-repo",
        required=True,
        help="HuggingFace repo with the LoRA adapter (e.g., marcgreen/therapeutic-gemma3-12b)",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model ID (e.g., google/gemma-3-12b-it or Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for merged model and GGUF",
    )
    parser.add_argument(
        "--quant-type",
        default="q4_k_m",
        choices=["q4_0", "q4_1", "q4_k_m", "q5_0", "q5_1", "q5_k_m", "q8_0", "f16"],
        help="Quantization type for GGUF (default: q4_k_m)",
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=Path,
        help="Path to llama.cpp directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for merging (slower but no GPU required, uses ~28GB RAM for 14B model)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip adapter download (use existing adapter in output-dir/adapter)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step (use existing merged model in output-dir/merged)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload GGUF to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--gguf-repo",
        help="HuggingFace repo for GGUF upload (default: adapter-repo + '-gguf')",
    )

    args = parser.parse_args()

    # Check llama.cpp
    llama_path_check = args.llama_cpp_path or check_llama_cpp()
    if llama_path_check is None:
        print("Error: llama.cpp not found. Please clone it:")
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        sys.exit(1)

    assert llama_path_check is not None  # Already checked above
    llama_path = llama_path_check
    print(f"Using llama.cpp at {llama_path}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download adapter
    if args.skip_download:
        adapter_path = args.output_dir / "adapter"
        print(f"Skipping download, using existing adapter at {adapter_path}")
    else:
        adapter_path = download_adapter(args.adapter_repo, args.output_dir)

    # Step 2: Merge adapter
    if args.skip_merge:
        merged_dir = args.output_dir / "merged"
        print(f"Skipping merge, using existing merged model at {merged_dir}")
    else:
        merged_dir = merge_adapter(
            args.base_model,
            adapter_path,
            args.output_dir,
            use_cpu=args.cpu,
        )

    # Step 3: Convert to GGUF
    gguf_path = convert_to_gguf(
        merged_dir,
        args.output_dir,
        llama_path,
        quant_type=args.quant_type,
    )

    # Step 4: Upload if requested
    if args.upload:
        gguf_repo = args.gguf_repo or f"{args.adapter_repo}-gguf"
        upload_gguf(gguf_path, gguf_repo, args.quant_type)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"GGUF file: {gguf_path}")
    print("\nTo test locally:")
    print(f"  llama-server -m {gguf_path} --port 8080 -ngl 99")
    if args.upload:
        gguf_repo = args.gguf_repo or f"{args.adapter_repo}-gguf"
        print(f"\nUploaded to: https://huggingface.co/{gguf_repo}")
    print("=" * 60)


if __name__ == "__main__":
    main()
