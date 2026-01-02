#!/bin/bash
# Submit fine-tuning jobs to HuggingFace Jobs
# Usage: ./scripts/submit_training_jobs.sh [gemma|qwen|both]
#
# Prerequisites (per model-trainer skill):
# - HF Pro/Team/Enterprise account
# - Logged in: hf auth login
# - Gemma license accepted: https://huggingface.co/google/gemma-3-12b-it

set -e

FLAVOR="a100-large"
TIMEOUT="6h"  # Training ~3-4h + buffer for loading/pushing

# Per model-trainer skill: flags MUST come BEFORE script path
submit_gemma() {
    echo "============================================================"
    echo "Submitting: Therapeutic Gemma 3 12B"
    echo "GPU: A100 (80GB), Timeout: $TIMEOUT"
    echo "Trackio: https://huggingface.co/spaces/marcgreen/trackio"
    echo "============================================================"
    hf jobs uv run \
        --flavor "$FLAVOR" \
        --timeout "$TIMEOUT" \
        --secrets HF_TOKEN \
        scripts/train_gemma3_12b.py
}

submit_qwen() {
    echo "============================================================"
    echo "Submitting: Therapeutic Qwen3 14B"
    echo "GPU: A100 (80GB), Timeout: $TIMEOUT"
    echo "Trackio: https://huggingface.co/spaces/marcgreen/trackio"
    echo "============================================================"
    hf jobs uv run \
        --flavor "$FLAVOR" \
        --timeout "$TIMEOUT" \
        --secrets HF_TOKEN \
        scripts/train_qwen3_14b.py
}

case "${1:-both}" in
    gemma)
        submit_gemma
        ;;
    qwen)
        submit_qwen
        ;;
    both)
        submit_gemma
        echo ""
        submit_qwen
        ;;
    *)
        echo "Usage: $0 [gemma|qwen|both]"
        echo "  gemma - Submit only Gemma 3 12B job"
        echo "  qwen  - Submit only Qwen3 14B job"
        echo "  both  - Submit both jobs (default)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Jobs submitted!"
echo "============================================================"
echo ""
echo "Monitor:"
echo "  hf jobs ps                  # List jobs"
echo "  hf jobs logs <job_id>       # View logs"
echo "  hf jobs inspect <job_id>    # Job details"
echo ""
echo "Trackio dashboard:"
echo "  https://huggingface.co/spaces/marcgreen/trackio"
echo ""
echo "Output models (after training):"
echo "  https://huggingface.co/marcgreen/therapeutic-gemma3-12b"
echo "  https://huggingface.co/marcgreen/therapeutic-qwen3-14b"
echo ""
echo "Estimated cost: ~\$12-16 per model on A100"
echo "Expected time: ~3-4 hours per job"
