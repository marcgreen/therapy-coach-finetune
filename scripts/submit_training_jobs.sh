#!/bin/bash
# Submit fine-tuning jobs to HuggingFace Jobs
# Usage: ./scripts/submit_training_jobs.sh [gemma|qwen|qwen-0.6b|all]
#
# Prerequisites (per model-trainer skill):
# - HF Pro/Team/Enterprise account
# - Logged in: hf auth login
# - Gemma license accepted: https://huggingface.co/google/gemma-3-12b-it

set -e

FLAVOR_A100="a100-large"
FLAVOR_A10G="a10g-large"
TIMEOUT_LONG="6h"   # Training ~3-4h + buffer for loading/pushing
TIMEOUT_SHORT="3h"  # Training ~1-2h + buffer for loading/pushing

# Per model-trainer skill: flags MUST come BEFORE script path
submit_gemma() {
    echo "============================================================"
    echo "Submitting: Therapeutic Gemma 3 12B"
    echo "GPU: A100 (80GB), Timeout: $TIMEOUT_LONG"
    echo "Trackio: https://huggingface.co/spaces/marcgreen/trackio"
    echo "============================================================"
    hf jobs uv run \
        --flavor "$FLAVOR_A100" \
        --timeout "$TIMEOUT_LONG" \
        --secrets HF_TOKEN \
        scripts/train_gemma3_12b.py
}

submit_qwen() {
    echo "============================================================"
    echo "Submitting: Therapeutic Qwen3 14B"
    echo "GPU: A100 (80GB), Timeout: $TIMEOUT_LONG"
    echo "Trackio: https://huggingface.co/spaces/marcgreen/trackio"
    echo "============================================================"
    hf jobs uv run \
        --flavor "$FLAVOR_A100" \
        --timeout "$TIMEOUT_LONG" \
        --secrets HF_TOKEN \
        scripts/train_qwen3_14b.py
}

submit_qwen_0.6b() {
    echo "============================================================"
    echo "Submitting: Therapeutic Qwen3 0.6B"
    echo "GPU: A10G (24GB), Timeout: $TIMEOUT_SHORT"
    echo "Trackio: https://huggingface.co/spaces/marcgreen/trackio"
    echo "============================================================"
    hf jobs uv run \
        --flavor "$FLAVOR_A10G" \
        --timeout "$TIMEOUT_SHORT" \
        --secrets HF_TOKEN \
        scripts/train_qwen3_0.6b.py
}

case "${1:-all}" in
    gemma)
        submit_gemma
        ;;
    qwen)
        submit_qwen
        ;;
    qwen-0.6b)
        submit_qwen_0.6b
        ;;
    all)
        submit_gemma
        echo ""
        submit_qwen
        echo ""
        submit_qwen_0.6b
        ;;
    *)
        echo "Usage: $0 [gemma|qwen|qwen-0.6b|all]"
        echo "  gemma     - Submit only Gemma 3 12B job"
        echo "  qwen      - Submit only Qwen3 14B job"
        echo "  qwen-0.6b - Submit only Qwen3 0.6B job"
        echo "  all       - Submit all jobs (default)"
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
echo "  https://huggingface.co/marcgreen/therapeutic-qwen3-0.6b"
echo ""
echo "Estimated cost:"
echo "  Gemma 3 12B:   ~\$12-16 (A100, 3-4h)"
echo "  Qwen3 14B:     ~\$12-16 (A100, 3-4h)"
echo "  Qwen3 0.6B:    ~\$2-4 (A10G, 1-2h)"
