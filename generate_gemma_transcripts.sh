#!/bin/bash
# Generate 10 Gemma 3 12B transcripts in 2 batches of 5
# Each batch runs 5 transcripts in parallel, then waits before starting the next batch
# Progress files are written so you can check status while AFK

set -e

OUTPUT_DIR="data/raw/transcripts"
PROGRESS_FILE="data/raw/generation_progress.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS_FILE"
}

echo "" > "$PROGRESS_FILE"
log "=========================================="
log "GEMMA TRANSCRIPT GENERATION"
log "10 transcripts in 2 batches of 5"
log "Seeds: 1020-1029"
log "=========================================="

# Check llama-server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log "ERROR: llama-server not running on port 8080"
    log "Start it with: llama-server -m ~/models/gemma-3-12b-it-q4_0.gguf --port 8080 -ngl 99"
    exit 1
fi
log "llama-server is running"
log ""

# Batch 1: seeds 1020-1024
log "=========================================="
log "BATCH 1/2 STARTING: Seeds 1020-1024"
log "=========================================="
BATCH1_START=$(date +%s)
BEFORE_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
PIDS=()
for seed in 1020 1021 1022 1023 1024; do
    log "  Starting seed $seed..."
    uv run python run_gemma_interactive.py $seed 15 &
    PIDS+=($!)
done
log "Waiting for batch 1 (PIDs: ${PIDS[*]})..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || log "  Process $pid exited with error"
done
BATCH1_END=$(date +%s)
BATCH1_TIME=$((BATCH1_END - BATCH1_START))
AFTER_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
CREATED=$((AFTER_COUNT - BEFORE_COUNT))
log "BATCH 1/2 COMPLETE (${BATCH1_TIME}s) - Created $CREATED/5 transcripts"
if [ "$CREATED" -lt 5 ]; then
    log "WARNING: Expected 5 transcripts, got $CREATED"
fi
log ""

# Batch 2: seeds 1025-1029
log "=========================================="
log "BATCH 2/2 STARTING: Seeds 1025-1029"
log "=========================================="
BATCH2_START=$(date +%s)
BEFORE_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
PIDS=()
for seed in 1025 1026 1027 1028 1029; do
    log "  Starting seed $seed..."
    uv run python run_gemma_interactive.py $seed 15 &
    PIDS+=($!)
done
log "Waiting for batch 2 (PIDs: ${PIDS[*]})..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || log "  Process $pid exited with error"
done
BATCH2_END=$(date +%s)
BATCH2_TIME=$((BATCH2_END - BATCH2_START))
AFTER_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
CREATED=$((AFTER_COUNT - BEFORE_COUNT))
log "BATCH 2/2 COMPLETE (${BATCH2_TIME}s) - Created $CREATED/5 transcripts"
if [ "$CREATED" -lt 5 ]; then
    log "WARNING: Expected 5 transcripts, got $CREATED"
fi
log ""

# Summary
TOTAL_TIME=$((BATCH1_TIME + BATCH2_TIME))
log "=========================================="
log "ALL BATCHES COMPLETE"
log "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m)"
log "=========================================="
log ""
log "Generated transcripts:"
ls -la $OUTPUT_DIR/*.json 2>/dev/null | tee -a "$PROGRESS_FILE"
log ""
log "Total files: $(ls $OUTPUT_DIR/*.json 2>/dev/null | wc -l | tr -d ' ')"
log ""
log "Progress saved to: $PROGRESS_FILE"
