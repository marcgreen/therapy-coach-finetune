#!/bin/bash
# Generate 20 Gemma 3 12B transcripts in 4 batches of 5
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
log "20 transcripts in 4 batches of 5"
log "Seeds: 1000-1019"
log "=========================================="

# Check llama-server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log "ERROR: llama-server not running on port 8080"
    log "Start it with: llama-server -m ~/models/gemma-3-12b-it-q4_0.gguf --port 8080 -ngl 99"
    exit 1
fi
log "llama-server is running"
log ""

# Batch 1: seeds 1000-1004
log "=========================================="
log "BATCH 1/4 STARTING: Seeds 1000-1004"
log "=========================================="
BATCH1_START=$(date +%s)
BEFORE_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
PIDS=()
for seed in 1000 1001 1002 1003 1004; do
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
log "BATCH 1/4 COMPLETE (${BATCH1_TIME}s) - Created $CREATED/5 transcripts"
if [ "$CREATED" -lt 5 ]; then
    log "WARNING: Expected 5 transcripts, got $CREATED"
fi
log ""

# Batch 2: seeds 1005-1009
log "=========================================="
log "BATCH 2/4 STARTING: Seeds 1005-1009"
log "=========================================="
BATCH2_START=$(date +%s)
BEFORE_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
PIDS=()
for seed in 1005 1006 1007 1008 1009; do
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
log "BATCH 2/4 COMPLETE (${BATCH2_TIME}s) - Created $CREATED/5 transcripts"
if [ "$CREATED" -lt 5 ]; then
    log "WARNING: Expected 5 transcripts, got $CREATED"
fi
log ""

# Batch 3: seeds 1010-1014
log "=========================================="
log "BATCH 3/4 STARTING: Seeds 1010-1014"
log "=========================================="
BATCH3_START=$(date +%s)
BEFORE_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
PIDS=()
for seed in 1010 1011 1012 1013 1014; do
    log "  Starting seed $seed..."
    uv run python run_gemma_interactive.py $seed 15 &
    PIDS+=($!)
done
log "Waiting for batch 3 (PIDs: ${PIDS[*]})..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || log "  Process $pid exited with error"
done
BATCH3_END=$(date +%s)
BATCH3_TIME=$((BATCH3_END - BATCH3_START))
AFTER_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
CREATED=$((AFTER_COUNT - BEFORE_COUNT))
log "BATCH 3/4 COMPLETE (${BATCH3_TIME}s) - Created $CREATED/5 transcripts"
if [ "$CREATED" -lt 5 ]; then
    log "WARNING: Expected 5 transcripts, got $CREATED"
fi
log ""

# Batch 4: seeds 1015-1019
log "=========================================="
log "BATCH 4/4 STARTING: Seeds 1015-1019"
log "=========================================="
BATCH4_START=$(date +%s)
BEFORE_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
PIDS=()
for seed in 1015 1016 1017 1018 1019; do
    log "  Starting seed $seed..."
    uv run python run_gemma_interactive.py $seed 15 &
    PIDS+=($!)
done
log "Waiting for batch 4 (PIDs: ${PIDS[*]})..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || log "  Process $pid exited with error"
done
BATCH4_END=$(date +%s)
BATCH4_TIME=$((BATCH4_END - BATCH4_START))
AFTER_COUNT=$(ls $OUTPUT_DIR/gemma_session_*.json 2>/dev/null | wc -l | tr -d ' ')
CREATED=$((AFTER_COUNT - BEFORE_COUNT))
log "BATCH 4/4 COMPLETE (${BATCH4_TIME}s) - Created $CREATED/5 transcripts"
if [ "$CREATED" -lt 5 ]; then
    log "WARNING: Expected 5 transcripts, got $CREATED"
fi
log ""

# Summary
TOTAL_TIME=$((BATCH1_TIME + BATCH2_TIME + BATCH3_TIME + BATCH4_TIME))
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
