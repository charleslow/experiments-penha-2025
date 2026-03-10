#!/usr/bin/env bash
# Run the full pipeline in dev mode with a 5-minute timeout.
# All output goes to outputs/dev_run.log (stdout + stderr).
# Exit codes: 0=success, 1=stage failed, 124=timeout
set -euo pipefail

TIMEOUT_SECONDS=300
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/outputs/dev_run.log"

cd "$PROJECT_DIR"
mkdir -p outputs

timeout --signal=KILL "$TIMEOUT_SECONDS" bash -c '
set -euo pipefail

echo "Dev pipeline run — started $(date +%H:%M:%S)"
echo "==========================================="
PIPELINE_START=$SECONDS

for stage in \
    "Stage 1: Data Preparation|python -m src.data.prepare --mode dev" \
    "Stage 2: Query Generation|python -m src.data.generate_queries --mode dev" \
    "Stage 3: Embeddings|python -m src.embeddings --mode dev" \
    "Stage 4: Tokenization|python -m src.tokenizers --mode dev" \
    "Stage 5: Generative Model|python -m src.generative --mode dev" \
    "Stage 6: Evaluation|python -m src.evaluation --mode dev"; do
    name="${stage%%|*}"
    cmd="${stage##*|}"
    echo ""
    echo "--- $name ---"
    start=$SECONDS
    $cmd
    echo "  done ($(( SECONDS - start ))s)"
done

echo ""
echo "==========================================="
echo "Pipeline complete in $(( SECONDS - PIPELINE_START ))s"
' > "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 137 ]; then
    echo "TIMEOUT after ${TIMEOUT_SECONDS}s" >> "$LOG_FILE"
    exit 124
elif [ "$EXIT_CODE" -ne 0 ]; then
    exit 1
fi
