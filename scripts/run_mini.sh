#!/usr/bin/env bash
# Run Stage 1 (Data Preparation) and Stage 2 (Query Generation) for mini validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "  Mini pipeline validation"
echo "=========================================="

echo ""
echo "--- Stage 1: Data Preparation ---"
python -m src.data.prepare --mode mini

echo ""
echo "--- Stage 2: Query Generation ---"
python -m src.data.generate_queries --mode mini

echo ""
echo "=========================================="
echo "  Mini pipeline complete!"
echo "=========================================="
