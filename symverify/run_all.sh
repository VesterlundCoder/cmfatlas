#!/usr/bin/env bash
# CMF Atlas — Full symbolic verification run
# Usage:
#   ./run_all.sh               # verify everything
#   ./run_all.sh --limit 20    # quick test on first 20
#   ./run_all.sh --source "RamanujanTools"
#   ./run_all.sh --mode poly   # polynomial CMFs only (faster)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/reports/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/reports"

echo "Starting CMF symbolic verification..."
echo "Log: $LOG"
echo ""

sage "$SCRIPT_DIR/verify_cmfs.sage" "$@" 2>&1 | tee "$LOG"

echo ""
echo "Done. Log written to: $LOG"
