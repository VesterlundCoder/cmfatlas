#!/usr/bin/env bash
# run_pipeline_post_audit.sh
#
# Waits for the Dreams audit to finish, then:
#   1. Purges failing CMFs from atlas_2d.db + store files
#   2. Launches the deep PSLQ analysis on all passing CMFs
#
# Run with:
#   bash gauge_agents/run_pipeline_post_audit.sh &

set -euo pipefail

PYTHON="/Users/davidsvensson/Desktop/cmf_atlas/venv/bin/python3"
GAUGE="$(dirname "$0")"
AUDIT="/tmp/dreams_full_audit_v3.jsonl"
PSLQ_OUT="/tmp/pslq_results.jsonl"
LOG_PURGE="/tmp/purge_failing_cmfs.log"
LOG_PSLQ="/tmp/deep_dreams_pslq.log"

# ── 1. Wait for audit to complete ─────────────────────────────────────────
echo "[pipeline] Waiting for audit process to finish..."
while pgrep -f "verify_convergence_dreams.py" > /dev/null 2>&1; do
    sleep 30
    DONE=$(wc -l < "$AUDIT" 2>/dev/null || echo 0)
    echo "[pipeline] $(date '+%H:%M:%S') — audit lines so far: $DONE"
done

echo "[pipeline] Audit finished at $(date '+%H:%M:%S')"
TOTAL=$(wc -l < "$AUDIT" 2>/dev/null || echo 0)
FAILS=$(python3 -c "
import json
n=0
with open('$AUDIT') as f:
    for l in f:
        try:
            r=json.loads(l)
            if not r.get('pass'): n+=1
        except: pass
print(n)
" 2>/dev/null || echo "?")
echo "[pipeline] Total records: $TOTAL  |  Failures: $FAILS"

# ── 2. Dry-run purge first ────────────────────────────────────────────────
echo ""
echo "[pipeline] Dry-run purge (preview)..."
"$PYTHON" "$GAUGE/purge_failing_cmfs.py" --audit "$AUDIT" --dry-run 2>&1 | tee /tmp/purge_dryrun.log

# ── 3. Execute purge ──────────────────────────────────────────────────────
echo ""
echo "[pipeline] Running purge..."
"$PYTHON" "$GAUGE/purge_failing_cmfs.py" --audit "$AUDIT" 2>&1 | tee "$LOG_PURGE"
echo "[pipeline] Purge complete. Log: $LOG_PURGE"

# ── 4. Launch deep PSLQ analysis ─────────────────────────────────────────
echo ""
echo "[pipeline] Starting deep PSLQ analysis..."
echo "[pipeline]   Output: $PSLQ_OUT"
echo "[pipeline]   Log:    $LOG_PSLQ"
"$PYTHON" "$GAUGE/deep_dreams_pslq.py" \
    --audit "$AUDIT" \
    --jobs 0 \
    --n-traj 10000 \
    --out "$PSLQ_OUT" \
    2>&1 | tee "$LOG_PSLQ"

echo ""
echo "[pipeline] ALL DONE at $(date '+%H:%M:%S')"
echo "[pipeline]   Purge log : $LOG_PURGE"
echo "[pipeline]   PSLQ log  : $LOG_PSLQ"
echo "[pipeline]   PSLQ out  : $PSLQ_OUT"
echo "[pipeline]   PSLQ hits DB: $(dirname "$GAUGE")/data/pslq_hits.db"
