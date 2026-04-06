"""
purge_failing_cmfs.py

Reads the Dreams walk audit JSONL, deletes every failing CMF from:
  1. data/atlas_2d.db  (full cascade)
  2. gauge_agents/store_*.jsonl  (rewrite without failing lines)

Usage:
    python purge_failing_cmfs.py --audit /tmp/dreams_full_audit_v3.jsonl [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

HERE    = Path(__file__).parent
DB_PATH = HERE.parent / "data" / "atlas_2d.db"


# ── Cascade-delete one fingerprint from atlas_2d.db ──────────────────────

def _delete_fp(cur: sqlite3.Cursor, fp: str) -> int:
    """Delete all rows for fingerprint fp. Returns number of representations deleted."""
    cur.execute("SELECT id FROM representation WHERE canonical_fingerprint = ?", (fp,))
    rows = cur.fetchall()
    if not rows:
        return 0
    repr_ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(repr_ids))

    cur.execute(f"SELECT id FROM cmf WHERE representation_id IN ({placeholders})", repr_ids)
    cmf_ids = [r[0] for r in cur.fetchall()]

    if cmf_ids:
        cmf_ph = ",".join("?" * len(cmf_ids))
        cur.execute(f"SELECT id FROM eval_run WHERE cmf_id IN ({cmf_ph})", cmf_ids)
        run_ids = [r[0] for r in cur.fetchall()]
        if run_ids:
            run_ph = ",".join("?" * len(run_ids))
            cur.execute(f"DELETE FROM recognition_attempt WHERE eval_run_id IN ({run_ph})", run_ids)
        cur.execute(f"DELETE FROM eval_run WHERE cmf_id IN ({cmf_ph})", cmf_ids)
        cur.execute(f"DELETE FROM cmf WHERE id IN ({cmf_ph})", cmf_ids)

    cur.execute(f"DELETE FROM features WHERE representation_id IN ({placeholders})", repr_ids)
    cur.execute(f"DELETE FROM representation_equivalence WHERE representation_id IN ({placeholders})", repr_ids)
    cur.execute(f"DELETE FROM representation WHERE id IN ({placeholders})", repr_ids)
    return len(repr_ids)


# ── Remove failing lines from store JSONL files ───────────────────────────

def _purge_store_files(failing_fps: set[str], dry_run: bool) -> dict:
    store_files = sorted(HERE.glob("store_*.jsonl"))
    stats = {}
    for sf in store_files:
        lines = sf.read_text(errors="replace").splitlines()
        kept, removed = [], 0
        for line in lines:
            if not line.strip():
                continue
            try:
                fp = json.loads(line).get("fingerprint", "")
            except Exception:
                fp = ""
            if fp in failing_fps:
                removed += 1
            else:
                kept.append(line)
        if removed:
            stats[sf.name] = removed
            if not dry_run:
                with tempfile.NamedTemporaryFile("w", dir=sf.parent,
                                                 delete=False, suffix=".tmp") as tmp:
                    tmp.write("\n".join(kept) + "\n")
                shutil.move(tmp.name, sf)
    return stats


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="/tmp/dreams_full_audit_v3.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be deleted without touching anything")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"ERROR: audit file not found: {audit_path}")
        return

    # ── Load failing fingerprints ─────────────────────────────────────────
    total_records = 0
    failing_fps: set[str] = set()
    fail_reasons: dict[str, str] = {}

    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total_records += 1
            if not rec.get("pass", True):
                fp = rec.get("fp", "")
                if fp:
                    failing_fps.add(fp)
                    fail_reasons[fp] = rec.get("reason", "?")

    print(f"Audit: {total_records} records — {len(failing_fps)} failing")

    if not failing_fps:
        print("Nothing to purge.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {len(failing_fps)} CMFs.")
        for fp, reason in list(fail_reasons.items())[:20]:
            print(f"  {fp}  {reason}")
        if len(failing_fps) > 20:
            print(f"  ... and {len(failing_fps) - 20} more")
        return

    # ── DB purge ──────────────────────────────────────────────────────────
    print(f"\nDeleting {len(failing_fps)} CMFs from {DB_PATH} ...")
    t0 = time.time()
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = OFF")
    cur = con.cursor()

    n_deleted = 0
    for fp in failing_fps:
        n_deleted += _delete_fp(cur, fp)

    con.commit()
    con.close()
    print(f"  DB: deleted {n_deleted} representation rows in {time.time()-t0:.1f}s")

    # ── Store JSONL purge ─────────────────────────────────────────────────
    print(f"Cleaning store JSONL files ...")
    store_stats = _purge_store_files(failing_fps, dry_run=False)
    for name, count in sorted(store_stats.items()):
        print(f"  {name}: removed {count} lines")

    # ── Final counts ──────────────────────────────────────────────────────
    con2 = sqlite3.connect(DB_PATH)
    remaining = con2.execute("SELECT COUNT(*) FROM representation").fetchone()[0]
    con2.close()

    print(f"\nDone. DB now has {remaining:,} representations.")
    print(f"Purged {len(failing_fps)} failing CMFs in {time.time()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
