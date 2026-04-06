#!/usr/bin/env python3
"""
export_v2_candidates.py
=======================
Extracts CMFs discovered by the v2 irrational scout (anti-reward-hacking run)
from /tmp/scout_v2.log, looks them up in the store_*.jsonl files, and writes:

  data/irrational_candidates_v2.json   — full records of v2 irrational hits

Usage:
    python3 export_v2_candidates.py [--log /tmp/scout_v2.log]
    python3 export_v2_candidates.py --ingest  # also runs ingest + backfill + push
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE  = Path(__file__).parent
GAUGE = HERE / "gauge_agents"
OUT   = HERE / "data" / "irrational_candidates_v2.json"

_FP_LINE = re.compile(r'fp=([0-9a-f]{16})')
_IRR_LINE = re.compile(r'🔥|IRRATIONAL_UNKNOWN|TRUE_TRANSCENDENTAL')
_SCAN_LINE = re.compile(r'\[([A-J])\] #\d+ scanned.*?(\d+)/\d+ irrational.*fp=([0-9a-f]{16})')


def parse_log_fps(log_path: Path) -> set[str]:
    """Return fingerprints of CMFs classified as irrational in the v2 scout log."""
    irrational_fps: set[str] = set()

    lines = log_path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _SCAN_LINE.search(line)
        if m:
            fp = m.group(3)
            # Look ahead up to 10 lines for irrational marker
            window = "\n".join(lines[i:i+10])
            if _IRR_LINE.search(window):
                irrational_fps.add(fp)
        i += 1

    return irrational_fps


def load_store_records(fps: set[str]) -> list[dict]:
    """Read all store_*.jsonl and return records matching the given fingerprints."""
    records: dict[str, dict] = {}
    for sf in sorted(GAUGE.glob("store_*.jsonl")):
        for line in sf.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fp = (rec.get("fingerprint") or "")[:16]
            if fp in fps and fp not in records:
                records[fp] = rec
    return list(records.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/scout_v2.log", type=Path)
    ap.add_argument("--ingest", action="store_true",
                    help="Also run ingest_gauge_agents.py + backfill + Railway push")
    ap.add_argument("--all-irrational", action="store_true",
                    help="Export ALL irrational CMFs (not just v2)")
    args = ap.parse_args()

    log_path = args.log
    if not log_path.exists():
        print(f"[!] Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing v2 log: {log_path}")
    v2_fps = parse_log_fps(log_path)
    print(f"  V2 irrational fingerprints found: {len(v2_fps)}")

    if not v2_fps:
        print("  [!] No irrational FPs parsed — check log format.")
        sys.exit(1)

    print("Loading store records …")
    records = load_store_records(v2_fps)
    print(f"  Matched {len(records)} records from store files")

    # Build clean export records
    export = []
    for rec in sorted(records, key=lambda r: r.get("agent", ""), ):
        fp = (rec.get("fingerprint") or "")[:16]
        export.append({
            "fingerprint":      fp,
            "agent":            rec.get("agent"),
            "dim":              rec.get("dim"),
            "matrix_size":      rec.get("matrix_size") or rec.get("dim"),
            "best_delta":       rec.get("best_delta"),
            "looks_irrational": rec.get("looks_irrational", False),
            "irrational_type":  rec.get("irrational_type"),
            "limit_label":      rec.get("limit_label"),
            "limit_value":      rec.get("limit_value"),
            "limit_identified": rec.get("limit_identified"),
            "scout_batch":      "v2_anti_hack",
            "X0":               rec.get("X0"),
            "X1":               rec.get("X1"),
            "X2":               rec.get("X2"),
            "X3":               rec.get("X3"),
            "params":           rec.get("params"),
            "timestamp":        rec.get("timestamp"),
        })

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(export, indent=2))
    print(f"\n  Exported {len(export)} v2 irrational CMFs → {OUT}")

    # Print summary table
    from collections import Counter
    agent_counts = Counter(r["agent"] for r in export)
    gate_counts  = Counter(
        (r.get("irrational_type") or "unknown") for r in export
    )
    print("\n  By agent:", dict(sorted(agent_counts.items())))
    print("  By type: ", dict(gate_counts))

    if args.ingest:
        print("\n── Running ingest_gauge_agents.py …")
        subprocess.run([sys.executable, str(HERE / "ingest_gauge_agents.py"),
                        "--db", str(HERE / "data" / "atlas_2d.db")],
                       cwd=str(HERE), check=True)

        print("\n── Running backfill_irrational.py …")
        subprocess.run([sys.executable, str(HERE / "backfill_irrational.py")],
                       cwd=str(HERE), check=True)

        print("\n── Pushing to Railway …")
        subprocess.run(["railway", "up", "--detach"],
                       cwd=str(HERE), check=False)
        print("  Done.")


if __name__ == "__main__":
    main()
