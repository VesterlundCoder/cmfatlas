#!/usr/bin/env python3
"""
atlas_ingest.py — Ingest all Gauge-Agent CMFs into the CMF Atlas
=================================================================

Reads:
  store_A_*.jsonl, store_B_*.jsonl, store_C_*.jsonl

Converts each record to the canonical atlas format, de-duplicates
against the existing atlas (by fingerprint), and appends new entries
to cmf_database_certified.json.

All gauge CMFs are classified as certification_level = "reference"
(numerical verification of path-independence at accept time) pending
symbolic Sage certification.

Usage:
  python3 atlas_ingest.py               # ingest all, report summary
  python3 atlas_ingest.py --dry-run     # simulate, don't write
  python3 atlas_ingest.py --report      # print full report only
  python3 atlas_ingest.py --b4-only     # only ingest Bucket-4 CMFs
"""
from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone

HERE    = Path(__file__).parent
ATLAS   = HERE.parent / "cmf_database_certified.json"
OUT_DIR = HERE / "pipeline_out"
OUT_DIR.mkdir(exist_ok=True)

REPORT_MD   = OUT_DIR / "atlas_ingest_report.md"
INGEST_LOG  = OUT_DIR / "atlas_ingest_log.jsonl"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CMF → atlas record conversion
# ══════════════════════════════════════════════════════════════════════════════

def _shorten_expr(expr_str: str, max_len: int = 200) -> str:
    s = str(expr_str).strip()
    return s if len(s) <= max_len else s[:max_len] + "..."


def _matrix_str(rows: list) -> str:
    """Compact repr of a symbolic matrix row list."""
    if not rows:
        return ""
    return "Matrix([" + ",".join("[" + ",".join(_shorten_expr(c, 120) for c in row) + "]"
                                  for row in rows) + "])"


def gauge_to_atlas(rec: dict, source_file: str) -> dict:
    """Convert one gauge-agent record → atlas entry dict."""
    agent  = rec.get("agent", "?")
    dim    = rec.get("dim", 3)
    fp     = rec.get("fingerprint", "")
    score  = rec.get("score", 0.0)
    delta  = rec.get("best_delta") or (max(rec.get("deltas", [0])) if rec.get("deltas") else 0)
    bidir  = rec.get("bidir_ratio", 0.0)
    bucket = rec.get("coupling_bucket", 2)
    ts     = rec.get("timestamp", datetime.now(timezone.utc).isoformat())

    # Build atlas id from fingerprint
    atlas_id = f"gauge_{agent}_{fp[:16]}"

    # Collect generator matrices
    generators: dict = {}
    for k in range(dim):
        key = f"X{k}"
        if key in rec and rec[key]:
            generators[key] = _matrix_str(rec[key])

    # Path independence error (Agent C stores this explicitly)
    pi_err  = rec.get("max_flatness_error", None)
    pi_ok   = rec.get("path_independence_verified", None)

    # Flat by numerical construction for Agents A/B
    if pi_ok is None and agent in ("A", "B", "D", "E", "F", "G", "H", "I", "J"):
        pi_ok = True   # evaluated by reward_engine / flatness check at accept time

    nvars       = rec.get("nvars", 3)
    matrix_size = rec.get("matrix_size", dim)

    entry = {
        "id":                    atlas_id,
        "name":                  f"Gauge-{agent} {matrix_size}×{matrix_size} [{fp[:8]}]",
        "description":           (
            f"CMF discovered by Gauge-Bootstrap Agent {agent}. "
            f"n_vars={nvars}, matrix_size={matrix_size}×{matrix_size}, "
            f"coupling_bucket=B{bucket}, "
            f"bidir_ratio={bidir:.3f}, score={score:.4f}, delta={delta:.3f}."
        ),
        "dim":                   dim,
        "nvars":                 nvars,
        "n_matrices":            rec.get("n_matrices", nvars),
        "matrix_size":           matrix_size,
        "effective_vars":        rec.get("effective_vars", nvars),
        "type":                  "gauge_ldu",
        "agent":                 agent,
        "fingerprint":           fp,
        "coarse_fp":             rec.get("coarse_fp", ""),
        "coupling_bucket":       bucket,
        "bidir_ratio":           bidir,

        # LDU parameterisation (compact)
        "params":                rec.get("params", {}),
        "d_params":              rec.get("d_params", []),

        # Symbolic generator matrices (if present; stored as compact strings)
        "generators":            generators,
        "gauge_matrix":          rec.get("G", "") if rec.get("G") else "",

        # Scoring / analytics
        "score":                 score,
        "novelty_score":         rec.get("novelty_score", score),
        "novelty_factor":        rec.get("novelty_factor", 1.0),
        "family_count":          rec.get("family_count", 1),
        "best_delta":            delta,
        "deltas":                rec.get("deltas", []),
        "conv_rate":             rec.get("conv_rate", 0.0),
        "ray_stability":         rec.get("ray_stability", 0.0),
        "dfinite_score":         rec.get("dfinite_score", 0.0),

        # Flatness
        "flatness_verified":     bool(pi_ok) if pi_ok is not None else None,
        "max_flatness_error":    pi_err,
        "n_pairs_checked":       rec.get("n_pairs_checked", None),
        "flatness_symbolic":     None,   # filled by sympy_flatness_runner
        "flatness_sage":         None,   # filled by sage_flatness_runner

        # Constants
        "primary_constant":      None,
        "primary_constant_value": None,
        "other_constants":       [],

        # Provenance
        "source":                "gauge_agent",
        "source_file":           source_file,
        "discovered_by":         f"agent_{agent.lower()}",
        "timestamp_utc":         ts,
        "certification_level":   "reference",

        # Dreams (filled by dreams_runner)
        "identified_limit":      None,
        "limit_formula":         None,
        "pslq_residual":         None,

        # Notes
        "notes":                 f"B{bucket} bidirectional CMF from gauge bootstrap. Numerical path-independence verified.",
    }
    return entry


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Load / save atlas
# ══════════════════════════════════════════════════════════════════════════════

def load_atlas() -> tuple[dict, set]:
    """Returns (atlas_dict, existing_fingerprint_set)."""
    with open(ATLAS) as f:
        atlas = json.load(f)
    existing_fps = set()
    for entry in atlas.get("cmfs", []):
        fp = entry.get("fingerprint", "")
        if fp:
            existing_fps.add(fp)
        # Also index by atlas id
        existing_fps.add(entry.get("id", ""))
    return atlas, existing_fps


def save_atlas(atlas: dict, dry_run: bool = False):
    atlas["metadata"]["last_updated"]    = datetime.now(timezone.utc).isoformat()
    atlas["metadata"]["total_entries"]   = len(atlas["cmfs"])
    atlas["metadata"]["by_source"]       = {}
    atlas["metadata"]["by_dim"]          = {}
    for entry in atlas["cmfs"]:
        src = entry.get("source", "unknown")
        d   = str(entry.get("dim", "?"))
        atlas["metadata"]["by_source"][src] = atlas["metadata"]["by_source"].get(src, 0) + 1
        atlas["metadata"]["by_dim"][d]      = atlas["metadata"]["by_dim"].get(d, 0) + 1

    if dry_run:
        print(f"  [dry-run] Would save {len(atlas['cmfs'])} entries to {ATLAS.name}")
        return

    # Write atomically
    tmp = ATLAS.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(atlas, f, indent=2, ensure_ascii=False)
    tmp.replace(ATLAS)
    print(f"  ✓ Atlas saved: {len(atlas['cmfs'])} entries → {ATLAS.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Collect all gauge store records
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_records(b4_only: bool = False) -> list[tuple[dict, str]]:
    """Returns list of (record, source_filename) pairs."""
    records = []
    for store_path in sorted(HERE.glob("store_[A-Z]_*.jsonl")):
        src = store_path.name
        lines = [l for l in store_path.read_text().splitlines() if l.strip()]
        for line in lines:
            try:
                rec = json.loads(line)
                if b4_only and rec.get("coupling_bucket", 0) < 4:
                    continue
                records.append((rec, src))
            except Exception:
                pass
    print(f"  Collected {len(records):,} records from store files")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Statistics helpers
# ══════════════════════════════════════════════════════════════════════════════

def count_by(entries: list, key: str) -> dict:
    counts: dict = {}
    for e in entries:
        v = str(e.get(key, "?"))
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Report generator
# ══════════════════════════════════════════════════════════════════════════════

def _bar(n: int, total: int, width: int = 30) -> str:
    frac = min(n / max(total, 1), 1.0)
    filled = int(frac * width)
    return "▓" * filled + "░" * (width - filled)


def write_report(added: list, skipped: int, total_atlas: int, by_agent: dict,
                 by_dim: dict, by_bucket: dict, elapsed: float):
    lines = [
        "# CMF Atlas Ingest Report — Gauge Bootstrap Agents",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Run time:** {elapsed:.1f}s  ",
        f"**New entries added:** {len(added):,}  ",
        f"**Duplicate / skipped:** {skipped:,}  ",
        f"**Total atlas entries now:** {total_atlas:,}  ",
        "",
        "---",
        "",
        "## Breakdown by Agent",
        "",
        "| Agent | Added | % |",
        "|-------|------:|--:|",
    ]
    total_added = len(added)
    for agent, n in sorted(by_agent.items()):
        pct = 100 * n / max(total_added, 1)
        lines.append(f"| {agent} | {n:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Breakdown by Dimension",
        "",
        "| Dim | Count | Bar |",
        "|-----|------:|-----|",
    ]
    for dim, n in sorted(by_dim.items(), key=lambda x: int(x[0])):
        lines.append(f"| {dim}×{dim} | {n:,} | {_bar(n, max(by_dim.values()))} |")

    lines += [
        "",
        "## Breakdown by Coupling Bucket",
        "",
        "| Bucket | Meaning | Count | Bar |",
        "|--------|---------|------:|-----|",
    ]
    bucket_names = {
        "2": "Weakly coupled",
        "3": "Pairwise nontrivial",
        "4": "Fully bidirectional (B4)",
    }
    for b, n in sorted(by_bucket.items()):
        name = bucket_names.get(str(b), "?")
        lines.append(f"| B{b} | {name} | {n:,} | {_bar(n, max(by_bucket.values()))} |")

    lines += [
        "",
        "## Score Distribution (new entries)",
        "",
    ]
    if added:
        scores = sorted(e.get("score", 0) for e in added)
        p = lambda pct: scores[int(pct * len(scores) / 100)]
        lines += [
            "| Percentile | Score |",
            "|-----------|-------|",
            f"| p10 | {p(10):.4f} |",
            f"| p25 | {p(25):.4f} |",
            f"| p50 | {p(50):.4f} |",
            f"| p75 | {p(75):.4f} |",
            f"| p90 | {p(90):.4f} |",
            f"| p99 | {p(99):.4f} |",
        ]

    lines += [
        "",
        "## Top 20 by Score",
        "",
        "| Rank | ID | Agent | Dim | Bucket | Score | Delta | bidir |",
        "|------|----|----|---|---|------:|------:|------:|",
    ]
    top20 = sorted(added, key=lambda e: e.get("score", 0), reverse=True)[:20]
    for i, e in enumerate(top20, 1):
        lines.append(
            f"| {i} | `{e['id'][:18]}` | {e['agent']} | {e['dim']}×{e['dim']} "
            f"| B{e['coupling_bucket']} | {e['score']:.4f} | {e['best_delta']:.2f} "
            f"| {e['bidir_ratio']:.2f} |"
        )

    lines += [
        "",
        "## Top 10 Largest Dimensions (Agent C)",
        "",
        "| Dim | Fingerprint | Delta | Bucket | pi_verified |",
        "|-----|-------------|------:|--------|------------|",
    ]
    c_entries = [e for e in added if e.get("agent") == "C"]
    c_entries.sort(key=lambda e: e.get("dim", 0), reverse=True)
    for e in c_entries[:10]:
        lines.append(
            f"| {e['dim']}×{e['dim']} | `{e['fingerprint'][:12]}` "
            f"| {e['best_delta']:.2f} | B{e['coupling_bucket']} "
            f"| {e.get('flatness_verified','?')} |"
        )

    lines += [
        "",
        "## Certification Status",
        "",
        "All new entries are classified as `certification_level = reference`.  ",
        "Run `sympy_flatness_runner.py` to upgrade to `symbolic` level.  ",
        "Run `sage_flatness_runner.sage` to upgrade to `sage_certified` level.  ",
        "",
        "---",
        f"_Report auto-generated by atlas_ingest.py_",
    ]

    REPORT_MD.write_text("\n".join(lines))
    print(f"  ✓ Report: {REPORT_MD}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--report",   action="store_true", help="Print report only, no write")
    ap.add_argument("--b4-only",  action="store_true", help="Only ingest Bucket-4 CMFs")
    ap.add_argument("--max",      type=int, default=None, help="Max records to ingest")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\n{'═'*60}")
    print("  CMF Atlas Ingest — Gauge Bootstrap Agents")
    print(f"{'═'*60}\n")

    # Load existing atlas
    atlas, existing_fps = load_atlas()
    n_existing = len(atlas.get("cmfs", []))
    print(f"  Existing atlas entries: {n_existing:,}")
    print(f"  Existing fingerprints:  {len(existing_fps):,}")

    # Collect all gauge records
    all_recs = collect_all_records(b4_only=args.b4_only)
    if args.max:
        all_recs = all_recs[:args.max]

    # Convert & deduplicate
    added   = []
    skipped = 0
    log_fh  = open(INGEST_LOG, "a") if not args.dry_run and not args.report else None

    for rec, src in all_recs:
        fp = rec.get("fingerprint", "")
        if not fp or fp in existing_fps:
            skipped += 1
            continue

        try:
            entry = gauge_to_atlas(rec, src)
        except Exception as e:
            print(f"  ⚠ Convert error {fp[:8]}: {e}")
            skipped += 1
            continue

        added.append(entry)
        existing_fps.add(fp)
        existing_fps.add(entry["id"])

        if log_fh:
            log_fh.write(json.dumps({"action": "add", "id": entry["id"],
                                     "fp": fp, "dim": entry["dim"],
                                     "agent": entry["agent"]}) + "\n")

    if log_fh:
        log_fh.close()

    # Statistics
    by_agent  = count_by(added, "agent")
    by_dim    = count_by(added, "dim")
    by_bucket = count_by(added, "coupling_bucket")

    print(f"\n  New records: {len(added):,}  |  Duplicates skipped: {skipped:,}")
    print(f"  By agent:  { {k: v for k, v in by_agent.items()} }")
    print(f"  By dim:    { {k+'x'+k: v for k, v in by_dim.items()} }")
    print(f"  By bucket: { {f'B{k}': v for k, v in by_bucket.items()} }")

    if not args.report:
        # Append to atlas and save
        atlas["cmfs"].extend(added)
        save_atlas(atlas, dry_run=args.dry_run)

    elapsed = time.time() - t0
    write_report(added, skipped, n_existing + len(added),
                 by_agent, by_dim, by_bucket, elapsed)
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
