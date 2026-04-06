#!/usr/bin/env python3
"""
extract_golden_irrational.py — Golden Irrational CMF Dataset Extractor
========================================================================
Scans the store files and atlas DB, applies fast rationality checking to
the convergent limits of every CMF, and writes the top-1% irrational
candidates to data/golden_irrational_dataset.json.

Also ingests confirmed PSLQ hits from /tmp/pslq_golden_hits.jsonl if present.

Usage:
    python3 extract_golden_irrational.py
    python3 extract_golden_irrational.py --out data/golden_irrational_dataset.json
    python3 extract_golden_irrational.py --min-irrat 0.7 --top-n 5000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent

# ── Rationality check (mirrors reward_engine logic) ──────────────────────────

RATIONAL_DENOM_THRESHOLD = 100_000


def fast_rationality_check(r: float, max_denominator: int = RATIONAL_DENOM_THRESHOLD) -> bool:
    if not math.isfinite(r) or abs(r) > 1e12:
        return False
    if abs(r) < 1e-15:
        return True
    frac = Fraction(r).limit_denominator(max_denominator)
    if abs(float(frac) - r) / (abs(r) + 1e-30) < 1e-8:
        return True
    x = abs(r)
    for _ in range(5):
        a = int(x); x -= a
        if x < 1e-9:
            return True
        x = 1.0 / x
    return False


def irrationality_score(ratios: list[float]) -> float:
    valid = [r for r in ratios if math.isfinite(r) and 1e-15 < abs(r) < 1e12]
    if not valid:
        return 0.5
    n_rational = sum(1 for r in valid if fast_rationality_check(r))
    return 1.0 - n_rational / len(valid)


# ── Walk to get convergent limit ──────────────────────────────────────────────

def _quick_limit(fn, n_vars: int, depth: int = 400) -> Optional[float]:
    """Fast float64 walk: returns v[0]/v[-1] limit or None."""
    pos = [2] * n_vars
    v = np.zeros(3, dtype=float); v[0] = 1.0
    dim = None
    for step in range(depth):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
            if dim is None:
                dim = M.shape[0]
                v = np.zeros(dim, dtype=float); v[0] = 1.0
            if not np.all(np.isfinite(M)):
                continue
            v_new = M @ v
        except Exception:
            continue
        n = np.max(np.abs(v_new))
        if n > 1e25:
            v = v_new / n
        elif n < 1e-25:
            return None
        else:
            v = v_new
    if dim is None or abs(v[-1]) < 1e-18:
        return None
    return float(v[0] / v[-1])


def _make_step_fn(record: dict):
    """Build a numpy step function from a store record (D_params or symbolic)."""
    params = record.get("params", {})
    D = params.get("D_params")
    if D is not None:
        # Numeric D_params path
        dim    = len(D[0]) if D else 3
        L_off  = params.get("L_off", [0] * dim)
        U_off  = params.get("U_off", [0] * dim)
        n_vars = len(D)

        def fn(*coords):
            M = np.zeros((dim, dim), dtype=float)
            for r in range(dim):
                for c in range(dim):
                    k = coords[0]
                    m = coords[1] if n_vars > 1 else 2
                    M[r, c] = float(
                        sum(D[ax][r][c] * (coords[ax] if ax < len(coords) else 2)
                            for ax in range(n_vars))
                        + (L_off[r] if c == 0 else 0)
                        + (U_off[r] if c == dim - 1 else 0)
                    )
            return M

        # Use the proper make_step_fns if available
        try:
            sys.path.insert(0, str(HERE))
            from verify_convergence_dreams import make_step_fns
            fns = make_step_fns(record)
            if fns:
                return fns[0], n_vars
        except Exception:
            pass
        return fn, n_vars

    # Symbolic path
    try:
        sys.path.insert(0, str(HERE))
        from verify_convergence_dreams import make_step_fns
        fns = make_step_fns(record)
        n_vars = sum(1 for ax in range(10) if record.get(f"X{ax}") is not None)
        if fns:
            return fns[0], max(n_vars, 1)
    except Exception:
        pass
    return None, 1


# ── PSLQ hits ingestion ───────────────────────────────────────────────────────

def load_pslq_hits(path: str = "/tmp/pslq_golden_hits.jsonl") -> dict[str, list[dict]]:
    """Load confirmed PSLQ hits keyed by fingerprint."""
    hits: dict[str, list[dict]] = {}
    p = Path(path)
    if not p.exists():
        return hits
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fp = rec.get("fingerprint", "")
                if fp:
                    hits.setdefault(fp, []).append(rec)
            except Exception:
                pass
    return hits


# ── Main extraction ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract golden irrational CMF dataset")
    parser.add_argument("--out",      default="data/golden_irrational_dataset.json",
                        help="Output JSON path")
    parser.add_argument("--min-irrat", type=float, default=0.6,
                        help="Minimum irrationality score to include (0-1)")
    parser.add_argument("--top-n",    type=int,   default=0,
                        help="Keep only top-N by irrationality score (0 = all)")
    parser.add_argument("--pslq-hits", default="/tmp/pslq_golden_hits.jsonl",
                        help="Path to PSLQ golden hits JSONL")
    parser.add_argument("--depth",    type=int,   default=400,
                        help="Walk depth for limit estimation")
    args = parser.parse_args()

    out_path = HERE.parent / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PSLQ confirmed hits (already irrational by definition)
    pslq_hits = load_pslq_hits(args.pslq_hits)
    print(f"Loaded PSLQ confirmed hits for {len(pslq_hits):,} CMFs")

    # Scan all store files
    store_files = sorted(HERE.glob("store_*.jsonl"))
    print(f"Scanning {len(store_files)} store files …")

    golden: list[dict] = []
    n_processed = 0
    n_trivial   = 0
    n_irrational = 0
    t0 = time.time()

    for sf in store_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                n_processed += 1
                fp = rec.get("fingerprint", "")

                # If we have a confirmed PSLQ hit → always golden
                pslq = pslq_hits.get(fp, [])

                fn, n_vars = _make_step_fn(rec)
                ratios: list[float] = []
                if fn is not None:
                    for _ in range(3):  # 3 starting points
                        r = _quick_limit(fn, n_vars, depth=args.depth)
                        if r is not None:
                            ratios.append(r)

                irrat = irrationality_score(ratios)

                if n_processed % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = n_processed / max(elapsed, 1)
                    print(f"  {n_processed:>7,} processed  irrational={n_irrational:,}  "
                          f"trivial={n_trivial:,}  {rate:.1f} cmf/s", flush=True)

                # Always include PSLQ-confirmed hits
                if pslq:
                    n_irrational += 1
                    golden.append({
                        "fingerprint":       fp,
                        "agent":             rec.get("agent", "?"),
                        "dim":               rec.get("dim", 3),
                        "irrationality_score": 1.0,
                        "irrationality_label": "PSLQ_CONFIRMED",
                        "pslq_hits":         pslq,
                        "ratios":            [round(r, 12) for r in ratios],
                        "params":            rec.get("params", {}),
                        "X0": rec.get("X0"), "X1": rec.get("X1"), "X2": rec.get("X2"),
                        "f_poly":    rec.get("f_poly"),
                        "fbar_poly": rec.get("fbar_poly"),
                        "source":    rec.get("source"),
                        "certification_level": rec.get("certification_level"),
                    })
                    continue

                if irrat < args.min_irrat:
                    n_trivial += 1
                    continue

                n_irrational += 1
                golden.append({
                    "fingerprint":       fp,
                    "agent":             rec.get("agent", "?"),
                    "dim":               rec.get("dim", 3),
                    "irrationality_score": round(irrat, 4),
                    "irrationality_label": "IRRATIONAL_CANDIDATE",
                    "pslq_hits":         [],
                    "ratios":            [round(r, 12) for r in ratios],
                    "params":            rec.get("params", {}),
                    "X0": rec.get("X0"), "X1": rec.get("X1"), "X2": rec.get("X2"),
                    "f_poly":    rec.get("f_poly"),
                    "fbar_poly": rec.get("fbar_poly"),
                    "source":    rec.get("source"),
                    "certification_level": rec.get("certification_level"),
                })

    # Sort by irrationality score descending
    golden.sort(key=lambda x: x["irrationality_score"], reverse=True)
    if args.top_n > 0:
        golden = golden[:args.top_n]

    output = {
        "metadata": {
            "total_processed": n_processed,
            "total_trivially_rational": n_trivial,
            "total_irrational_candidates": n_irrational,
            "pslq_confirmed": len(pslq_hits),
            "min_irrationality_score": args.min_irrat,
            "entries_in_dataset": len(golden),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "entries": golden,
    }

    with open(out_path, "w") as fout:
        json.dump(output, fout, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total processed:       {n_processed:>8,}")
    print(f"Trivially rational:    {n_trivial:>8,}  ({100*n_trivial/max(n_processed,1):.1f}%)")
    print(f"Irrational candidates: {n_irrational:>8,}  ({100*n_irrational/max(n_processed,1):.1f}%)")
    print(f"PSLQ confirmed:        {len(pslq_hits):>8,}")
    print(f"Golden dataset size:   {len(golden):>8,}")
    print(f"Output: {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
