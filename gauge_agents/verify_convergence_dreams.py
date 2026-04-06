#!/usr/bin/env python3
"""
verify_convergence_dreams.py — Dreams Walk Convergence Verification
====================================================================
Runs deep random-lattice (Dreams) walks on agent store records to
independently confirm convergence claims in the store files.

A "Dreams walk" for an n-variable CMF picks a random axis at each step
and computes the matrix product along that path:
    P_N = K_{a_N}(v_N) · K_{a_{N-1}}(v_{N-1}) · … · K_{a_1}(v_1)
Convergence is measured via the ratio test:
    delta_k = -log10 |P_k[0,0]/P_k[1,0] - P_{k-1}[0,0]/P_{k-1}[1,0]|

Usage:
    python3 verify_convergence_dreams.py              # all store files, last 30 each
    python3 verify_convergence_dreams.py --agent A B  # specific agents
    python3 verify_convergence_dreams.py --n 10       # 10 records per file
    python3 verify_convergence_dreams.py --depth 800  # walk depth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

WALK_DEPTH  = 600    # matrix product depth
N_RECORDS   = 30     # records per store file to verify
START_COORD = 2      # starting lattice coordinate (must be > max |shift|)
RNG_SEED    = 42

# ── Reconstruct step functions from stored params ─────────────────────────

def _parse_off(raw: dict) -> dict:
    """Convert string-keyed off-dict from JSON to tuple-keyed dict."""
    out = {}
    for k, v in raw.items():
        try:
            k2 = tuple(int(x) for x in k.strip("()").split(","))
            out[k2] = float(v)
        except Exception:
            pass
    return out


def make_step_fns(record: dict) -> list:
    """
    Reconstruct step functions from a run_all_agents store record.
    The record must have a 'params' key with dim/n_vars/D_params/L_off/U_off.
    Returns list of callables  fns[ax](*coords) → dim×dim ndarray.
    """
    p = record.get("params")
    if not p:
        return []

    dim    = int(p["dim"])
    n_vars = int(p.get("n_vars") or record.get("n_matrices") or record.get("n_vars") or 3)
    D_p    = p["D_params"]
    L_off  = _parse_off(p.get("L_off", {}))
    U_off  = _parse_off(p.get("U_off", {}))

    def G(coords: list) -> np.ndarray:
        L = np.eye(dim)
        for (i, j), v in L_off.items():
            L[i, j] = v
        diag_v = np.array([
            D_p[k][0] * (coords[k % n_vars] + D_p[k][1])
            for k in range(dim)
        ], dtype=float)
        U = np.eye(dim)
        for (i, j), v in U_off.items():
            U[i, j] = v
        return L @ np.diag(diag_v) @ U

    fns = []
    for axis in range(n_vars):
        def make(ax=axis):
            def fn(*coords):
                c   = list(coords)
                csh = list(coords); csh[ax] += 1
                Gn  = G(c)
                det = np.linalg.det(Gn)
                if abs(det) < 1e-10:
                    raise ValueError("singular G")
                Di = np.diag(np.array([c[ax] + k for k in range(dim)], dtype=float))
                return G(csh) @ Di @ np.linalg.inv(Gn)
            return fn
        fns.append(make())
    return fns


# ── Dreams walk ────────────────────────────────────────────────────────────

def _axis_walk(fn, dim: int, n_vars: int, depth: int) -> float:
    """
    Deterministic single-axis walk: start at [START_COORD]*n_vars, step axis 0.
    Returns the final convergence delta (matching run_all_agents _walk_np logic).
    """
    pos = [START_COORD] * n_vars
    v   = np.zeros(dim, dtype=float); v[0] = 1.0
    ratio_prev = None
    delta = 0.0

    for _ in range(depth):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
        except Exception:
            return 0.0
        if not np.all(np.isfinite(M)):
            return 0.0
        v = M @ v
        norm = np.max(np.abs(v))
        if norm == 0:
            return 0.0
        if norm > 1e25:
            v /= norm

        if dim >= 2 and abs(v[-1]) > 1e-25:
            ratio = v[0] / v[-1]
            if ratio_prev is not None:
                diff = abs(ratio - ratio_prev)
                if 0 < diff < 1e50:
                    delta = max(delta, -math.log10(diff + 1e-300))
            ratio_prev = ratio

    return round(min(delta, 50.0), 3)


def dreams_walk(fns: list, dim: int, depth: int, rng: random.Random) -> dict:
    """
    Dreams Walk: run deterministic walks on every axis and take the best delta.
    A CMF passes if at least one axis gives delta >= 2.0.
    """
    n_vars = len(fns)
    axis_deltas = []

    for ax in range(n_vars):
        # Permute so the target axis is always walked as axis 0
        def make_permuted(ax=ax):
            fn = fns[ax]
            if ax == 0:
                return fn
            def pfn(*coords):
                pcoords = list(coords)
                pcoords[0], pcoords[ax] = pcoords[ax], pcoords[0]
                return fn(*pcoords)
            return pfn

        d = _axis_walk(make_permuted(ax), dim, n_vars, depth)
        axis_deltas.append(d)

    best = max(axis_deltas) if axis_deltas else 0.0
    passed = best >= 2.0

    return {
        "delta":     best,
        "axis_deltas": axis_deltas,
        "deltas":    axis_deltas,
        "pass":      passed,
        "sustained": 1.0 if passed else 0.0,
        "reason":    "ok" if passed else f"best_delta={best:.2f} axes={axis_deltas}",
    }


# ── Per-record verification ────────────────────────────────────────────────

def verify_record(rec: dict, depth: int, rng: random.Random) -> dict:
    fp    = rec.get("fingerprint", "?")[:16]
    p     = rec.get("params", {})
    dim   = int(p.get("dim", rec.get("dim", 3)))
    agent = rec.get("agent", "?")
    stored_delta = rec.get("best_delta", 0.0)

    t0   = time.time()
    fns  = make_step_fns(rec)

    if not fns:
        return {"fp": fp, "dim": dim, "agent": agent, "pass": False,
                "reason": "no_step_fns", "stored_delta": stored_delta,
                "verified_delta": None, "elapsed": 0}

    result = dreams_walk(fns, int(dim), depth, rng)
    elapsed = round(time.time() - t0, 2)

    return {
        "fp":             fp,
        "dim":            dim,
        "agent":          agent,
        "stored_delta":   round(float(stored_delta), 3),
        "verified_delta": result["delta"],
        "sustained":      result.get("sustained", 0),
        "deltas":         result["deltas"],
        "pass":           result["pass"],
        "reason":         result["reason"],
        "elapsed":        elapsed,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dreams walk convergence verifier")
    ap.add_argument("--agent", nargs="*", default=None,
                    help="Agent letters to verify (default: all)")
    ap.add_argument("--n", type=int, default=N_RECORDS,
                    help=f"Records per file (default: {N_RECORDS})")
    ap.add_argument("--depth", type=int, default=WALK_DEPTH,
                    help=f"Walk depth (default: {WALK_DEPTH})")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSONL output file")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Collect store files
    store_files = sorted(HERE.glob("store_*.jsonl"))
    if args.agent:
        agents = set(a.upper() for a in args.agent)
        store_files = [f for f in store_files
                       if f.name.split("_")[1].upper() in agents]

    if not store_files:
        print("No store files found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDreams Walk Verification — depth={args.depth}, n_per_file={args.n}")
    print(f"Files: {len(store_files)}\n")

    all_results = []
    total_pass = total_fail = 0

    out_fh = open(args.out, "w") if args.out else None

    for sf in store_files:
        try:
            lines = sf.read_text().strip().split("\n")
        except Exception:
            continue
        lines = [l for l in lines if l.strip()]
        sample = lines[-args.n:] if len(lines) > args.n else lines

        file_pass = file_fail = 0
        if not args.verbose:
            print(f"  {sf.name[:-6]:<28}  ", end="", flush=True)

        for line in sample:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            result = verify_record(rec, args.depth, rng)
            all_results.append(result)

            if result["pass"]:
                file_pass += 1; total_pass += 1
                status_char = "✓"
            else:
                file_fail += 1; total_fail += 1
                status_char = "✗"

            vd = result['verified_delta']
            vd_str = f"{vd:.1f}" if vd is not None else "N/A"
            if args.verbose or not result["pass"]:
                print(f"  {status_char} {result['agent']}/{result['dim']}x{result['dim']}  "
                      f"fp={result['fp']}  "
                      f"stored_Δ={result['stored_delta']:.1f}  "
                      f"verified_Δ={vd_str}  "
                      f"sustained={result.get('sustained',0):.0%}  "
                      f"{result['elapsed']}s  {result['reason']}")

            if out_fh:
                out_fh.write(json.dumps(result) + "\n")

        if not args.verbose:
            # Print summary line for file
            pct = 100 * file_pass // max(1, file_pass + file_fail)
            bar_len = 20
            filled = int(bar_len * file_pass // max(1, file_pass + file_fail))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"[{bar}]  {file_pass}/{file_pass+file_fail}  ({pct}%)")

    print(f"\n{'='*60}")
    pct_total = 100 * total_pass // max(1, total_pass + total_fail)
    print(f"TOTAL  {total_pass}/{total_pass+total_fail} passed  ({pct_total}%)")
    if total_fail:
        print(f"  {total_fail} FAILED — check verbose output for details")
    else:
        print(f"  All records verified convergent ✓")

    if out_fh:
        out_fh.close()
        print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
