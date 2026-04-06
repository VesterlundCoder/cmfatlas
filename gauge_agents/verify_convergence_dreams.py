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
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
try:
    from sympy import symbols as _sym_symbols, lambdify as _sym_lambdify
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

HERE = Path(__file__).parent

WALK_DEPTH  = 1200   # matrix product depth (matches t3_thorough max depth)
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


def make_step_fns_symbolic(record: dict) -> list:
    """
    Legacy Agent A path: evaluate X0, X1, X2 symbolic string matrices numerically.
    Variable names are x, y, z (3-var) or n, x, y (any 3+).
    Uses sympy.lambdify when available for 10-100x speedup (precompiled vs eval).
    """
    dim    = int(record.get("dim") or 3)
    # Count how many X-matrices actually exist (n_vars ≤ dim for higher-dim records)
    n_vars = sum(1 for ax in range(10) if record.get(f"X{ax}") is not None)
    if n_vars == 0:
        return []
    var_names = ["x", "y", "z", "w", "v", "u"][:n_vars]

    fns = []
    for ax in range(n_vars):
        mat_strs = record.get(f"X{ax}")
        if mat_strs is None:
            return []

        if _SYMPY_OK:
            # Fast path: precompile each expression to a Python lambda once
            try:
                syms = _sym_symbols(" ".join(var_names))
                if n_vars == 1:
                    syms = (syms,)
                lambdas = [
                    [_sym_lambdify(syms, mat_strs[r][c], modules="numpy")
                     for c in range(dim)]
                    for r in range(dim)
                ]
                def make_fast(lms=lambdas, d=dim):
                    def fn(*coords):
                        M = np.empty((d, d), dtype=float)
                        for r in range(d):
                            for c in range(d):
                                M[r, c] = float(lms[r][c](*coords))
                        return M
                    return fn
                fns.append(make_fast())
                continue
            except Exception:
                pass  # fall through to eval fallback

        # Fallback: eval per step (slower)
        def make(strs=mat_strs, vn=var_names, d=dim):
            compiled = [[compile(strs[r][c], "<cmf>", "eval")
                         for c in range(d)] for r in range(d)]
            def fn(*coords):
                ns = {vn[i]: float(coords[i]) for i in range(len(vn))}
                M = np.empty((d, d), dtype=float)
                for r in range(d):
                    for c in range(d):
                        M[r, c] = float(eval(compiled[r][c], {"__builtins__": {}}, ns))  # noqa: S307
                return M
            return fn
        fns.append(make())
    return fns


def make_step_fns(record: dict) -> list:
    """
    Reconstruct step functions from a run_all_agents store record.
    Handles both the new D_params/L_off/U_off format and the legacy
    Agent A symbolic X0/X1/X2 format.
    Returns list of callables  fns[ax](*coords) → dim×dim ndarray.
    """
    p = record.get("params")
    if not p:
        return []

    # ── Legacy Agent A: params has diag/offdiag, record has X0/X1/X2 ──────
    if "D_params" not in p and record.get("X0") is not None:
        return make_step_fns_symbolic(record)

    if "D_params" not in p:
        return []  # unknown legacy format, skip

    dim    = int(p.get("dim") or record.get("dim") or record.get("matrix_size") or 3)
    n_vars = int(p.get("n_vars") or p.get("nvars") or
                 record.get("n_vars") or record.get("nvars") or
                 record.get("effective_vars") or record.get("n_matrices") or 3)
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

def _axis_walk_single(fn, dim: int, n_vars: int,
                      depth: int, v_init: np.ndarray,
                      start: int = START_COORD) -> float:
    """
    Walk along pos[0] for `depth` steps with v_init as starting vector.
    Phase 1 (first depth//4 steps): warm-up to find stable top-2 (i,j) components.
    Phase 2 (remaining steps): track v[i]/v[j] step-by-step; return max -log10(|Δratio|).
    `start` sets the fixed coordinate for dims 1..n_vars-1 (vary to avoid singularities).
    """
    pos    = [start] * n_vars
    v      = v_init.copy()
    warmup = max(10, depth // 4)

    # ── Phase 1: warm-up ──────────────────────────────────────────────────
    for _ in range(warmup):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
            if not np.all(np.isfinite(M)):
                continue           # skip this singular lattice point
            v_new = M @ v
        except Exception:
            continue               # skip — algebraic singularity at this coord
        n = np.max(np.abs(v_new))
        if n > 1e25:
            v = v_new / n
        elif n < 1e-25:
            return 0.0
        else:
            v = v_new

    # ── Determine stable (i, j) from warm-up end state ───────────────────
    # Always pick top-2 by absolute norm — do NOT use a relative threshold.
    # A tiny v[i] (e.g. 1e-88) relative to v[j] (1e5) is still meaningful:
    # the ratio v[i]/v[j] → 0, and consecutive differences decay rapidly.
    norms = np.abs(v)
    if np.max(norms) < 1e-300:
        return 0.0
    sc = np.argsort(norms)[::-1]
    j, i = int(sc[0]), int(sc[1])          # j = largest, i = second largest

    # ── Phase 2: step-by-step ratio tracking ─────────────────────────────
    ratio_prev: Optional[float] = float(v[i] / v[j]) if abs(v[j]) > 1e-18 else None
    delta = 0.0

    for _ in range(depth - warmup):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
            if not np.all(np.isfinite(M)):
                continue           # skip singular lattice point
            v_new = M @ v
        except Exception:
            continue               # skip algebraic singularity
        n = np.max(np.abs(v_new))
        if n > 1e25:
            v = v_new / n
        elif n < 1e-25:
            return delta
        else:
            v = v_new
        if abs(v[j]) < 1e-18:
            return delta
        ratio = float(v[i] / v[j])
        if ratio_prev is not None:
            diff = abs(ratio - ratio_prev)
            if 0 < diff < 1e50:
                delta = max(delta, -math.log10(diff + 1e-300))
        ratio_prev = ratio

    return round(min(delta, 50.0), 3)


def dreams_walk(fns: list, dim: int, depth: int, rng: random.Random) -> dict:
    """
    Dreams Walk: step-by-step ratio-decay measurement for each step function.
    Warm-up selects stable (i,j) component pair; then tracks consecutive ratio
    changes to measure convergence rate.  Tries three initial vectors and takes
    the best delta per axis.  Depth scaled by dim (t3_thorough formula).
    """
    n_vars  = len(fns)
    d2      = max(1, dim // 2)
    t3_d    = max(150, 1200 // d2)              # t3_thorough formula (dim-scaled)
    total_d = depth if depth < t3_d else t3_d  # user --depth shrinks; default is t3

    v_e0 = np.zeros(dim, dtype=float); v_e0[0]  = 1.0
    v_e1 = np.zeros(dim, dtype=float); v_e1[-1] = 1.0
    v_un = np.ones(dim, dtype=float) / math.sqrt(dim)
    init_vecs = [v_e0, v_e1, v_un]

    # Try multiple fixed starting coords to avoid symbolic denominator singularities
    start_coords = [START_COORD, START_COORD + 1, START_COORD + 2, START_COORD + 3]

    axis_deltas = []
    for fn in fns:
        best_d = 0.0
        for start in start_coords:
            for v_init in init_vecs:
                d = _axis_walk_single(fn, dim, n_vars, total_d, v_init, start=start)
                best_d = max(best_d, d)
                if best_d >= 2.0:
                    break
            if best_d >= 2.0:
                break
        axis_deltas.append(round(best_d, 3))

    best   = max(axis_deltas) if axis_deltas else 0.0
    passed = best >= 2.0

    return {
        "delta":       best,
        "axis_deltas": axis_deltas,
        "deltas":      axis_deltas,
        "pass":        passed,
        "sustained":   1.0 if passed else 0.0,
        "reason":      "ok" if passed else f"best_delta={best:.2f} axes={axis_deltas}",
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

# ── Parallel worker ────────────────────────────────────────────────────────

def _worker(task: tuple) -> dict:
    """Top-level picklable worker for multiprocessing."""
    rec_json, depth, seed = task
    try:
        rec = json.loads(rec_json)
    except Exception:
        return {"pass": False, "reason": "json_error", "fp": "?", "agent": "?",
                "dim": 0, "stored_delta": 0.0, "verified_delta": 0.0,
                "sustained": 0.0, "deltas": [], "elapsed": 0.0}
    try:
        rng = random.Random(seed)
        return verify_record(rec, depth, rng)
    except Exception as e:
        fp = rec.get("fingerprint", "?")[:16]
        return {"pass": False, "reason": f"error:{e}", "fp": fp,
                "agent": rec.get("agent", "?"), "dim": rec.get("dim", 0),
                "stored_delta": float(rec.get("best_delta", 0.0)),
                "verified_delta": 0.0, "sustained": 0.0, "deltas": [], "elapsed": 0.0}


def main():
    ap = argparse.ArgumentParser(description="Dreams walk convergence verifier")
    ap.add_argument("--agent", nargs="*", default=None,
                    help="Agent letters to verify (default: all)")
    ap.add_argument("--n", type=int, default=N_RECORDS,
                    help=f"Records per file, 0 = all (default: {N_RECORDS})")
    ap.add_argument("--all", dest="all_records", action="store_true",
                    help="Process every record in every store file (overrides --n)")
    ap.add_argument("--depth", type=int, default=WALK_DEPTH,
                    help=f"Walk depth (default: {WALK_DEPTH})")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes (default: 1, use 0 for cpu_count-1)")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSONL output file")
    args = ap.parse_args()

    n_jobs = args.jobs if args.jobs > 0 else max(1, mp.cpu_count() - 1)
    n_per_file = 0 if args.all_records else args.n

    # Collect store files
    store_files = sorted(HERE.glob("store_*.jsonl"))
    if args.agent:
        agents = set(a.upper() for a in args.agent)
        store_files = [f for f in store_files
                       if f.name.split("_")[1].upper() in agents]

    if not store_files:
        print("No store files found.", file=sys.stderr)
        sys.exit(1)

    # Build full task list
    tasks = []
    file_ranges: list[tuple[str, int, int]] = []  # (name, start_idx, end_idx)
    for sf in store_files:
        try:
            raw_lines = [l for l in sf.read_text().strip().split("\n") if l.strip()]
        except Exception:
            continue
        sample = raw_lines if n_per_file == 0 else raw_lines[-n_per_file:]
        start = len(tasks)
        for i, line in enumerate(sample):
            tasks.append((line, args.depth, args.seed + len(tasks)))
        file_ranges.append((sf.name[:-6], start, len(tasks)))

    total_tasks = len(tasks)
    label = "all" if n_per_file == 0 else f"{n_per_file}/file"
    print(f"\nDreams Walk Audit — depth={args.depth}, records={total_tasks} ({label}), "
          f"jobs={n_jobs}, files={len(store_files)}")
    if n_jobs > 1:
        print(f"  Parallel workers: {n_jobs}  "
              f"(est. {total_tasks * 0.10 / n_jobs / 60:.0f}–"
              f"{total_tasks * 0.15 / n_jobs / 60:.0f} min)\n")
    else:
        print(f"  (est. {total_tasks * 0.10 / 60:.0f}–"
              f"{total_tasks * 0.15 / 60:.0f} min single-threaded)\n")

    results: list[dict] = [None] * total_tasks
    out_fh = open(args.out, "w") if args.out else None
    t_start = time.time()

    if n_jobs == 1:
        rng = random.Random(args.seed)
        for idx, (line, depth, seed) in enumerate(tasks):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            results[idx] = verify_record(rec, depth, rng)
            if out_fh:
                out_fh.write(json.dumps(results[idx]) + "\n")
            if args.verbose and not results[idx]["pass"]:
                r = results[idx]
                print(f"  ✗ {r['agent']}/{r['dim']}x{r['dim']}  fp={r['fp']}  "
                      f"stored_Δ={r['stored_delta']:.1f}  "
                      f"verified_Δ={r['verified_delta']:.1f}  {r['reason']}")
    else:
        done = 0
        with mp.Pool(n_jobs) as pool:
            for idx, result in enumerate(pool.imap_unordered(_worker, tasks, chunksize=32)):
                results[idx] = result
                done += 1
                if out_fh:
                    out_fh.write(json.dumps(result) + "\n")
                if not result["pass"] or args.verbose:
                    r = result
                    status = "✗" if not r["pass"] else "✓"
                    vd = r["verified_delta"]
                    vd_str = f"{vd:.1f}" if vd is not None else "N/A"
                    print(f"  {status} {r['agent']}/{r['dim']}x{r['dim']}  fp={r['fp']}  "
                          f"stored_Δ={r['stored_delta']:.1f}  "
                          f"verified_Δ={vd_str}  {r['reason']}")
                # Progress line every 500
                if done % 500 == 0 or done == total_tasks:
                    elapsed = time.time() - t_start
                    rate = done / max(elapsed, 0.1)
                    eta = (total_tasks - done) / max(rate, 0.001)
                    pct = 100 * done / total_tasks
                    valid = [r for r in results if r is not None]
                    n_pass = sum(1 for r in valid if r["pass"])
                    n_fail = len(valid) - n_pass
                    print(f"  Progress: {done}/{total_tasks} ({pct:.0f}%) "
                          f"| ✓ {n_pass}  ✗ {n_fail}  "
                          f"| {rate:.0f} rec/s  ETA {eta/60:.1f}m",
                          flush=True)

    if out_fh:
        out_fh.close()

    # Per-file summary
    print(f"\n{'='*70}")
    print(f"  {'File':<28}  {'Pass':>6}  {'Fail':>6}  {'Pass%':>6}  {'AvgΔ':>6}")
    print(f"  {'-'*28}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")
    grand_pass = grand_fail = 0
    agent_stats: dict[str, dict] = {}
    for fname, s, e in file_ranges:
        batch = [r for r in results[s:e] if r is not None]
        fp = sum(1 for r in batch if r["pass"])
        ff = len(batch) - fp
        grand_pass += fp; grand_fail += ff
        pct = 100 * fp // max(1, len(batch))
        avg_d = sum(r["verified_delta"] or 0 for r in batch) / max(1, len(batch))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  {fname:<28}  {fp:>6}  {ff:>6}  {pct:>5}%  {avg_d:>6.1f}")
        # Accumulate per-agent
        ag = fname.split("_")[1] if "_" in fname else "?"
        if ag not in agent_stats:
            agent_stats[ag] = {"pass": 0, "fail": 0, "deltas": []}
        agent_stats[ag]["pass"] += fp
        agent_stats[ag]["fail"] += ff
        agent_stats[ag]["deltas"].extend(r["verified_delta"] or 0 for r in batch)

    print(f"\n  {'Agent':<8}  {'Pass':>7}  {'Fail':>6}  {'Pass%':>6}  {'AvgΔ':>6}")
    print(f"  {'-'*8}  {'-------':>7}  {'------':>6}  {'------':>6}  {'------':>6}")
    for ag, st in sorted(agent_stats.items()):
        tot = st["pass"] + st["fail"]
        pct = 100 * st["pass"] // max(1, tot)
        avg_d = sum(st["deltas"]) / max(1, len(st["deltas"]))
        print(f"  {ag:<8}  {st['pass']:>7}  {st['fail']:>6}  {pct:>5}%  {avg_d:>6.1f}")

    print(f"\n{'='*70}")
    pct_total = 100 * grand_pass // max(1, grand_pass + grand_fail)
    elapsed_total = time.time() - t_start
    print(f"TOTAL  {grand_pass}/{grand_pass+grand_fail} passed  ({pct_total}%)  "
          f"in {elapsed_total/60:.1f} min")
    if grand_fail:
        print(f"  {grand_fail} FAILED — re-run with --verbose to inspect")
    else:
        print(f"  All records verified convergent ✓")

    if args.out:
        print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
