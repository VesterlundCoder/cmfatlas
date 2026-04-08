#!/usr/bin/env python3
"""
irrational_scout.py — Run agents A→J sequentially, 10 new CMFs each.
After every discovered CMF, walk to its limit value and check irrationality.
Symbolic matrices (X0/X1/X2/…) are computed via SymPy and stored in the record.

Usage:
    python3 gauge_agents/irrational_scout.py
    python3 gauge_agents/irrational_scout.py --agents A B C --target 5
    python3 gauge_agents/irrational_scout.py --target 3 --no-ingest
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

HERE   = Path(__file__).parent
ROOT   = HERE.parent
sys.path.insert(0, str(HERE))

from run_all_agents import (
    AGENT_CONFIGS,
    sample_params, build_eval_fns,
    t1_pole_check, t2_fast, t3_thorough, t4_flatness,
    fingerprint, bidir_ratio, load_seen,
    DELTA_FAST_MIN, DELTA_FULL_MIN, store_path, SENTINEL,
)
from asymptotic_filter import asymptotic_filter
from reward_engine import fast_rationality_check, classify_limit_strict

PYTHON = sys.executable
INGEST = ROOT / "ingest_gauge_agents.py"
DB     = ROOT / "data" / "atlas_2d.db"

ID_DPS   = 60    # precision for limit-value walk
ID_DEPTH = 1200  # walk depth for identification


# ══════════════════════════════════════════════════════════════════════════════
# Symbolic Matrix Builder
# ══════════════════════════════════════════════════════════════════════════════

_VARNAMES = ["x", "y", "z", "w", "v"]

def _as_rational(v: float) -> sp.Rational:
    """Convert float parameter to exact SymPy Rational."""
    f = Fraction(v).limit_denominator(1000)
    return sp.Rational(f.numerator, f.denominator)


def build_symbolic_matrices(params: dict) -> dict[str, list]:
    """
    Compute symbolic K-matrices X0, X1, ... for the CMF defined by params.
    X_i(coords) = G(coords + e_i) · D_i(coords[i]) · G(coords)^{-1}
    Returns {"X0": rows, "X1": rows, ...} where rows = list[list[str]].
    Skips if dim > 5 (too slow for symbolic inversion).
    """
    dim    = params["dim"]
    n_vars = params["n_vars"]
    D_p    = params["D_params"]
    L_off  = params["L_off"]
    U_off  = params["U_off"]

    if dim > 5:
        return {}   # skip large matrices — symbolic inversion is too slow

    coords = sp.symbols(_VARNAMES[:n_vars])
    if n_vars == 1:
        coords = (coords,)

    def make_G(cvars):
        L = sp.eye(dim)
        for (i, j), v in L_off.items():
            L[i, j] = _as_rational(v)
        diag_v = [_as_rational(D_p[k][0]) * (cvars[k % n_vars] + _as_rational(D_p[k][1]))
                  for k in range(dim)]
        U = sp.eye(dim)
        for (i, j), v in U_off.items():
            U[i, j] = _as_rational(v)
        return L * sp.diag(*diag_v) * U

    G     = make_G(coords)
    G_inv = G.inv()

    result = {}
    for axis in range(n_vars):
        shifted = list(coords)
        shifted[axis] = shifted[axis] + 1
        G_sh = make_G(shifted)
        D_i  = sp.diag(*[coords[axis] + k for k in range(dim)])
        X    = G_sh * D_i * G_inv
        X    = sp.simplify(X)
        rows = [[str(X[r, c]) for c in range(dim)] for r in range(dim)]
        result[f"X{axis}"] = rows

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Limit-Value Walk + Irrational Detection
# ══════════════════════════════════════════════════════════════════════════════

def get_limit_value(fn, dim: int, n_vars: int,
                    dps: int = ID_DPS, depth: int = ID_DEPTH) -> mp.mpf | None:
    """High-precision walk along axis 0 → return v[0]/v[-1]."""
    mp.mp.dps = dps + 10
    pos = [2] * n_vars
    v   = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)
    for _ in range(depth):
        pos[0] += 1
        try:
            raw = np.asarray(fn(*pos), dtype=float)
            M   = mp.matrix([[mp.mpf(str(raw[r][c])) for c in range(dim)]
                              for r in range(dim)])
            v = M * v
        except Exception:
            return None
        scale = max(abs(v[i]) for i in range(dim))
        if scale > mp.power(10, 40):
            v /= scale
        elif scale < mp.power(10, -40):
            return None
    if abs(v[dim - 1]) < mp.power(10, -(dps - 5)):
        return None
    return v[0] / v[dim - 1]


# ══════════════════════════════════════════════════════════════════════════════
# Gate 5 — Cross-Ray Consistency
# ══════════════════════════════════════════════════════════════════════════════

GATE5_REL_TOL = 1e-3   # rays must agree to 0.1% relative

def _gate5_cross_ray(fns: list, dim: int, n_vars: int,
                    primary_val: mp.mpf,
                    dps: int = ID_DPS,
                    depth: int = 600) -> tuple[bool, list[float]]:
    """
    Walk along axes 1, 2 (if available) and verify limits agree with axis 0.
    Returns (consistent: bool, all_limits: list[float]).
    A spread > GATE5_REL_TOL * |primary| means FATAL_INCONSISTENT_RAYS.
    """
    mp.mp.dps = dps + 5
    primary_f = float(primary_val)
    limits = [primary_f]

    for ax in range(1, min(n_vars, 3)):
        pos = [2] * n_vars
        v   = mp.zeros(dim, 1)
        v[0] = mp.mpf(1)
        fn  = fns[ax % len(fns)]
        ok  = True
        for _ in range(depth):
            pos[ax % n_vars] += 1
            try:
                raw = np.asarray(fn(*pos), dtype=float)
                M   = mp.matrix([[mp.mpf(str(raw[r][c]))
                                   for c in range(dim)] for r in range(dim)])
                v = M * v
            except Exception:
                ok = False; break
            scale = max(abs(v[i]) for i in range(dim))
            if scale > mp.power(10, 40):
                v /= scale
            elif scale < mp.power(10, -40):
                ok = False; break
        if ok and abs(v[dim - 1]) >= mp.power(10, -(dps - 5)):
            limits.append(float(v[0] / v[dim - 1]))

    if len(limits) < 2:
        return True, limits   # single-axis CMF — can't falsify

    spread = max(abs(a - b) for a in limits for b in limits)
    ref    = abs(primary_f) + 1e-30
    return (spread / ref) < GATE5_REL_TOL, limits


# Gate label sets
_GATE_IRRATIONAL_LABELS = {"IRRATIONAL_UNKNOWN", "TRUE_TRANSCENDENTAL"}
_GATE_FATAL_LABELS = {
    "FATAL_ZERO_TRAP", "FATAL_DIVERGENCE_TRAP", "FATAL_NEAR_ONE",
    "FATAL_NEAR_MINUS_ONE", "FATAL_TRIVIAL_RATIONAL",
    "FATAL_ALGEBRAIC_ESCAPE", "FATAL_PSLQ_OVERFIT",
}


def identify_limit(val: mp.mpf) -> dict:
    """
    4-gate strict irrationality classifier wrapping reward_engine.classify_limit_strict.
    Returns dict with keys expected by write_enriched_record.

    Gate summary:
      G1  Zero/Divergence/±1 trap      → FATAL (not counted as irrational)
      G2  Algebraic purge (prime roots) → FATAL
      G3  PSLQ coefficient cap > 50     → FATAL
      G4  True transcendental (PSLQ ok) → MASSIVE BONUS (counted as irrational)
      G3  Pass all, no ID               → IRRATIONAL_UNKNOWN (counted as irrational)
    """
    mp.mp.dps = ID_DPS
    value_str = mp.nstr(val, 20)
    fval      = float(val)

    gate = classify_limit_strict(fval, hp_val=val, verbose=True)
    label      = gate["label"]
    is_irr     = label in _GATE_IRRATIONAL_LABELS
    is_fatal   = label in _GATE_FATAL_LABELS

    # Map gate label to irrational_type string
    if label == "TRUE_TRANSCENDENTAL":
        irr_type = "true_transcendental"
        ident    = gate.get("identify_str") or gate.get("reason", "")
    elif label == "IRRATIONAL_UNKNOWN":
        irr_type = "unknown_irrational"
        ident    = gate.get("identify_str")
    elif label == "FATAL_ALGEBRAIC_ESCAPE":
        irr_type = "algebraic_escape_rejected"
        ident    = gate.get("identify_str")
    elif label == "FATAL_PSLQ_OVERFIT":
        irr_type = "pslq_overfit_rejected"
        ident    = None
    else:
        irr_type = None
        ident    = None

    # Full verbose label for display
    reason = gate.get("reason", label)
    if label == "TRUE_TRANSCENDENTAL" and gate.get("pslq_relation"):
        from reward_engine import _GATE3_BASIS_NAMES
        names   = ["L"] + _GATE3_BASIS_NAMES
        rel_str = " + ".join(
            f"({c})*{n}" for c, n in zip(gate["pslq_relation"], names) if c != 0
        ) + " = 0"
        display_label = f"TRUE_TRANSCENDENTAL ({rel_str})"
    else:
        display_label = f"{label} | {reason[:80]}"

    return {
        "value_str":        value_str,
        "identified":       ident,
        "is_rational":      is_fatal and label in {
            "FATAL_TRIVIAL_RATIONAL", "FATAL_NEAR_ONE", "FATAL_NEAR_MINUS_ONE"},
        "looks_irrational": is_irr,
        "irrational_type":  irr_type,
        "label":            display_label,
        "gate_result":      gate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Record Writer (extends run_all_agents.write_record with limit + matrices)
# ══════════════════════════════════════════════════════════════════════════════

def write_enriched_record(agent: str, dim: int, params: dict,
                          deltas: list, fp: str, pi_err: float,
                          br: float, limit_info: dict,
                          sym_matrices: dict) -> None:
    rec = {
        "dim":                        dim,
        "n_matrices":                 len(deltas),
        "matrix_size":                dim,
        "agent":                      agent,
        "fingerprint":                fp,
        "best_delta":                 round(max(deltas) if deltas else 0.0, 3),
        "deltas":                     [round(d, 3) for d in deltas],
        "n_converging_axes":          sum(1 for d in deltas if d > 1.0),
        "path_independence_verified": True,
        "flatness_verified":          True,
        "max_flatness_error":         round(pi_err, 12),
        "n_pairs_checked":            params["n_vars"] * (params["n_vars"] - 1) // 2,
        "coupling_bucket":            4 if br >= 0.50 else 3 if br >= 0.10 else 2,
        "bidir_ratio":                round(br, 4),
        "params":                     _serialise_params(params),
        "timestamp":                  time.time(),
        "scout_batch":                "v2_anti_hack",
        # Irrational identification
        "limit_value":                limit_info.get("value_str"),
        "limit_identified":           limit_info.get("identified"),
        "limit_label":                limit_info.get("label"),
        "looks_irrational":           limit_info.get("looks_irrational", False),
        "irrational_type":            limit_info.get("irrational_type"),
        "gate_result":                limit_info.get("gate_result"),
    }
    # Add symbolic matrices if available
    rec.update(sym_matrices)

    sp_path = store_path(agent, dim)
    with open(sp_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _serialise_params(p: dict) -> dict:
    return {
        "dim":      p["dim"],
        "n_vars":   p["n_vars"],
        "D_params": p["D_params"],
        "L_off":    {str(k): v for k, v in p["L_off"].items()},
        "U_off":    {str(k): v for k, v in p["U_off"].items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-Agent Scout
# ══════════════════════════════════════════════════════════════════════════════

def scout_agent(agent: str, target: int) -> list[dict]:
    """Find `target` new CMFs for this agent. Returns list of result dicts."""
    cfg    = AGENT_CONFIGS[agent]
    dims   = cfg["dims"]
    n_vars = cfg["n_vars"]
    rng    = np.random.default_rng(int(time.time() * 1000) % (2**31))
    seen   = load_seen(agent)

    found_irrational = 0
    found_total  = 0
    trials  = 0
    results = []

    print(f"\n{'━'*60}", flush=True)
    print(f"  Agent {agent}  |  dims={dims}  n_vars={n_vars}  "
          f"already_seen={len(seen)}  target={target} irrational", flush=True)
    print(f"{'━'*60}", flush=True)
    t0 = time.time()

    while found_irrational < target:
        if SENTINEL.exists():
            print(f"  [{agent}] STOP_AGENTS detected — exiting.", flush=True)
            break

        trials += 1
        dim = int(rng.choice(dims))

        try:
            params = sample_params(dim, cfg, rng)
            fns    = build_eval_fns(params)
        except Exception:
            continue

        # T0 asymptotic filter
        ok, _ = asymptotic_filter(fns, dim, n_vars)
        if not ok:
            continue

        # T1 pole check
        if not t1_pole_check(fns, dim, n_vars, rng):
            continue

        # T2 fast numpy convergence
        t2_ok, _ = t2_fast(fns, dim, n_vars)
        if not t2_ok:
            continue

        # T3 high-precision convergence
        t3_ok, deltas = t3_thorough(fns, dim, n_vars)
        if not t3_ok:
            continue

        # T4 path independence
        pi_ok, pi_err = t4_flatness(fns, dim, n_vars)
        if not pi_ok:
            continue

        # Deduplicate
        fp = fingerprint(fns, dim, n_vars)
        if fp in seen:
            continue
        seen.add(fp)

        br = bidir_ratio(fns, dim, n_vars)
        found_total += 1

        # ── Get limit value + irrational check ────────────────────────────
        print(f"\n  [{agent}] #{found_total} scanned | {found_irrational}/{target} irrational  "
              f"{dim}×{dim}  Δ={max(deltas):.1f}  fp={fp}  "
              f"(trials={trials:,}  t={time.time()-t0:.0f}s)", flush=True)
        print(f"      Identifying limit value …", flush=True)

        limit_val  = get_limit_value(fns[0], dim, n_vars)
        if limit_val is not None:
            limit_info = identify_limit(limit_val)
        else:
            limit_info = {"value_str": None, "identified": None,
                          "label": "WALK_FAILED", "looks_irrational": False,
                          "irrational_type": None}

        label  = limit_info["label"]
        is_irr = limit_info.get("looks_irrational", False)

        # ── Gate 5: cross-ray consistency ─────────────────────────────────
        if is_irr and limit_val is not None:
            g5_ok, ray_limits = _gate5_cross_ray(fns, dim, n_vars, limit_val)
            if not g5_ok:
                spread = max(abs(a - b) for a in ray_limits for b in ray_limits)
                print(f"      [Gate5] REJECTED: inconsistent rays  "
                      f"spread={spread:.2e}  limits={[round(x,6) for x in ray_limits]}",
                      flush=True)
                limit_info["label"]          = "FATAL_INCONSISTENT_RAYS"
                limit_info["looks_irrational"] = False
                limit_info["gate5_ray_limits"] = ray_limits
                is_irr = False
                label  = "FATAL_INCONSISTENT_RAYS"
            else:
                print(f"      [Gate5] PASS  ({len(ray_limits)} rays consistent, "
                      f"limits={[round(x,6) for x in ray_limits]})", flush=True)
                limit_info["gate5_ray_limits"] = ray_limits

        if is_irr:
            found_irrational += 1
        marker = "🔥" if is_irr else "·"
        print(f"      {marker} Limit: {limit_info.get('value_str', 'N/A')[:40]}", flush=True)
        print(f"      {marker} Classification: {label}  [{found_irrational}/{target} irrational]", flush=True)

        # ── Build symbolic matrices ────────────────────────────────────────
        if dim <= 5:
            print(f"      Building symbolic matrices …", flush=True)
            sym_t0 = time.time()
            try:
                sym_matrices = build_symbolic_matrices(params)
                print(f"      ✓ Symbolic matrices ready ({time.time()-sym_t0:.1f}s)",
                      flush=True)
            except Exception as e:
                print(f"      ✗ Symbolic matrices failed: {e}", flush=True)
                sym_matrices = {}
        else:
            sym_matrices = {}
            print(f"      (dim={dim} > 5 — skipping symbolic matrices)", flush=True)

        # ── Store record ───────────────────────────────────────────────────
        write_enriched_record(agent, dim, params, deltas, fp,
                              pi_err, br, limit_info, sym_matrices)

        results.append({
            "agent": agent, "dim": dim, "fp": fp,
            "best_delta": max(deltas),
            "limit_label": label,
            "looks_irrational": limit_info.get("looks_irrational", False),
            "identified": limit_info.get("identified"),
        })

    elapsed = time.time() - t0
    print(f"\n  [{agent}] Done: {found_irrational}/{target} irrational "
          f"({found_total} total converging) in {trials:,} trials ({elapsed:.0f}s)", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=list("ABCDEFGHIJ"),
                    help="Which agents to run (default: all A–J)")
    ap.add_argument("--target", type=int, default=10,
                    help="New CMFs to find per agent (default: 10)")
    ap.add_argument("--no-ingest", action="store_true",
                    help="Skip DB ingest + Railway push at the end")
    args = ap.parse_args()

    print(f"\n{'═'*60}", flush=True)
    print(f"  Irrational Scout — agents {args.agents}  target={args.target}", flush=True)
    print(f"{'═'*60}", flush=True)

    all_results: list[dict] = []

    for agent in args.agents:
        if agent not in AGENT_CONFIGS:
            print(f"  [!] Unknown agent '{agent}' — skipping.", flush=True)
            continue
        res = scout_agent(agent, args.target)
        all_results.extend(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}", flush=True)
    print(f"  SUMMARY — {len(all_results)} new CMFs discovered", flush=True)
    print(f"{'═'*60}", flush=True)
    irrational_hits = [r for r in all_results if r["looks_irrational"]]
    known_hits      = [r for r in all_results if r.get("identified")]
    print(f"  Total found:      {len(all_results)}", flush=True)
    print(f"  Irrational limits:{len(irrational_hits)}", flush=True)
    print(f"  Known constants:  {len(known_hits)}", flush=True)
    print(flush=True)
    for r in all_results:
        marker = "🔥" if r["looks_irrational"] else "·"
        print(f"  {marker} [{r['agent']}] {r['dim']}×{r['dim']}  fp={r['fp']}  "
              f"Δ={r['best_delta']:.1f}  {r['limit_label']}", flush=True)

    # ── DB Ingest + Railway Push ──────────────────────────────────────────────
    if not args.no_ingest and all_results:
        print(f"\n{'─'*60}", flush=True)
        print("  Ingesting new CMFs into atlas_2d.db …", flush=True)
        try:
            subprocess.run([PYTHON, str(INGEST), "--db", str(DB)],
                           cwd=str(ROOT), timeout=180, check=True)
            print("  ✓ Ingest done.", flush=True)
        except Exception as e:
            print(f"  ✗ Ingest error: {e}", flush=True)

        print("  Pushing to Railway …", flush=True)
        try:
            subprocess.run(["railway", "up", "--detach"],
                           cwd=str(ROOT), timeout=60, check=False)
            print("  ✓ Railway push queued.", flush=True)
        except Exception as e:
            print(f"  ✗ Railway push error: {e}", flush=True)

    print(f"\n{'═'*60}", flush=True)
    print("  Scout complete.", flush=True)
    print(f"{'═'*60}\n", flush=True)


if __name__ == "__main__":
    main()
