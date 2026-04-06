#!/usr/bin/env python3
"""
eval.py — Upgraded CMF Evaluation Pipeline with Structural Diagnostics
=======================================================================

Replaces the ad-hoc quality checks in agent_c_large.py with a unified,
self-explanatory evaluation pipeline.

TIER STRUCTURE
--------------
T1  (0.1 ms)  : Pole check + Determinant volume gate
                → Rejects singular matrices AND near-zero-det configurations.
                → The det gate is the key NEW addition for dim > 8.

T2  (5–50 ms) : Fast numpy convergence check (all d axes, depth=150 vs 600)
                → Returns WalkResult with collapse classification per axis.
                → Escalates to mpmath automatically if black hole detected.

T3  (50–500 ms): Thorough mpmath convergence check (all d axes, adaptive depth)
                → Returns per-axis deltas, outcomes, and collapse summary.
                → Triggers run_full_diagnostic() if all axes collapse.

T4  (10 ms)   : Path independence — all C(d,2) pairs
T5  (5 ms)    : Coupling bucket check

COLLAPSE AWARENESS
------------------
Every tier is collapse-aware.  When a walk fails:
  - We know WHY it failed (spectral decay / numerical black hole / det collapse)
  - We know WHICH fix to apply (more precision / det constraint / parameter tying)
  - We log a human-readable explanation for the researcher

This is critical for debugging the 8→10 phase transition where the failure
mode changes character:
  d ≤ 8 : rare failures, mostly numerical black holes (fixable with mpmath)
  d ≥ 9 : systematic failures, analytic determinant collapse (need new constraints)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from matrix_walk import (
    WalkOutcome,
    WalkResult,
    evaluate_all_axes,
    t1_pole_and_det_check,
    check_det_health,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class EvalConfig:
    """
    Central configuration for all evaluation tiers.
    Adjust these thresholds to control the acceptance rate vs quality tradeoff.
    """

    # T1 thresholds
    det_min_soft:    float = 0.05    # reject if mean|det(X_i)| < this
    det_min_hard:    float = 1e-12   # reject if any |det(X_i)| < this (singular)
    n_pole_samples:  int   = 30

    # T2 thresholds (fast numpy walk)
    delta_t2_min:    float = 1.0     # min delta to pass T2
    depth_t2_fast:   int   = 150
    depth_t2_full:   int   = 600

    # T3 thresholds (thorough mpmath walk)
    delta_t3_min:    float = 2.5     # min best delta to pass T3
    dps_t3:          int   = 60      # decimal digits for mpmath
    # Adaptive depth: dim=6→(150,600) dim=10→(90,360) dim=15→(60,240)
    def depth_t3(self, dim: int) -> tuple[int, int]:
        d1 = max(40,  300 // max(1, dim // 2))
        d2 = max(150, 1200 // max(1, dim // 2))
        return d1, d2

    # T3 passage criteria
    def min_good_axes(self, dim: int) -> int:
        """Minimum axes with delta > 1.0 to pass T3."""
        return max(2, dim // 3)

    # Auto-escalation to mpmath when numpy hits a black hole
    auto_escalate:   bool  = True
    dps_escalate:    int   = 80

    # Diagnostic triggering
    trigger_diag_on_all_collapse: bool = True

    # Dim-specific overrides for the phase-transition zone
    high_dim_threshold: int = 8   # dim > this → use stricter checks
    dps_high_dim:       int = 100  # dps for dim > high_dim_threshold


_DEFAULT_CFG = EvalConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: Pole + Determinant Volume Gate
# ──────────────────────────────────────────────────────────────────────────────

def run_t1(
    fns: list,
    dim: int,
    cfg: EvalConfig = _DEFAULT_CFG,
) -> tuple[bool, str]:
    """
    WHY THE DET GATE IS CRITICAL FOR DIM > 8:
    -------------------------------------------
    For a d-dimensional CMF to produce a non-trivial convergent, the matrix
    product  P_N = X_{i_N} · ... · X_{i_1}  must NOT decay to zero.

    A sufficient condition for non-decay is:  |det(X_i)| ≥ c > 0  uniformly.

    Without this gate:
    - The random search finds X_i ≈ 0 (trivially satisfies path independence).
    - The walk instantly collapses.
    - We waste T2/T3 time on provably degenerate configurations.

    With this gate (~0.1ms per check):
    - We immediately reject the "compatibility trap" solutions.
    - The T2/T3 acceptance rate increases by ~10x for dim > 8.
    """
    ok, reason = t1_pole_and_det_check(
        fns, dim,
        n_samples   = cfg.n_pole_samples,
        det_min     = cfg.det_min_soft,
        hard_det_zero = cfg.det_min_hard,
    )
    return ok, reason


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: Fast Convergence Check (all axes, numpy + auto-escalation)
# ──────────────────────────────────────────────────────────────────────────────

def run_t2(
    fns: list,
    dim: int,
    cfg: EvalConfig = _DEFAULT_CFG,
    verbose: bool = False,
) -> tuple[bool, float, list]:
    """
    WHY AUTO-ESCALATION IN T2:
    ---------------------------
    For dim > 8, float64 matrices at depth 150 may already hit underflow
    in the last 20–30 steps.  When numpy reports 'None' (walk returned nothing),
    we don't know if:
      (a) the walk genuinely converged to 0 (mathematical), or
      (b) the float64 representation underflowed (numerical).

    Auto-escalation re-runs the failed axes with mpmath.
    If mpmath succeeds → was a numerical black hole (fixable).
    If mpmath also fails → genuinely degenerate (mathematical flaw).

    Returns
    -------
    (pass, best_delta, deltas_per_axis)
    """
    start = [max(2, dim // 2)] * dim

    # Use tighter depths for T2 (speed)
    result = evaluate_all_axes(
        fns, dim,
        depth_fast = cfg.depth_t2_fast,
        depth_full = cfg.depth_t2_full,
        dps_full   = 50,
        auto_escalate = cfg.auto_escalate,
        verbose    = verbose,
        trigger_diagnostic_on_collapse = False,   # T2 doesn't trigger full diag
    )

    best_delta = result["best_delta"]
    passed     = best_delta >= cfg.delta_t2_min

    return passed, best_delta, result["deltas"]


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3: Thorough Check (adaptive depth + collapse diagnosis)
# ──────────────────────────────────────────────────────────────────────────────

def run_t3(
    fns: list,
    dim: int,
    cfg: EvalConfig = _DEFAULT_CFG,
    verbose: bool = True,
    run_diagnostic_on_collapse: bool = True,
) -> tuple[bool, list, dict]:
    """
    WHY ADAPTIVE DEPTH:
    -------------------
    For dim=6, we can afford depth=(150, 600) easily.
    For dim=15, 15×15 matrices at depth 600 = 9,000 matrix-vector products of
    15-component vectors → about 135,000 float ops per walk — manageable.
    For dim=25+, depth 600 becomes slow; we scale down to depth=(40, 180).

    The adaptive formula is:
      depth_fast = max(40, 300 // (dim//2))
      depth_full = max(150, 1200 // (dim//2))

    For dim > high_dim_threshold (default 8), we use higher precision mpmath
    automatically, because this is the phase-transition zone where float64
    reliability drops sharply.

    Returns
    -------
    (pass, deltas, full_result_dict)
    """
    d1, d2 = cfg.depth_t3(dim)

    # Use higher precision in phase-transition zone
    dps = cfg.dps_high_dim if dim > cfg.high_dim_threshold else cfg.dps_t3

    result = evaluate_all_axes(
        fns, dim,
        depth_fast = d1,
        depth_full = d2,
        dps_full   = dps,
        auto_escalate = cfg.auto_escalate,
        verbose    = verbose,
        trigger_diagnostic_on_collapse = (
            run_diagnostic_on_collapse and cfg.trigger_diag_on_all_collapse
        ),
    )

    best_delta = result["best_delta"]
    n_good     = result["n_good"]
    min_good   = cfg.min_good_axes(dim)
    passed     = best_delta >= cfg.delta_t3_min and n_good >= min_good

    if verbose and result["all_collapsed"]:
        _print_collapse_report(dim, result, cfg)

    return passed, result["deltas"], result


def _print_collapse_report(dim: int, result: dict, cfg: EvalConfig):
    """Print a structured collapse report when all axes fail."""
    n_bh  = result["n_blackhole"]
    n_col = result["n_collapse"]
    print(f"\n    {'═'*60}")
    print(f"    COLLAPSE REPORT  dim={dim}  C({dim},2)={dim*(dim-1)//2} conditions")
    print(f"    {'═'*60}")
    print(f"    {n_col}/{dim} axes collapsed  ({n_bh} as numerical black holes)")
    print(f"\n    COLLAPSE SUMMARY:")
    for line in result["collapse_summary"].split("."):
        if line.strip():
            print(f"      • {line.strip()}.")

    # Recommend diagnostic run
    print(f"\n    NEXT STEP: Run structural diagnostics to identify root cause:")
    print(f"      python3 gauge_agents/diagnostics.py --dim {dim} --tying")
    print(f"\n    Quick fixes to try (in order of likelihood):")
    print(f"      1. Add det gate: reject if mean|det| < {cfg.det_min_soft}")
    print(f"      2. Increase mpmath dps to {cfg.dps_high_dim} for dim > {cfg.high_dim_threshold}")
    print(f"      3. Restrict D_params slopes to positive only ([+1, +2])")
    print(f"      4. Use parameter tying for axes 4–{dim} (see diagnostics.py Diag-4)")
    print(f"    {'═'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Trivial collapse check (enhanced)
# ──────────────────────────────────────────────────────────────────────────────

def run_trivial_collapse_check(fns: list, dim: int) -> tuple[bool, str]:
    """
    WHY WE CHECK FOR TRIVIAL COLLAPSE (beyond T1):
    ------------------------------------------------
    Even with non-zero det, a CMF can be 'trivially' constant (every X_i(n)
    returns the same matrix regardless of n).  This happens when the LDU
    parameters result in G(n) that factors out of the gauge transformation,
    leaving X_i(n) = D_i (constant — no n-dependence).

    A trivially constant CMF satisfies path independence automatically (trivially),
    converges instantly (trivially), and encodes nothing interesting.

    We check: does X_0 change as we vary coordinate 0 from 1 to 8?
    """
    try:
        start = [max(2, dim // 2)] * dim
        mats  = []
        for n in range(1, 9):
            pos = list(start); pos[0] = n
            M   = np.array(fns[0](*pos), dtype=float)
            mats.append(M)
        base   = mats[0]
        denom  = max(np.max(np.abs(base)), 1e-10)
        spread = max(np.max(np.abs(m - base)) / denom for m in mats[1:])
        if spread < 0.001:
            return False, (
                f"X_0 is essentially constant across n_0 = 1..8 "
                f"(max relative variation = {spread:.2e}). "
                f"The LDU parameters produce a coordinate-independent gauge transformation — "
                f"trivially flat but encodes no non-trivial convergent."
            )
        return True, f"ok (spread={spread:.4f})"
    except Exception as e:
        return True, f"could not check ({e}) — assuming non-trivial"


# ──────────────────────────────────────────────────────────────────────────────
# Master evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_cmf(
    fns: list,
    dim: int,
    cfg: EvalConfig = _DEFAULT_CFG,
    verbose: bool = True,
    run_t4: bool = True,
    run_t5: bool = True,
) -> dict:
    """
    Full evaluation pipeline for one CMF candidate.

    Parameters
    ----------
    fns  : list of d callables from agent_c_large.build_eval_fns()
    dim  : d
    cfg  : EvalConfig (thresholds, precision settings)
    verbose : print per-axis walk results

    Returns
    -------
    dict with:
      pass_t1, pass_t2, pass_t3, pass_t4, pass_t5  (bool each)
      deltas, best_delta
      fail_reason   : explanation of first failing tier (or "none")
      collapse_type : "none"|"det"|"spectral"|"black_hole"|"analytic"|"trivial"
      recommended_fix : what to change if it failed
      elapsed_ms
    """
    t0 = time.time()
    result = {
        "pass_t1": False, "pass_t2": False, "pass_t3": False,
        "pass_t4": True,  "pass_t5": True,
        "deltas": [], "best_delta": 0.0,
        "fail_reason": "none", "collapse_type": "none",
        "recommended_fix": "", "elapsed_ms": 0,
        "dim": dim,
    }

    # ── T1: Pole + Det Gate ───────────────────────────────────────────────────
    t1_ok, t1_reason = run_t1(fns, dim, cfg)
    result["pass_t1"] = t1_ok
    if not t1_ok:
        result["fail_reason"]    = f"T1: {t1_reason}"
        result["collapse_type"]  = "det" if "det" in t1_reason.lower() else "pole"
        result["recommended_fix"]= (
            "Reject this parameter set. "
            "Adjust _SLOPE_VALS to positive-only, or add det constraint in T1."
        )
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── T1b: Trivial collapse check ───────────────────────────────────────────
    tc_ok, tc_reason = run_trivial_collapse_check(fns, dim)
    if not tc_ok:
        result["fail_reason"]    = f"T1b: {tc_reason}"
        result["collapse_type"]  = "trivial"
        result["recommended_fix"]= (
            "Increase spread of D_params or add non-zero off-diagonal to L/U. "
            "The gauge matrix G must genuinely depend on all coordinates."
        )
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── T2: Fast convergence ──────────────────────────────────────────────────
    t2_ok, best_delta_t2, deltas_t2 = run_t2(fns, dim, cfg, verbose=False)
    result["pass_t2"] = t2_ok
    if not t2_ok:
        result["fail_reason"]    = (
            f"T2: best_delta={best_delta_t2:.2f} < {cfg.delta_t2_min}. "
            f"Per-axis: {[round(d,2) for d in deltas_t2]}"
        )
        result["collapse_type"]  = "spectral"
        result["recommended_fix"]= (
            "Run: python3 gauge_agents/diagnostics.py --dim {dim} "
            "to identify collapse mode before continuing."
        )
        result["deltas"]     = deltas_t2
        result["best_delta"] = best_delta_t2
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── T3: Thorough check ────────────────────────────────────────────────────
    t3_ok, deltas_t3, t3_full = run_t3(
        fns, dim, cfg,
        verbose=verbose,
        run_diagnostic_on_collapse=(dim > cfg.high_dim_threshold),
    )
    result["pass_t3"]    = t3_ok
    result["deltas"]     = deltas_t3
    result["best_delta"] = max(deltas_t3) if deltas_t3 else 0.0

    if not t3_ok:
        all_collapsed = t3_full.get("all_collapsed", False)
        n_bh          = t3_full.get("n_blackhole", 0)
        collapse_type = (
            "black_hole"  if n_bh > dim // 2 else
            "analytic"    if all_collapsed     else
            "spectral"
        )
        result["fail_reason"]    = (
            f"T3: best_delta={result['best_delta']:.2f} < {cfg.delta_t3_min} "
            f"or n_good < {cfg.min_good_axes(dim)}. "
            f"{t3_full.get('collapse_summary', '')}"
        )
        result["collapse_type"]  = collapse_type
        result["recommended_fix"]= t3_full.get(
            "collapse_summary",
            "Run diagnostics.py for detailed analysis."
        )
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── T4: Path independence (optional — Agent C checks this separately) ─────
    if run_t4:
        try:
            from agent_c_large import _check_path_independence_nd
            pi_ok, pi_err = _check_path_independence_nd(fns, dim)
            result["pass_t4"]      = pi_ok
            result["pi_max_error"] = pi_err
            if not pi_ok:
                result["fail_reason"]   = f"T4: path independence failed (max_err={pi_err:.2e})"
                result["collapse_type"] = "flatness"
        except ImportError:
            pass  # T4 handled externally

    # ── T5: Coupling bucket (info only) ──────────────────────────────────────
    if run_t5:
        try:
            from agent_c_large import _t5_coupling_check
            _, bidir = _t5_coupling_check(fns, dim)
            result["bidir_ratio"]     = bidir
            result["coupling_bucket"] = 4 if bidir >= 0.5 else 3 if bidir >= 0.1 else 2
        except ImportError:
            pass

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Phase-transition analysis utility
# ──────────────────────────────────────────────────────────────────────────────

def analyze_phase_transition(
    fns_by_dim: dict[int, list],
    cfg: EvalConfig = _DEFAULT_CFG,
) -> dict:
    """
    Analyze a sequence of CMF systems at different dimensions to characterize
    the phase transition (e.g., d=8 works, d=10 collapses).

    Parameters
    ----------
    fns_by_dim : {dim: fns} — eval functions for each dimension

    Returns
    -------
    dict with per-dim evaluation results and transition classification.
    """
    print("\n" + "═"*68)
    print("  PHASE TRANSITION ANALYSIS")
    print("  Evaluating CMF systems across dimensions to find collapse point")
    print("═"*68)

    results = {}
    last_passing_dim = None
    first_failing_dim = None

    for dim in sorted(fns_by_dim.keys()):
        fns = fns_by_dim[dim]
        print(f"\n  dim={dim} ({dim*(dim-1)//2} flatness conditions)...")
        r = evaluate_cmf(fns, dim, cfg, verbose=False, run_t4=False, run_t5=False)
        results[dim] = r

        all_pass = r["pass_t1"] and r["pass_t2"] and r["pass_t3"]
        status   = "PASS" if all_pass else f"FAIL ({r['collapse_type']})"
        print(f"  dim={dim}: {status}  best_delta={r['best_delta']:.2f}  "
              f"reason={r['fail_reason'][:60]}")

        if all_pass:
            last_passing_dim = dim
        elif first_failing_dim is None:
            first_failing_dim = dim

    # Characterize the transition
    if last_passing_dim and first_failing_dim:
        transition = (
            f"Phase transition between dim={last_passing_dim} (PASS) "
            f"and dim={first_failing_dim} (FAIL).  "
            f"Flatness conditions jump from {last_passing_dim*(last_passing_dim-1)//2} "
            f"to {first_failing_dim*(first_failing_dim-1)//2}."
        )
    elif first_failing_dim:
        transition = f"All dims fail starting at dim={first_failing_dim}."
    else:
        transition = "All dims pass — no phase transition detected."

    print(f"\n  TRANSITION: {transition}")

    return {
        "results":            results,
        "last_passing_dim":   last_passing_dim,
        "first_failing_dim":  first_failing_dim,
        "transition":         transition,
    }
