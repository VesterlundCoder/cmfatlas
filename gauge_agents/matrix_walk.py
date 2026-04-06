#!/usr/bin/env python3
"""
matrix_walk.py — Enhanced CMF Matrix Walk with Collapse Diagnostics
=====================================================================

Drop-in replacement for the walk functions in agent_c_large.py, augmented with:
  - Per-step Frobenius norm logging (walk_norm_trajectory hook)
  - Spectral radius monitoring along the walk
  - Determinant health check before walking
  - Automatic collapse classification and explanation
  - Adaptive precision escalation (numpy → mpmath) on suspected numerical black hole

This module is the primary interface for all CMF matrix walks.
Import it instead of the inline walk functions in agent_c_large.py.

DESIGN PHILOSOPHY
-----------------
Every function that walks a matrix product must know *why* it stopped or failed.
We distinguish three regimes:

  CONVERGING  : walk ratio v[0]/v[-1] stabilises — non-trivial CMF
  DECAYING    : walk collapses to 0 — spectral radius < 1 or det collapsing
  BLACK_HOLE  : walk is stable then suddenly NaN/inf — floating-point exhaustion

The CollapseClassifier at the end of every walk returns one of these verdicts
plus a suggested fix, so the calling agent can make an informed decision.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import mpmath as mp
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

HERE = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────────
# Walk result types
# ──────────────────────────────────────────────────────────────────────────────

class WalkOutcome(str, Enum):
    CONVERGING   = "converging"
    DECAYING     = "decaying"
    BLACK_HOLE   = "numerical_blackhole"
    DIVERGING    = "diverging"
    UNKNOWN      = "unknown"


@dataclass
class WalkResult:
    """Complete result of a matrix walk along one lattice ray."""

    # Core output
    outcome:         WalkOutcome = WalkOutcome.UNKNOWN
    ratio:           Optional[float] = None     # v[0]/v[dim-1] at final step
    delta:           float = 0.0                # convergence quality (log10 scale)

    # Norm trajectory (step, log10_frobenius_norm)
    norm_trajectory: list = field(default_factory=list)
    decay_slope:     float = 0.0   # linear fit slope (negative = decay)

    # Spectral tracking (step, log10_spectral_radius)
    spectral_trajectory: list = field(default_factory=list)
    geo_mean_rho:    float = 1.0   # geometric mean of spectral radius during walk

    # Determinant health (measured before the walk)
    det_health:      str = "unchecked"   # "ok" | "decaying" | "zero" | "unchecked"
    mean_det:        float = 0.0

    # Walk metadata
    depth_reached:   int = 0
    collapse_step:   Optional[int] = None   # step where norm suddenly dropped
    nan_step:        Optional[int] = None
    elapsed_s:       float = 0.0
    used_mpmath:     bool = False

    # Human explanation (auto-filled by CollapseClassifier)
    explanation:     str = ""
    recommended_fix: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Pre-walk determinant health check
# ──────────────────────────────────────────────────────────────────────────────

def check_det_health(
    fns: list,
    dim: int,
    n_samples: int = 40,
    ray_length: int = 10,
    axis: int = 0,
) -> tuple[str, float]:
    """
    WHY WE CHECK THIS BEFORE WALKING:
    ----------------------------------
    det(X_i(n)) represents the volume scaling of the state space at each step.
    If det(X_i) is uniformly small or shrinking along the walk ray, the matrix
    product ‖P_N‖ will decay at rate |det|^N → 0, regardless of matrix size.

    This pre-flight check catches analytic collapse before wasting time on a
    full walk.  It takes ~5ms for a 10×10 system.

    Returns
    -------
    (health: str, mean_det: float)
    health: "ok" | "decaying" | "zero" | "unknown"
    """
    rng = np.random.default_rng(11)
    box = max(3, min(15, 30 // max(dim, 1)))
    pts = rng.integers(2, box + 1, size=(n_samples, dim)).tolist()

    dets = []
    for coords in pts:
        try:
            M = np.array(fns[axis](*coords), dtype=complex)
            d = abs(np.linalg.det(M))
            if np.isfinite(d):
                dets.append(d)
        except Exception:
            pass

    if not dets:
        return "unknown", 0.0

    mean_det = float(np.mean(dets))
    min_det  = float(np.min(dets))

    # Check det growth along a ray (n=2,4,...,2*ray_length)
    det_ray = []
    for k in range(1, ray_length + 1):
        pos = [2] * dim; pos[axis] = k * 2
        try:
            M = np.array(fns[axis](*pos), dtype=complex)
            d = abs(np.linalg.det(M))
            if np.isfinite(d) and d > 1e-300:
                det_ray.append(math.log(d))
        except Exception:
            pass

    if len(det_ray) >= 4:
        xs = np.arange(len(det_ray), dtype=float)
        slope = float(np.polyfit(xs, det_ray, 1)[0])
    else:
        slope = 0.0

    if min_det < 1e-10:
        return "zero", mean_det
    elif slope < -0.5 and mean_det < 0.5:
        return "decaying", mean_det
    else:
        return "ok", mean_det


# ──────────────────────────────────────────────────────────────────────────────
# Collapse classifier
# ──────────────────────────────────────────────────────────────────────────────

class CollapseClassifier:
    """
    Analyses a WalkResult and fills in the human-readable explanation
    and recommended_fix fields.

    WHY THIS EXISTS:
    When a walk collapses, the researcher needs to know:
      (a) IS this a math problem or a numerics problem?
      (b) WHAT specifically failed?
      (c) WHAT should be changed in the search parameters?

    This classifier answers all three questions in plain language.
    """

    # Thresholds
    DECAY_SLOPE_THRESH     = -0.005   # decades/step
    SUDDEN_DROP_DECADES    = 5.0      # log10 drop per log_every steps
    GEO_MEAN_THRESH        = 0.95

    @classmethod
    def classify(cls, result: WalkResult, dim: int) -> WalkResult:
        """Fill result.outcome, .explanation, .recommended_fix in place."""

        is_decay   = result.decay_slope < cls.DECAY_SLOPE_THRESH
        is_bh      = result.collapse_step is not None or result.nan_step is not None
        is_det_bad = result.det_health in ("decaying", "zero")
        is_rho_bad = result.geo_mean_rho < cls.GEO_MEAN_THRESH

        if result.ratio is not None and math.isfinite(result.ratio) and result.delta > 1.0:
            result.outcome = WalkOutcome.CONVERGING
            result.explanation = (
                f"Walk converged: ratio={result.ratio:.8g}, delta={result.delta:.2f}. "
                f"Non-trivial CMF — walk ratio is numerically stable."
            )
            result.recommended_fix = "None needed."
            return result

        if is_bh and not is_rho_bad and not is_det_bad:
            result.outcome = WalkOutcome.BLACK_HOLE
            step_info = (f"at step {result.collapse_step}"
                         if result.collapse_step else
                         f"NaN at step {result.nan_step}")
            result.explanation = (
                f"NUMERICAL BLACK HOLE ({step_info}): norm was stable, then suddenly "
                f"dropped {cls.SUDDEN_DROP_DECADES:.0f}+ decades.  "
                f"The underlying mathematics may be non-trivial — this is a floating-point "
                f"exhaustion problem.  For a {dim}×{dim} matrix, entries can span 100+ "
                f"orders of magnitude after 200+ steps; float64 loses all precision."
            )
            result.recommended_fix = (
                "Switch _walk_numpy() to _walk_mp() with mpmath dps=80 for dim > 8.  "
                "Also normalize v every step (not every 50) to prevent underflow.  "
                "This is fixable without changing the CMF parameters."
            )

        elif is_rho_bad and is_decay:
            result.outcome = WalkOutcome.DECAYING
            result.explanation = (
                f"SPECTRAL DECAY: geometric mean of ρ_max = {result.geo_mean_rho:.4f} < 1.0, "
                f"norm decay slope = {result.decay_slope:.5f} decades/step.  "
                f"The Lyapunov exponent is negative (γ₁ ≈ {math.log(result.geo_mean_rho + 1e-300):.4f}).  "
                f"Every matrix step shrinks the state vector — the product P_N → 0 exponentially.  "
                f"At depth 300: expected ‖P‖ ~ {result.geo_mean_rho**300:.2e}."
            )
            result.recommended_fix = (
                "Use only positive slopes in D_params (a_k > 0) to ensure det(D_i) = n_i*(n_i+1)*...*(n_i+d-1) > 1.  "
                "Start walk at n=d instead of n=2 (Pochhammer product grows with n).  "
                "Reject params where mean|det(X_i)| < 0.5 in T1 pole check."
            )

        elif is_det_bad and not is_rho_bad:
            result.outcome = WalkOutcome.DECAYING
            result.explanation = (
                f"DETERMINANT COLLAPSE: det health = '{result.det_health}', "
                f"mean|det| = {result.mean_det:.3e}.  "
                f"The matrices are compressing the state space to near-zero volume.  "
                f"This is the 'trivial solution' trap: the optimizer zeros out G to satisfy "
                f"the {dim*(dim-1)//2} path-independence conditions for dim={dim}."
            )
            result.recommended_fix = (
                "Add hard rejection: if mean|det(X_i)| < 0.1 in _t1_pole_check(), reject.  "
                "Or normalize each X_i to det=1 (SL(d) constraint) before the walk.  "
                "Or use _SLOPE_VALS = [+1, +2, +3] only (no negative slopes)."
            )

        elif is_decay and not is_bh:
            result.outcome = WalkOutcome.DECAYING
            result.explanation = (
                f"ANALYTIC SPECTRAL DECAY: walk norm decays at {result.decay_slope:.5f} dec/step "
                f"from the very first step.  Higher precision will NOT fix this — it is a "
                f"mathematical property of the chosen LDU parameters.  "
                f"The {dim}×{dim} system is over-constrained (C({dim},2)={dim*(dim-1)//2} conditions) "
                f"and the optimizer is finding near-zero matrices as the 'easy' solution."
            )
            result.recommended_fix = (
                "Try dimensionality tying (build_eval_fns_tied in diagnostics.py) to "
                "reduce search space.  Or use a warm start: begin walk at n=10 to avoid "
                "small-n regime where Pochhammer products are tiny."
            )

        else:
            result.outcome = WalkOutcome.UNKNOWN
            result.explanation = (
                f"Walk gave None/0 without clear signature.  "
                f"decay_slope={result.decay_slope:.5f}, geo_mean_rho={result.geo_mean_rho:.4f}, "
                f"det_health={result.det_health}, collapse_step={result.collapse_step}."
            )
            result.recommended_fix = "Run full diagnostics: python3 diagnostics.py --dim {dim}"

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Core walk functions
# ──────────────────────────────────────────────────────────────────────────────

def walk_numpy_diagnosed(
    fns: list,
    dim: int,
    axis: int,
    start: list,
    depth: int,
    log_every: int = 25,
    collect_spectral: bool = False,
) -> WalkResult:
    """
    WHY THIS REPLACES _walk_numpy():
    ---------------------------------
    The original _walk_numpy() returns only a single float (the final ratio).
    When the walk returns None (collapse), we have no idea *why* — the function
    discards all diagnostic information.

    This version retains:
      - per-step norm trajectory (to detect black holes vs smooth decay)
      - spectral radius sampling (to detect eigenvalue-driven collapse)
      - exact collapse step (to pinpoint NaN/underflow location)
      - collapse classification with human explanation

    It is ~5% slower than the bare walk but provides essential debugging info.
    """
    result = WalkResult()
    t0 = time.time()

    # Pre-flight det check
    det_health, mean_det = check_det_health(fns, dim, n_samples=20, axis=axis)
    result.det_health = det_health
    result.mean_det   = mean_det

    pos  = list(start)
    v    = np.zeros(dim, dtype=float)
    v[0] = 1.0

    norm_log_prev: Optional[float] = None
    log_rho: list = []

    for step in range(1, depth + 1):
        pos[axis] += 1
        try:
            M = np.array(fns[axis](*pos), dtype=float)
            if not np.all(np.isfinite(M)):
                result.nan_step = step
                break
            v = M @ v
        except Exception:
            result.nan_step = step
            break

        # Spectral radius (expensive — only every log_every steps if requested)
        if collect_spectral and step % log_every == 0:
            try:
                M_ev = np.array(fns[axis](*pos), dtype=complex)
                evals = np.linalg.eigvals(M_ev)
                rho = float(np.max(np.abs(evals)))
                if rho > 1e-300 and np.isfinite(rho):
                    log_rho.append(math.log(rho))
                    result.spectral_trajectory.append((step, math.log10(rho + 1e-300)))
            except Exception:
                pass

        # Norm trajectory
        norm = float(np.linalg.norm(v))
        if step % log_every == 0:
            if norm > 1e-300 and np.isfinite(norm):
                log_norm = math.log10(norm)
                result.norm_trajectory.append((step, log_norm))

                # Sudden drop detection (numerical black hole signature)
                if norm_log_prev is not None:
                    drop = norm_log_prev - log_norm
                    if drop > 5.0 and result.collapse_step is None:
                        result.collapse_step = step
                norm_log_prev = log_norm
            else:
                # Underflow
                result.norm_trajectory.append((step, -math.inf))
                if result.collapse_step is None:
                    result.collapse_step = step
                break

        # Normalization (prevent overflow; do NOT normalize too rarely)
        if step % 30 == 0:
            n = np.linalg.norm(v)
            if n > 1e50:
                v /= n
            elif n < 1e-50 and n > 0:
                # Underflow approaching — record and stop
                if result.collapse_step is None:
                    result.collapse_step = step
                break

    result.depth_reached = step

    # Compute decay slope from norm trajectory
    finite_norms = [(s, ln) for s, ln in result.norm_trajectory if np.isfinite(ln)]
    if len(finite_norms) >= 4:
        xs = np.array([s for s, _ in finite_norms], dtype=float)
        ys = np.array([ln for _, ln in finite_norms], dtype=float)
        result.decay_slope = float(np.polyfit(xs, ys, 1)[0])

    # Geometric mean of spectral radius
    if log_rho:
        result.geo_mean_rho = math.exp(float(np.mean(log_rho)))

    # Convergent readout
    if abs(v[dim - 1]) > 1e-20 and np.isfinite(v[dim - 1]):
        result.ratio = float(v[0] / v[dim - 1])

    result.elapsed_s = time.time() - t0
    return CollapseClassifier.classify(result, dim)


def walk_mpmath_diagnosed(
    fns: list,
    dim: int,
    axis: int,
    start: list,
    depth: int,
    dps: int = 80,
    log_every: int = 50,
) -> WalkResult:
    """
    High-precision mpmath walk with collapse diagnostics.

    WHY HIGH PRECISION FOR dim > 8:
    --------------------------------
    A 10×10 matrix product after 400 steps involves numbers spanning
    ~40 orders of magnitude in floating point.  float64 has ~15.6 decimal
    digits of precision.  When large entries and small entries co-exist in P_N,
    catastrophic cancellation destroys all information.

    mpmath with dps=80 gives 80 decimal digits — enough headroom for dim≤15
    at depth≤600 with per-step normalization.

    For dim > 15, use dps=150 or implement a Schur decomposition tracker.
    """
    if not HAS_MPMATH:
        raise ImportError("mpmath is required for walk_mpmath_diagnosed()")

    result = WalkResult()
    result.used_mpmath = True
    t0 = time.time()

    mp.mp.dps = dps + 10  # extra guard digits

    det_health, mean_det = check_det_health(fns, dim, n_samples=15, axis=axis)
    result.det_health = det_health
    result.mean_det   = mean_det

    pos  = list(start)
    v    = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)

    norm_log_prev: Optional[float] = None
    _ten_30 = mp.power(10, 30)
    _ten_neg30 = mp.power(10, -30)

    for step in range(1, depth + 1):
        pos[axis] += 1
        try:
            M_raw = fns[axis](*pos)
            M = mp.matrix([[mp.mpf(str(float(M_raw[r][c])))
                            for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception as exc:
            result.nan_step = step
            break

        # Check for NaN/inf in mpmath result
        try:
            scale = max(abs(v[i]) for i in range(dim))
        except Exception:
            result.nan_step = step
            break

        if step % log_every == 0:
            if scale > 0 and mp.isfinite(scale):
                try:
                    log_norm = float(mp.log10(scale + mp.mpf(10)**-300))
                    result.norm_trajectory.append((step, log_norm))
                    if norm_log_prev is not None:
                        drop = norm_log_prev - log_norm
                        if drop > 5.0 and result.collapse_step is None:
                            result.collapse_step = step
                    norm_log_prev = log_norm
                except Exception:
                    pass
            else:
                result.norm_trajectory.append((step, -math.inf))
                if result.collapse_step is None:
                    result.collapse_step = step
                break

        # Normalize every 20 steps (critical for large dim)
        if step % 20 == 0:
            if scale > _ten_30:
                v /= scale
            elif scale < _ten_neg30:
                if result.collapse_step is None:
                    result.collapse_step = step
                break

    result.depth_reached = step

    finite_norms = [(s, ln) for s, ln in result.norm_trajectory if np.isfinite(ln)]
    if len(finite_norms) >= 4:
        xs = np.array([s for s, _ in finite_norms], dtype=float)
        ys = np.array([ln for _, ln in finite_norms], dtype=float)
        result.decay_slope = float(np.polyfit(xs, ys, 1)[0])

    try:
        denom = v[dim - 1]
        if abs(denom) > mp.power(10, -(dps - 10)):
            result.ratio = float(mp.re(v[0] / denom))
    except Exception:
        pass

    result.elapsed_s = time.time() - t0
    return CollapseClassifier.classify(result, dim)


# ──────────────────────────────────────────────────────────────────────────────
# Delta estimator with collapse awareness
# ──────────────────────────────────────────────────────────────────────────────

def estimate_delta_diagnosed(
    fns: list,
    dim: int,
    axis: int,
    start: list,
    depth_fast: int = 150,
    depth_full: int = 600,
    auto_escalate: bool = True,
    dps_escalate: int = 80,
) -> tuple[float, WalkResult, WalkResult]:
    """
    Estimate convergence delta for one axis with full collapse diagnostics.

    WHY AUTO_ESCALATE:
    ------------------
    If the fast numpy walk ends in a numerical black hole, we automatically
    retry with mpmath.  If mpmath also collapses, the failure is mathematical.
    This two-stage approach correctly separates fixable numerical issues from
    fundamental mathematical ones — critical for the 8→10 phase transition.

    Returns
    -------
    (delta, result_fast, result_full)
    delta: convergence quality (log10 scale; 0 = no convergence, >2 = good)
    """
    result_fast = walk_numpy_diagnosed(fns, dim, axis, start, depth_fast,
                                       collect_spectral=False)
    result_full = walk_numpy_diagnosed(fns, dim, axis, start, depth_full,
                                       collect_spectral=True)

    # Auto-escalate to mpmath if either walk hit a numerical black hole
    if auto_escalate and HAS_MPMATH:
        if (result_fast.outcome == WalkOutcome.BLACK_HOLE or
                result_full.outcome == WalkOutcome.BLACK_HOLE):
            print(f"    [walk] Numerical black hole detected on axis {axis} — "
                  f"escalating to mpmath dps={dps_escalate} ...")
            result_fast = walk_mpmath_diagnosed(fns, dim, axis, start,
                                                depth_fast, dps=dps_escalate)
            result_full = walk_mpmath_diagnosed(fns, dim, axis, start,
                                                depth_full, dps=dps_escalate)
            result_fast.used_mpmath = True
            result_full.used_mpmath = True

    r1 = result_fast.ratio
    r2 = result_full.ratio

    if r1 is None or r2 is None:
        return 0.0, result_fast, result_full

    diff = abs(r2 - r1)
    if diff < 1e-50:
        delta = 50.0
    else:
        delta = min(50.0, -math.log10(diff + 1e-55))

    return delta, result_fast, result_full


# ──────────────────────────────────────────────────────────────────────────────
# Full multi-axis walk evaluation (replaces _t3_thorough_check)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all_axes(
    fns: list,
    dim: int,
    depth_fast: int = 150,
    depth_full: int = 600,
    dps_full: int   = 50,
    auto_escalate: bool = True,
    verbose: bool = True,
    trigger_diagnostic_on_collapse: bool = True,
) -> dict:
    """
    Run walked convergence check on all d axes.  Replaces _t3_thorough_check()
    with full collapse awareness.

    WHY CHECK ALL AXES:
    -------------------
    A d-dimensional CMF has d lattice directions.  For a non-trivial CMF to
    encode a mathematical constant, the matrix product must converge along
    AT LEAST one axis (the 'interesting' direction).  All axes collapsing
    simultaneously is the hallmark of a trivially degenerate system.

    Parameters
    ----------
    trigger_diagnostic_on_collapse : if True, call run_full_diagnostic() when
        ALL axes collapse (the 8→10 phase transition scenario)

    Returns
    -------
    dict with 'pass', 'deltas', 'best_delta', 'outcomes', 'collapse_summary'
    """
    start = [max(2, dim // 2)] * dim  # start at n = dim//2 (avoid small-n regime)
    # Adaptive depth: large dim → shorter individual walks
    d1 = max(40,  300 // max(1, dim // 2))
    d2 = max(150, 1200 // max(1, dim // 2))

    deltas   = []
    outcomes = []
    all_results_fast = []
    all_results_full = []

    n_collapse = 0
    n_blackhole = 0

    for ax in range(dim):
        delta, rf, rfl = estimate_delta_diagnosed(
            fns, dim, ax, start,
            depth_fast=d1, depth_full=d2,
            auto_escalate=auto_escalate,
            dps_escalate=dps_full,
        )
        deltas.append(delta)
        outcomes.append(rfl.outcome.value)
        all_results_fast.append(rf)
        all_results_full.append(rfl)

        if rfl.outcome in (WalkOutcome.DECAYING, WalkOutcome.BLACK_HOLE, WalkOutcome.UNKNOWN):
            n_collapse += 1
        if rfl.outcome == WalkOutcome.BLACK_HOLE:
            n_blackhole += 1

        if verbose:
            symbol = "✓" if delta > 2.0 else "~" if delta > 0.5 else "✗"
            bh_flag = " [BH]" if rfl.outcome == WalkOutcome.BLACK_HOLE else ""
            print(f"    axis {ax:>2}: δ={delta:5.2f}  {symbol}  "
                  f"{rfl.outcome.value:18s}{bh_flag}  "
                  f"det={rfl.det_health:8s}  "
                  f"slope={rfl.decay_slope:+.4f}",
                  flush=True)

    best_delta = max(deltas) if deltas else 0.0
    n_good     = sum(1 for d in deltas if d > 1.0)
    min_good   = max(2, dim // 3)
    passed     = (best_delta >= 2.5 and n_good >= min_good)

    # Build collapse summary
    all_collapsed = (n_collapse == dim)
    if all_collapsed:
        if n_blackhole > dim // 2:
            collapse_summary = (
                f"ALL {dim} AXES COLLAPSED ({n_blackhole} as numerical black holes). "
                f"Likely cause: float64 underflow for {dim}×{dim} matrices at depth {d2}. "
                f"Try mpmath dps=80 or normalize every step."
            )
        else:
            collapse_summary = (
                f"ALL {dim} AXES COLLAPSED ({n_collapse - n_blackhole} as analytic decay). "
                f"The optimizer found a trivially degenerate solution to the "
                f"C({dim},2)={dim*(dim-1)//2} path-independence conditions. "
                f"This is the Compatibility Trap — try parameter tying or det constraint."
            )
    elif n_collapse > dim // 2:
        collapse_summary = (
            f"{n_collapse}/{dim} axes collapsed.  Partial failure — "
            f"the system is marginally non-trivial but likely not a genuine CMF."
        )
    else:
        collapse_summary = (
            f"{n_good}/{dim} axes converging (δ>1.0).  "
            f"System appears non-trivial."
        )

    if verbose and all_collapsed and trigger_diagnostic_on_collapse:
        print(f"\n    {'!'*60}")
        print(f"    PHASE TRANSITION DETECTED: ALL {dim} AXES COLLAPSED")
        print(f"    {collapse_summary}")
        print(f"    Run full diagnostics: python3 gauge_agents/diagnostics.py --dim {dim}")
        print(f"    {'!'*60}\n")

    return {
        "pass":            passed,
        "deltas":          deltas,
        "best_delta":      best_delta,
        "n_good":          n_good,
        "n_collapse":      n_collapse,
        "n_blackhole":     n_blackhole,
        "all_collapsed":   all_collapsed,
        "outcomes":        outcomes,
        "collapse_summary": collapse_summary,
        "fast_results":    all_results_fast,
        "full_results":    all_results_full,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Enhanced T1 pole + det check (replaces _t1_pole_check)
# ──────────────────────────────────────────────────────────────────────────────

def t1_pole_and_det_check(
    fns: list,
    dim: int,
    n_samples: int = 30,
    det_min: float = 0.05,
    hard_det_zero: float = 1e-12,
) -> tuple[bool, str]:
    """
    WHY WE ADD A DET CONSTRAINT TO T1:
    ------------------------------------
    The original T1 only checks if det(X_i) ≈ 0 (singular matrix).
    But for the phase transition at d=9-10, the collapse is GRADUAL:
    det(X_i) is not zero, but is uniformly small (0.001 to 0.01).
    Over 300 walk steps, 0.01^300 = 10^{-600} — total annihilation.

    By rejecting configurations where mean|det(X_i)| < det_min (default 0.05),
    we prevent the optimizer from finding the 'trivial collapse' solution.

    Parameters
    ----------
    det_min      : soft threshold — reject if mean|det| < this
    hard_det_zero: hard threshold — reject if any |det| < this (singular)

    Returns
    -------
    (pass: bool, reason: str)
    """
    rng = np.random.default_rng(0)
    box = max(3, min(15, 30 // max(dim, 1)))
    pts = rng.integers(1, box + 1, size=(n_samples, dim)).tolist()

    all_dets = []
    for coords in pts:
        for ax in range(min(3, dim)):   # check first 3 axes (speed)
            try:
                M = np.array(fns[ax](*coords), dtype=complex)
                if not np.all(np.isfinite(M)):
                    return False, f"Non-finite entry in X_{ax} at {coords[:3]}"
                d = abs(np.linalg.det(M))
                if d < hard_det_zero:
                    return False, (
                        f"|det(X_{ax})| = {d:.2e} < {hard_det_zero} at {coords[:3]}. "
                        f"Matrix is singular — G(n) has a zero diagonal in LDU "
                        f"(a_k * (n_k + b_k) = 0 at this lattice point)."
                    )
                all_dets.append(d)
            except (ZeroDivisionError, ValueError, OverflowError) as e:
                return False, f"Exception in X_{ax}: {e}"

    if not all_dets:
        return False, "No valid determinant samples"

    mean_det = float(np.mean(all_dets))
    if mean_det < det_min:
        return False, (
            f"mean|det(X_i)| = {mean_det:.4f} < {det_min}. "
            f"WHY THIS MATTERS: Over 300 walk steps, det(P_300) ~ {mean_det**300:.2e}. "
            f"The walk will collapse to 0 — this is the Compatibility Trap. "
            f"The optimizer found near-zero matrices to trivially satisfy the "
            f"C({dim},2)={dim*(dim-1)//2} path-independence conditions."
        )

    return True, f"ok (mean|det|={mean_det:.3f})"
