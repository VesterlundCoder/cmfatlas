#!/usr/bin/env python3
"""
diagnostics.py — Phase-Transition Structural Diagnostics for CMF Matrix Walks
==============================================================================

MOTIVATION
----------
When scaling d-dimensional CMF systems (d matrices of size d×d) beyond d=8,
the matrix walk catastrophically collapses to zero.  This module implements
four targeted diagnostics that pinpoint *exactly why* a given set of CMF
matrices fails:

  1. det_volume_check        — Is the state-space volume being compressed to 0?
  2. spectral_gap_analysis   — Is the dominant eigenvalue < 1 (guaranteed decay)?
  3. walk_norm_trajectory    — Smooth decay vs sudden numerical black-hole?
  4. dimensionality_tying    — Fallback: tie matrices 4–d to cyclic shifts of 1–3

USAGE
-----
  from diagnostics import run_full_diagnostic, build_eval_fns_tied

  # On a collapse candidate (fns = list of eval callables from agent_c_large.py):
  report = run_full_diagnostic(fns, dim, label="10x10_candidate_001")

  # Tied-parameter fallback mode:
  fns_tied = build_eval_fns_tied(params_3var, dim_target=10)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# mpmath is used for high-precision norm tracking and eigenvalue computation
try:
    import mpmath as mp
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

HERE = Path(__file__).parent

# ── ANSI colour helpers (graceful fallback on dumb terminals) ─────────────────
_RED    = "\033[91m" if sys.stdout.isatty() else ""
_YELLOW = "\033[93m" if sys.stdout.isatty() else ""
_GREEN  = "\033[92m" if sys.stdout.isatty() else ""
_CYAN   = "\033[96m" if sys.stdout.isatty() else ""
_BOLD   = "\033[1m"  if sys.stdout.isatty() else ""
_RESET  = "\033[0m"  if sys.stdout.isatty() else ""

# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1: Determinant Volume Preservation
# ──────────────────────────────────────────────────────────────────────────────

def det_volume_check(
    fns: list,
    dim: int,
    n_samples: int = 60,
    det_zero_thresh: float = 1e-6,
    det_penalty_thresh: float = 0.01,
) -> dict:
    """
    Track the determinant of each CMF matrix X_i(n) across the lattice.

    WHY WE MEASURE THIS
    -------------------
    det(X_i) is the volume-scaling factor when X_i acts on the state space.
    If det(X_i) → 0, the matrix is compressing the full d-dimensional state
    space into a lower-dimensional subspace.

    In a long matrix walk  P = X_{i_N} · ... · X_{i_1},
        det(P) = ∏ det(X_{i_k})

    If each factor has |det| < 1 (even slightly), the product determinant
    decays *exponentially* to 0.  For d=10 with 10 matrices each having
    |det| ≈ 0.9, after 300 steps: det(P) ≈ 0.9^300 ≈ 10^{-14} — total collapse.

    A non-trivial CMF convergent (like those encoding ζ(3), π², etc.) requires
    that the matrix walk remains in a regime where the state vector is NOT
    annihilated.  This means we want: |det(X_i)| ≈ 1 or grows polynomially.

    WHAT A COLLAPSE LOOKS LIKE
    --------------------------
    - Collapse mode A (Analytic):  all |det(X_i)| < 1 at every lattice point.
      The system is mathematically doomed — no parameter tuning can fix it.
    - Collapse mode B (Structural): |det(X_i)| → 0 only at specific lattice
      points (poles / zeros of G). Need better parameter selection.
    - Healthy system: |det(X_i)| grows polynomially in n_i (like Pochhammer
      products), so the walk can stay non-trivial after normalization.

    Parameters
    ----------
    fns              : list of d callables  fn_i(*coords) → d×d ndarray
    dim              : d (number of matrices = matrix size)
    n_samples        : number of random lattice points to sample
    det_zero_thresh  : if min |det| < this → HARD COLLAPSE (mode B)
    det_penalty_thresh: if all |det| < this uniformly → ANALYTIC COLLAPSE (mode A)

    Returns
    -------
    dict with keys:
      collapse_mode  : "none" | "analytic" | "structural" | "mixed"
      min_det        : minimum |det(X_i)| across all samples
      mean_det       : mean |det(X_i)|
      det_by_axis    : list of (mean, min) per axis
      det_log_growth : estimated log-slope of |det| vs n (positive = growing)
      penalty_score  : float in [0,1], where 1 = totally collapsed, 0 = healthy
      explanation    : human-readable string explaining the failure mode
    """
    rng = np.random.default_rng(42)
    # Use positive lattice points (n >= 2 to avoid coordinate singularities)
    box = max(4, min(20, 40 // max(dim, 1)))
    pts = rng.integers(2, box + 1, size=(n_samples, dim)).tolist()

    print(f"\n{_BOLD}{_CYAN}[DIAG-1] Determinant Volume Preservation Check  "
          f"(dim={dim}){_RESET}")
    print(f"         Sampling |det(X_i)| at {n_samples} lattice points × {dim} axes")
    print(f"         WHY: det(P_N) = ∏ det(X_k).  If |det(X_k)| < 1 uniformly,")
    print(f"         the walk product collapses exponentially to 0.")

    det_by_axis = []  # list of lists of |det| values
    for ax in range(dim):
        dets = []
        for coords in pts:
            try:
                M = np.array(fns[ax](*coords), dtype=complex)
                d = abs(np.linalg.det(M))
                if np.isfinite(d):
                    dets.append(d)
            except Exception:
                pass
        det_by_axis.append(dets)

    # Also measure det growth along a simple ray (how does det scale with n?)
    det_growth_slopes = []
    for ax in range(min(3, dim)):
        base = [2] * dim
        axis_vals = list(range(2, 22, 2))  # n_ax = 2,4,...,20
        det_traj = []
        for n_ax in axis_vals:
            pos = list(base); pos[ax] = n_ax
            try:
                M = np.array(fns[ax](*pos), dtype=complex)
                d = abs(np.linalg.det(M))
                if np.isfinite(d) and d > 1e-300:
                    det_traj.append(math.log(d + 1e-300))
            except Exception:
                pass
        if len(det_traj) >= 4:
            # Linear regression: log|det| ~ slope * n + intercept
            xs = np.array(axis_vals[:len(det_traj)], dtype=float)
            ys = np.array(det_traj, dtype=float)
            slope = np.polyfit(xs, ys, 1)[0]
            det_growth_slopes.append(slope)

    # Aggregate statistics
    all_dets = [d for ax_dets in det_by_axis for d in ax_dets]
    if not all_dets:
        return {
            "collapse_mode": "unknown",
            "min_det": 0.0, "mean_det": 0.0,
            "det_by_axis": [],
            "det_log_growth": 0.0,
            "penalty_score": 1.0,
            "explanation": "Could not evaluate matrices (numerical failure at all sample points).",
        }

    min_det  = min(all_dets)
    mean_det = float(np.mean(all_dets))
    det_log_growth = float(np.mean(det_growth_slopes)) if det_growth_slopes else 0.0
    per_axis = [(float(np.mean(d)), float(min(d))) if d else (0.0, 0.0)
                for d in det_by_axis]

    # Classify collapse mode
    frac_near_zero = sum(1 for d in all_dets if d < det_penalty_thresh) / len(all_dets)
    has_zeros      = min_det < det_zero_thresh

    if frac_near_zero > 0.80:
        collapse_mode = "analytic"
        explanation = (
            f"ANALYTIC COLLAPSE: {frac_near_zero*100:.0f}% of sampled det(X_i) < {det_penalty_thresh}. "
            f"Mean |det| = {mean_det:.3e}.  The optimizer has found matrices that satisfy path-independence "
            f"by collapsing to near-zero rank — the 'easy' solution to the over-constrained system.  "
            f"Fix: add a hard det ≥ 1 constraint or SL(d) normalization."
        )
    elif has_zeros:
        collapse_mode = "structural"
        explanation = (
            f"STRUCTURAL COLLAPSE: min |det(X_i)| = {min_det:.3e} (below hard threshold {det_zero_thresh}). "
            f"The matrices are singular at some lattice points.  "
            f"Fix: check G(n) for zero diagonals in the LDU decomposition — a_k*(n_k + b_k) = 0 at "
            f"integer lattice points causes det(G) = 0 and thus det(X_i) = det(G_shifted)/det(G) → ∞."
        )
    elif det_log_growth < -0.5:
        collapse_mode = "decaying"
        explanation = (
            f"DECAYING DETERMINANT: det(X_i) grows as e^({det_log_growth:.3f}·n) along the ray. "
            f"Negative growth rate means the walk product shrinks with depth.  "
            f"Fix: use D_i = diag(n_i, n_i+1, ...) which gives det(D_i) = n_i·(n_i+1)···(n_i+d-1), "
            f"a polynomial in n_i — positive log-growth guaranteed."
        )
    else:
        collapse_mode = "none"
        explanation = (
            f"OK: det(X_i) is non-degenerate. Mean |det| = {mean_det:.3f}, "
            f"min = {min_det:.3e}, log-growth = {det_log_growth:.3f}/step. "
            f"Volume preservation is not the primary failure mode here."
        )

    penalty_score = min(1.0, frac_near_zero + float(has_zeros) * 0.5)

    # Print summary
    colour = _RED if collapse_mode != "none" else _GREEN
    print(f"\n         {colour}Collapse mode: {collapse_mode.upper()}{_RESET}")
    print(f"         min|det|={min_det:.3e}  mean|det|={mean_det:.3f}  "
          f"log_growth/step={det_log_growth:.3f}  penalty={penalty_score:.2f}")
    print(f"         ↳ {explanation}")

    for ax, (m, mn) in enumerate(per_axis[:min(6, dim)]):
        bar_len = int(min(m, 5.0) / 5.0 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"           axis {ax}: mean={m:.3f}  min={mn:.3e}  [{bar}]")
    if dim > 6:
        print(f"           ... ({dim - 6} more axes)")

    return {
        "collapse_mode": collapse_mode,
        "min_det": min_det,
        "mean_det": mean_det,
        "det_by_axis": per_axis,
        "det_log_growth": det_log_growth,
        "penalty_score": penalty_score,
        "explanation": explanation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2: Spectral Gap / Maximum Eigenvalue Decay
# ──────────────────────────────────────────────────────────────────────────────

def spectral_gap_analysis(
    fns: list,
    dim: int,
    walk_depth: int = 200,
    n_point_samples: int = 30,
    geometric_mean_thresh: float = 1.0,
) -> dict:
    """
    Compute eigenvalues of X_i(n) across the lattice and track the spectral
    radius (max |λ|) along a matrix walk.

    WHY WE MEASURE THIS
    -------------------
    A matrix walk  P_N = X_{i_N} · ... · X_{i_1}  is governed by the
    LYAPUNOV EXPONENT of the sequence {X_{i_k}}.

    By Oseledec's theorem, the asymptotic growth rate is:
        lim_{N→∞} (1/N) log ‖P_N‖ = γ₁  (top Lyapunov exponent)

    If γ₁ < 0, then ‖P_N‖ → 0 EXPONENTIALLY — guaranteed collapse.
    If γ₁ = 0, the walk stays bounded — the interesting case for CMFs.
    If γ₁ > 0, the walk grows — also fine (we normalize during the walk).

    The SPECTRAL RADIUS ρ(X_i) = max |λ| of a single matrix is the
    single-step contribution to γ₁.  The geometric mean of ρ along the
    walk approximates γ₁.

    For the d=10 collapse:
    - If the d matrices are close to zero for large n, ρ → 0.
    - If the matrices have a mix of large/small eigenvalues depending on axis,
      the walk will favour the low-ρ direction → collapse.

    THE SPECTRAL GAP
    ----------------
    The ratio ρ₁/ρ₂ (largest/second-largest |eigenvalue|) determines how
    fast the walk "focuses" onto the dominant eigenvector.  A large gap is
    GOOD (fast convergence to limit).  A gap of 1 (degenerate spectrum) means
    the walk oscillates without converging.

    Parameters
    ----------
    fns             : list of d callables fn_i(*coords) → d×d ndarray
    dim             : d
    walk_depth      : depth for geometric-mean estimation
    n_point_samples : number of random lattice points to sample eigenvalues
    geometric_mean_thresh : if geometric mean of ρ_max < this → warn

    Returns
    -------
    dict with:
      geometric_mean_rho : geometric mean of spectral radius along walk
      spectral_gap       : mean (ρ₁-ρ₂)/ρ₁ ratio
      mean_rho_by_axis   : mean spectral radius per axis
      lyapunov_estimate  : log(geometric_mean_rho) ≈ top Lyapunov exponent
      collapse_risk      : "low" | "medium" | "high" | "critical"
      explanation        : human-readable string
    """
    rng = np.random.default_rng(77)
    box = max(4, min(20, 40 // max(dim, 1)))
    pts = rng.integers(2, box + 1, size=(n_point_samples, dim)).tolist()

    print(f"\n{_BOLD}{_CYAN}[DIAG-2] Spectral Gap / Eigenvalue Decay Analysis  "
          f"(dim={dim}){_RESET}")
    print(f"         WHY: If geometric-mean(ρ_max) < 1.0 along the walk, the")
    print(f"         Lyapunov exponent γ₁ < 0 → ‖P_N‖ → 0 EXPONENTIALLY.")
    print(f"         For CMF convergents, we need γ₁ ≈ 0 (boundary of stability).")

    rho_by_axis = [[] for _ in range(dim)]
    gap_by_axis = [[] for _ in range(dim)]

    for coords in pts:
        for ax in range(dim):
            try:
                M = np.array(fns[ax](*coords), dtype=complex)
                if not np.all(np.isfinite(M)):
                    continue
                evals = np.linalg.eigvals(M)
                abs_evals = sorted(np.abs(evals), reverse=True)
                rho1 = abs_evals[0]
                rho2 = abs_evals[1] if len(abs_evals) > 1 else 0.0
                rho_by_axis[ax].append(rho1)
                if rho1 > 1e-12:
                    gap_by_axis[ax].append((rho1 - rho2) / rho1)
            except Exception:
                pass

    # Geometric mean of ρ_max along an actual walk ray
    rho_walk = []
    start = [2] * dim
    pos   = list(start)
    for step in range(1, walk_depth + 1):
        ax = step % dim
        pos[ax] += 1
        try:
            M = np.array(fns[ax](*pos), dtype=complex)
            if not np.all(np.isfinite(M)):
                break
            evals = np.linalg.eigvals(M)
            rho   = float(np.max(np.abs(evals)))
            if rho > 1e-300:
                rho_walk.append(math.log(rho))
        except Exception:
            break

    # Aggregate
    mean_rho_by_axis = []
    for ax in range(dim):
        if rho_by_axis[ax]:
            mean_rho_by_axis.append(float(np.mean(rho_by_axis[ax])))
        else:
            mean_rho_by_axis.append(0.0)

    all_gaps = [g for ax_gaps in gap_by_axis for g in ax_gaps]
    spectral_gap = float(np.mean(all_gaps)) if all_gaps else 0.0

    if rho_walk:
        # geometric mean = exp(mean of log(rho))
        geo_mean_rho = math.exp(np.mean(rho_walk))
        lyapunov_est = float(np.mean(rho_walk))   # per step
    else:
        geo_mean_rho = 0.0
        lyapunov_est = -math.inf

    # Classify collapse risk
    if geo_mean_rho < 1e-3 or lyapunov_est < -5:
        collapse_risk = "critical"
        explanation = (
            f"CRITICAL: geometric mean of ρ_max = {geo_mean_rho:.3e} along the walk. "
            f"Lyapunov estimate γ₁ ≈ {lyapunov_est:.3f} per step.  "
            f"The dominant eigenvalue decays to 0 — every matrix step shrinks the state vector. "
            f"This is a pure spectral collapse: the {dim}×{dim} matrices have eigenvalues all < 1 "
            f"and the product ‖P_N‖ → 0 at rate e^({lyapunov_est:.2f}·N)."
        )
    elif geo_mean_rho < geometric_mean_thresh * 0.95:
        collapse_risk = "high"
        explanation = (
            f"HIGH RISK: geometric mean of ρ_max = {geo_mean_rho:.4f} < threshold {geometric_mean_thresh}. "
            f"Lyapunov estimate γ₁ ≈ {lyapunov_est:.4f}.  "
            f"The walk will decay, but slowly.  At depth 300, expected norm ≈ {geo_mean_rho**300:.2e}. "
            f"Consider forcing det(X_i) = polynomial in n_i to lift eigenvalues above 1."
        )
    elif geo_mean_rho < geometric_mean_thresh * 1.05:
        collapse_risk = "medium"
        explanation = (
            f"MEDIUM: geometric mean of ρ_max ≈ {geo_mean_rho:.4f} ≈ 1.0. "
            f"The walk is near the stability boundary — convergence possible but sensitive."
        )
    else:
        collapse_risk = "low"
        explanation = (
            f"OK: geometric mean of ρ_max = {geo_mean_rho:.4f} > 1.0. "
            f"The walk grows (we normalize) — consistent with a non-trivial convergent. "
            f"Spectral gap = {spectral_gap:.3f} (larger = faster convergence to limit)."
        )

    colour = _RED if collapse_risk in ("critical","high") else \
             _YELLOW if collapse_risk == "medium" else _GREEN
    print(f"\n         {colour}Collapse risk: {collapse_risk.upper()}{_RESET}")
    print(f"         geo_mean(ρ_max)={geo_mean_rho:.6f}  "
          f"Lyapunov_est={lyapunov_est:.4f}/step  "
          f"spectral_gap={spectral_gap:.3f}")
    print(f"         At depth 300: expected norm ~ {geo_mean_rho**300:.2e}")
    print(f"         ↳ {explanation}")

    print(f"\n         Spectral radius by axis (mean ρ_max):")
    for ax, rho in enumerate(mean_rho_by_axis[:min(8, dim)]):
        bar_len = int(min(rho, 5.0) / 5.0 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        flag = f" {_RED}← DECAYING{_RESET}" if rho < 1.0 else ""
        print(f"           axis {ax}: ρ_max_mean = {rho:.4f}  [{bar}]{flag}")
    if dim > 8:
        print(f"           ... ({dim - 8} more axes)")

    return {
        "geometric_mean_rho":  geo_mean_rho,
        "lyapunov_estimate":   lyapunov_est,
        "spectral_gap":        spectral_gap,
        "mean_rho_by_axis":    mean_rho_by_axis,
        "collapse_risk":       collapse_risk,
        "explanation":         explanation,
        "rho_walk_samples":    len(rho_walk),
    }


# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 3: Walk Norm Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def walk_norm_trajectory(
    fns: list,
    dim: int,
    walk_depth: int = 400,
    log_every: int  = 20,
    sudden_drop_ratio: float = 1e-6,
) -> dict:
    """
    Record the Frobenius norm of the cumulative product matrix P_N at every
    `log_every` steps of the walk.

    WHY WE MEASURE THIS
    -------------------
    We are trying to distinguish between two very different failure modes:

    MODE A — Analytic Spectral Decay
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The Frobenius norm ‖P_N‖_F decreases SMOOTHLY and EXPONENTIALLY from the
    very first step.  This means the mathematical eigenvalues of the matrices
    are genuinely less than 1.  No amount of higher floating-point precision
    will fix this — it's a fundamental mathematical problem.
    Signature: log(‖P_N‖_F) vs N is a straight line with negative slope.

    MODE B — Numerical Black Hole
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The norm stays roughly constant for many steps, then SUDDENLY drops to 0
    or becomes NaN.  This means:
      - Floating-point underflow: entries become smaller than 2^{-1074} ≈ 5e-324
      - Division by zero in normalization
      - Catastrophic cancellation in large (10×10) matrix products
    Signature: log(‖P_N‖_F) flat → sudden vertical drop.
    FIX: switch to mpmath with dps=100, or normalize more frequently.

    MODE C — Healthy Walk
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The norm grows polynomially or stays near constant (due to normalization).
    The per-step convergent ratio v[0]/v[dim-1] stabilizes.
    Signature: log(‖P_N‖_F) grows or stays flat.

    Parameters
    ----------
    fns              : list of d callables
    dim              : d
    walk_depth       : total walk depth
    log_every        : record norm every this many steps
    sudden_drop_ratio: ratio of consecutive norms triggering "black hole" flag

    Returns
    -------
    dict with:
      failure_mode   : "none" | "analytic_decay" | "numerical_blackhole" | "both"
      norm_trajectory : list of (step, log10_norm) pairs
      decay_slope    : linear regression slope of log10(norm) vs step
      drop_step      : step at which sudden drop occurred (None if mode A)
      final_norm     : ‖P_N‖_F at last recorded step (or 0 if collapsed)
      convergent_trajectory : list of (step, ratio) pairs
      explanation    : human-readable string
    """
    print(f"\n{_BOLD}{_CYAN}[DIAG-3] Walk Norm Trajectory  (dim={dim}, depth={walk_depth}){_RESET}")
    print(f"         WHY: Distinguish Analytic Decay (eigenvalues < 1, math flaw)")
    print(f"         from Numerical Black Hole (float underflow/cancellation, fixable).")
    print(f"         If norm drops SUDDENLY at one step → Numerical Black Hole.")
    print(f"         If norm decays SMOOTHLY from step 1 → Analytic Decay.")

    # Walk along the 'mixed' ray (1,1,1,...,1) — exercises all axes equally
    pos  = [2] * dim
    P    = np.eye(dim, dtype=float)   # cumulative product
    v    = np.zeros(dim, dtype=float)
    v[0] = 1.0                        # state vector (cheaper than full matrix product)

    norm_traj      = []   # (step, log10_norm)
    convergent_traj = []  # (step, ratio v[0]/v[-1])
    norms          = []
    drop_step      = None
    prev_log_norm  = None
    nan_step       = None

    t_start = time.time()

    for step in range(1, walk_depth + 1):
        ax = (step - 1) % dim
        pos[ax] += 1

        try:
            M   = np.array(fns[ax](*pos), dtype=float)
            if not np.all(np.isfinite(M)):
                nan_step = step
                print(f"         {_RED}NaN/Inf in matrix at step {step} (axis {ax}, "
                      f"coords {pos[:4]}...){_RESET}")
                break
            v = M @ v
        except Exception as exc:
            nan_step = step
            print(f"         {_RED}Exception at step {step}: {exc}{_RESET}")
            break

        # Frobenius norm of state vector (proxy for matrix product norm)
        norm = float(np.linalg.norm(v))

        if step % log_every == 0:
            if norm > 1e-300 and np.isfinite(norm):
                log_norm = math.log10(norm)
                norms.append(log_norm)
                norm_traj.append((step, log_norm))

                # Convergent readout
                if abs(v[dim - 1]) > 1e-30:
                    ratio = float(v[0] / v[dim - 1])
                    convergent_traj.append((step, ratio))
                else:
                    convergent_traj.append((step, None))

                # Detect sudden drop (numerical black hole signature)
                if prev_log_norm is not None:
                    drop = prev_log_norm - log_norm
                    if drop > -math.log10(sudden_drop_ratio):
                        if drop_step is None:
                            drop_step = step
                            print(f"         {_RED}SUDDEN NORM DROP at step {step}: "
                                  f"log10(norm) {prev_log_norm:.2f} → {log_norm:.2f} "
                                  f"(drop = {drop:.1f} decades){_RESET}")
                prev_log_norm = log_norm
            else:
                # Underflow / zero
                norm_traj.append((step, -math.inf))
                if drop_step is None and prev_log_norm is not None and prev_log_norm > -10:
                    drop_step = step
                    print(f"         {_RED}NORM UNDERFLOW at step {step} "
                          f"(prev log10(norm)={prev_log_norm:.2f} → -∞){_RESET}")
                break

        # Normalize to prevent overflow (every 50 steps)
        if step % 50 == 0 and norm > 1e50:
            v /= norm

    elapsed = time.time() - t_start

    # Linear regression on norm trajectory to estimate decay slope
    if len(norms) >= 4:
        xs = np.array([n[0] for n in norm_traj if np.isfinite(n[1])], dtype=float)
        ys = np.array([n[1] for n in norm_traj if np.isfinite(n[1])], dtype=float)
        if len(xs) >= 2:
            decay_slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            decay_slope = 0.0
    else:
        decay_slope = 0.0

    final_norm = 10**norms[-1] if norms else 0.0

    # Classify failure mode
    is_analytic  = decay_slope < -0.005  # more than 0.005 decades/step decay
    is_blackhole = drop_step is not None or nan_step is not None

    if is_analytic and is_blackhole:
        failure_mode = "both"
        explanation  = (
            f"BOTH MODES DETECTED. Analytic decay (slope={decay_slope:.4f} dec/step) "
            f"AND a numerical black hole at step {drop_step or nan_step}. "
            f"The math AND the numerics are both broken for dim={dim}. "
            f"Immediate action: (1) switch to mpmath dps=100, (2) add det constraint."
        )
    elif is_blackhole:
        failure_mode = "numerical_blackhole"
        explanation  = (
            f"NUMERICAL BLACK HOLE at step {drop_step or nan_step}. "
            f"The norm was stable until suddenly collapsing. "
            f"This is fixable: switch to mpmath with higher precision, "
            f"or normalize every step instead of every 50. "
            f"The underlying math may be non-trivial — the numerics are the bottleneck."
        )
    elif is_analytic:
        failure_mode = "analytic_decay"
        explanation  = (
            f"ANALYTIC SPECTRAL DECAY: norm decays at {decay_slope:.4f} decades/step. "
            f"At depth {walk_depth}: norm ≈ {10**(decay_slope*walk_depth):.2e}. "
            f"This is a mathematical flaw — the system's eigenvalues are < 1 everywhere. "
            f"Higher precision will NOT fix this. Need different parameter initialization "
            f"or stronger det ≥ 1 constraint."
        )
    else:
        failure_mode = "none"
        explanation  = (
            f"OK: norm is stable (decay slope = {decay_slope:.4f} dec/step). "
            f"Final log10(‖v‖) = {math.log10(final_norm + 1e-300):.2f}. "
            f"No collapse detected in the norm trajectory."
        )

    colour = _RED if failure_mode not in ("none",) else _GREEN
    print(f"\n         {colour}Failure mode: {failure_mode.upper()}{_RESET}")
    print(f"         Decay slope = {decay_slope:.5f} decades/step  "
          f"(negative = shrinking)  elapsed={elapsed:.1f}s")
    print(f"         ↳ {explanation}")

    # Print mini trajectory
    print(f"\n         Norm trajectory (step → log10‖v‖):")
    for step, ln in norm_traj[::max(1, len(norm_traj)//8)]:
        if np.isfinite(ln):
            bar_len = max(0, min(30, int((ln + 10) / 20 * 30)))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"           step {step:>4}: {ln:+7.2f}  [{bar}]")
        else:
            print(f"           step {step:>4}: -∞  {_RED}[COLLAPSED]{_RESET}")

    return {
        "failure_mode":         failure_mode,
        "decay_slope":          decay_slope,
        "drop_step":            drop_step,
        "nan_step":             nan_step,
        "final_norm":           final_norm,
        "norm_trajectory":      norm_traj,
        "convergent_trajectory":convergent_traj,
        "explanation":          explanation,
        "elapsed_s":            elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 4: Dimensionality Parameter Tying
# ──────────────────────────────────────────────────────────────────────────────

def build_eval_fns_tied(
    base_params: dict,
    dim_target: int,
    base_dim: int = 3,
    tie_mode: str = "cyclic_shift",
) -> list:
    """
    Build a d-dimensional CMF system where matrices 0..base_dim-1 are free,
    and matrices base_dim..dim_target-1 are tied to the first base_dim matrices
    via simple structural rules.

    WHY WE DO THIS
    --------------
    A d-dimensional CMF requires satisfying C(d,2) = d*(d-1)/2 path-independence
    (flatness) conditions:
      d=3  →   3 conditions   (manageable)
      d=6  →  15 conditions   (hard)
      d=8  →  28 conditions   (very hard)
      d=10 →  45 conditions   (nearly impossible with random search)

    By TYING the higher-dimensional matrices to the first 3, we reduce the
    effective search space to just the 3 base matrices.  The tying rule ensures
    the flatness conditions are either automatically satisfied or much easier
    to satisfy.

    TYING MODES
    -----------
    "cyclic_shift":
      X_k(n) ≈ X_{k % base_dim}(n shifted by k // base_dim in all coordinates)
      E.g., X_4 = X_1 with all coordinates shifted by +1.
      Geometric intuition: the higher axes are 'copies' of the base axes
      in a different region of the lattice, naturally compatible by translation
      symmetry of the gauge construction.

    "symmetric_permutation":
      X_k is a block-permuted version of X_{k % base_dim}.
      The block permutation is a cyclic shift of rows/columns.
      Ensures the spectral radius is preserved across all matrices.

    "identity_extension":
      X_k = Identity for k >= base_dim.  Trivially flat but trivially useless.
      Useful as a baseline: if even this collapses, the base 3 matrices are broken.

    Parameters
    ----------
    base_params : dict with 'dim', 'D_params', 'L_off', 'U_off'
                  (from agent_c_large.sample_params — base_dim-parameter format)
    dim_target  : target dimension d (e.g., 10 for 10×10)
    base_dim    : number of free base matrices (default 3)
    tie_mode    : "cyclic_shift" | "symmetric_permutation" | "identity_extension"

    Returns
    -------
    list of dim_target callables fn_k(*coords) → dim_target×dim_target ndarray.
    NOTE: The base matrices are padded to dim_target × dim_target with the
    identity block to preserve invertibility.

    IMPORTANT: coords must have length dim_target.
    """
    # Import here to avoid circular dependency
    sys.path.insert(0, str(HERE))
    from agent_c_large import _build_G, _build_Di

    base_dim_stored = base_params["dim"]

    print(f"\n{_BOLD}{_CYAN}[DIAG-4] Dimensionality Parameter Tying  "
          f"(base={base_dim_stored}→target={dim_target}, mode={tie_mode}){_RESET}")
    print(f"         WHY: d={dim_target} requires C({dim_target},2)={dim_target*(dim_target-1)//2} flatness conditions.")
    print(f"         By tying matrices {base_dim}..{dim_target-1} to matrices 0..{base_dim-1},")
    print(f"         we reduce to C({base_dim},2)={base_dim*(base_dim-1)//2} free conditions.")
    print(f"         Mode '{tie_mode}': {_get_tie_mode_description(tie_mode)}")

    eval_fns = []

    for axis in range(dim_target):
        if axis < base_dim_stored:
            # Free base matrices — direct evaluation
            def make_free_fn(ax=axis):
                def fn(*coords):
                    coords_base = list(coords[:base_dim_stored])
                    G_n  = _build_G(base_params, coords_base)
                    G_sh = _build_G(base_params,
                                    [c + (1 if k == ax else 0)
                                     for k, c in enumerate(coords_base)])
                    Di   = _build_Di(base_dim_stored, coords_base[ax])
                    det  = np.linalg.det(G_n)
                    if abs(det) < 1e-10:
                        raise ValueError(f"singular G at {coords_base}")
                    M_base = G_sh @ Di @ np.linalg.inv(G_n)
                    # Pad to dim_target × dim_target
                    M_full = np.eye(dim_target, dtype=float)
                    M_full[:base_dim_stored, :base_dim_stored] = M_base
                    return M_full
                return fn
            eval_fns.append(make_free_fn())
        else:
            # Tied matrices
            base_ax = axis % base_dim_stored
            tier    = axis // base_dim_stored  # 0-indexed 'copy number'

            if tie_mode == "cyclic_shift":
                # X_axis ≈ X_{base_ax} with coords shifted by 'tier' units
                def make_tied_fn(bax=base_ax, t=tier):
                    def fn(*coords):
                        # Shift all coordinates by t (translate on the lattice)
                        shifted = [c + t for c in coords[:base_dim_stored]]
                        G_n  = _build_G(base_params, shifted)
                        G_sh = _build_G(base_params,
                                        [c + (1 if k == bax else 0)
                                         for k, c in enumerate(shifted)])
                        Di   = _build_Di(base_dim_stored, shifted[bax])
                        det  = np.linalg.det(G_n)
                        if abs(det) < 1e-10:
                            raise ValueError("singular G (tied)")
                        M_base = G_sh @ Di @ np.linalg.inv(G_n)
                        # Permuted embedding: cycle rows/cols by t*base_dim
                        offset = t * base_dim_stored
                        M_full = np.eye(dim_target, dtype=float)
                        r0 = offset % dim_target
                        for i in range(base_dim_stored):
                            for j in range(base_dim_stored):
                                ri = (r0 + i) % dim_target
                                rj = (r0 + j) % dim_target
                                M_full[ri, rj] = M_base[i, j]
                        return M_full
                    return fn
                eval_fns.append(make_tied_fn())

            elif tie_mode == "symmetric_permutation":
                # Row/column permutation of base matrix: M_tied = P^T M_base P
                perm = list(range(dim_target))
                shift_amt = axis * (dim_target // max(dim_target - 1, 1))
                perm = [(i + shift_amt) % dim_target for i in range(dim_target)]
                def make_sym_fn(bax=base_ax, perm_=perm):
                    def fn(*coords):
                        coords_b = list(coords[:base_dim_stored])
                        G_n  = _build_G(base_params, coords_b)
                        G_sh = _build_G(base_params,
                                        [c + (1 if k == bax else 0)
                                         for k, c in enumerate(coords_b)])
                        Di   = _build_Di(base_dim_stored, coords_b[bax])
                        det  = np.linalg.det(G_n)
                        if abs(det) < 1e-10:
                            raise ValueError("singular G (sym)")
                        M_base = G_sh @ Di @ np.linalg.inv(G_n)
                        M_full = np.eye(dim_target, dtype=float)
                        M_full[:base_dim_stored, :base_dim_stored] = M_base
                        # Apply row/col permutation
                        P_mat = np.eye(dim_target)[perm_]
                        return P_mat.T @ M_full @ P_mat
                    return fn
                eval_fns.append(make_sym_fn())

            else:
                # identity_extension — trivial baseline
                def make_id_fn():
                    def fn(*coords):
                        return np.eye(dim_target, dtype=float)
                    return fn
                eval_fns.append(make_id_fn())

    print(f"         Built {dim_target} eval functions: "
          f"{base_dim_stored} free + {dim_target-base_dim_stored} tied.")
    return eval_fns


def _get_tie_mode_description(mode: str) -> str:
    return {
        "cyclic_shift":           "X_k = X_{k%3}(coords + k//3), cyclic lattice translations",
        "symmetric_permutation":  "X_k = P^T · X_{k%3} · P, row/col permuted embedding",
        "identity_extension":     "X_k = I for k >= base_dim (trivial baseline)",
    }.get(mode, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Master: run_full_diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def run_full_diagnostic(
    fns: list,
    dim: int,
    label: str = "",
    walk_depth: int = 400,
    n_samples: int = 50,
    run_tying: bool = False,
    tying_base_params: Optional[dict] = None,
) -> dict:
    """
    Run all structural diagnostics on a set of CMF evaluation functions.

    Call this when a walk collapses (T2 fails) or when investigating the
    8→10 phase transition in Agent C.

    Parameters
    ----------
    fns              : list of d callables from agent_c_large.build_eval_fns()
    dim              : d
    label            : identifier for logging (e.g., "10x10_trial_0042")
    walk_depth       : depth for norm trajectory and spectral walk
    n_samples        : samples for det and spectral checks
    run_tying        : if True, also build tied system and re-run diagnostics
    tying_base_params: required if run_tying=True

    Returns
    -------
    dict with all diagnostic results plus a 'verdict' and 'recommended_fix'.
    """
    sep = "═" * 68
    print(f"\n{_BOLD}{sep}")
    print(f"  STRUCTURAL COLLAPSE DIAGNOSTIC  —  {label or f'dim={dim}'}")
    print(f"  dim={dim}  ({dim} matrices of size {dim}×{dim})")
    print(f"  Path-independence conditions: C({dim},2) = {dim*(dim-1)//2}")
    print(f"{sep}{_RESET}")
    print(f"  Understanding the 8→10 phase transition:")
    print(f"    d=8 :  28 flatness conditions — survives (CMF walk non-trivial)")
    print(f"    d=10:  45 flatness conditions — collapses (walk → 0)")
    print(f"  We need to know WHY.  Running 4 targeted diagnostics...\n")

    t0 = time.time()

    # ── Diagnostic 1 ──────────────────────────────────────────────────────────
    det_result = det_volume_check(fns, dim, n_samples=n_samples)

    # ── Diagnostic 2 ──────────────────────────────────────────────────────────
    spec_result = spectral_gap_analysis(fns, dim, walk_depth=min(walk_depth, 200),
                                        n_point_samples=n_samples)

    # ── Diagnostic 3 ──────────────────────────────────────────────────────────
    norm_result = walk_norm_trajectory(fns, dim, walk_depth=walk_depth)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    det_bad    = det_result["collapse_mode"] not in ("none",)
    spec_bad   = spec_result["collapse_risk"] in ("critical", "high")
    norm_bad   = norm_result["failure_mode"] not in ("none",)

    # Priority-ordered verdict
    if norm_result["failure_mode"] == "numerical_blackhole" and not spec_bad:
        verdict = "NUMERICAL_BLACKHOLE"
        recommended_fix = (
            "The math is potentially fine — it's a floating-point issue. "
            "ACTION: (1) Switch _walk_mp() to mpmath dps=100-150 for dim > 8. "
            "(2) Normalize v every step (not every 50). "
            "(3) Use mpmath.matrix() throughout instead of numpy for dim > 8."
        )
    elif spec_result["collapse_risk"] == "critical":
        verdict = "SPECTRAL_COLLAPSE"
        recommended_fix = (
            "The matrices have eigenvalues < 1 everywhere — mathematical failure. "
            "ACTION: (1) Add det(X_i) ≥ ε constraint in _t1_pole_check(). "
            "(2) Use slope a_k ∈ {+1, +2} only (no negative slopes in D_params) "
            "to ensure Pochhammer growth n·(n+1)···(n+d-1) > 1. "
            "(3) Force D_params[k][0] > 0 in sample_params()."
        )
    elif det_result["collapse_mode"] == "analytic":
        verdict = "DETERMINANT_COLLAPSE"
        recommended_fix = (
            "Matrices are nearly singular everywhere — optimizer found the trivial solution. "
            "ACTION: (1) Add hard rejection if mean|det(X_i)| < 0.1 in _t1_pole_check(). "
            "(2) Normalize each X_i to have det = 1 (SL(d) constraint). "
            "(3) Or use _build_Di() with Pochhammer coordinates: det(D_i) = n·(n+1)···(n+d-1) > 1."
        )
    elif norm_result["failure_mode"] == "analytic_decay":
        verdict = "ANALYTIC_DECAY"
        recommended_fix = (
            "Walk decays exponentially — eigenvalues are uniformly < 1. "
            "ACTION: (1) Increase _SLOPE_VALS to include larger values (+3, +4). "
            "(2) Use start = [d, d, d, ...] instead of [2,2,2,...] (avoid small n). "
            "(3) Consider a 'warm start' walk (skip first 50 steps where n is small)."
        )
    elif det_bad or spec_bad or norm_bad:
        verdict = "PARTIAL_COLLAPSE"
        recommended_fix = (
            f"Mixed signals: det={'BAD' if det_bad else 'ok'}, "
            f"spec={'BAD' if spec_bad else 'ok'}, norm={'BAD' if norm_bad else 'ok'}. "
            "ACTION: Run with larger n_samples and walk_depth for clearer signal."
        )
    else:
        verdict = "HEALTHY"
        recommended_fix = "No structural collapse detected. Check T3 depth thresholds."

    # Print synthesis
    print(f"\n{_BOLD}{sep}")
    print(f"  SYNTHESIS  —  {label or f'dim={dim}'}")
    print(f"{sep}{_RESET}")
    print(f"\n  {'✗' if det_bad  else '✓'} DET VOLUME : {det_result['collapse_mode']:20s}  "
          f"penalty={det_result['penalty_score']:.2f}")
    print(f"  {'✗' if spec_bad else '✓'} SPECTRAL   : {spec_result['collapse_risk']:20s}  "
          f"geo_mean_rho={spec_result['geometric_mean_rho']:.4f}")
    print(f"  {'✗' if norm_bad else '✓'} NORM TRAJ  : {norm_result['failure_mode']:20s}  "
          f"decay_slope={norm_result['decay_slope']:.5f}")

    colour = _RED if verdict not in ("HEALTHY",) else _GREEN
    print(f"\n  {colour}{_BOLD}VERDICT: {verdict}{_RESET}")
    print(f"\n  {_YELLOW}RECOMMENDED FIX:{_RESET}")
    for line in recommended_fix.split(". "):
        if line.strip():
            print(f"    {line.strip()}.")

    total_t = time.time() - t0
    print(f"\n  Total diagnostic time: {total_t:.1f}s")
    print(f"{sep}\n")

    result = {
        "label":            label,
        "dim":              dim,
        "verdict":          verdict,
        "recommended_fix":  recommended_fix,
        "det_volume":       det_result,
        "spectral_gap":     spec_result,
        "norm_trajectory":  norm_result,
        "elapsed_s":        total_t,
    }

    # ── Optional tying mode ────────────────────────────────────────────────────
    if run_tying and tying_base_params is not None:
        print(f"\n{_BOLD}  Running tied-parameter fallback (dim {tying_base_params['dim']} → {dim})...{_RESET}")
        try:
            fns_tied = build_eval_fns_tied(tying_base_params, dim_target=dim)
            print(f"  Re-running norm trajectory with tied system...")
            tied_norm = walk_norm_trajectory(fns_tied, dim, walk_depth=walk_depth)
            result["tying_norm_trajectory"] = tied_norm
            if tied_norm["failure_mode"] == "none" and norm_bad:
                print(f"\n  {_GREEN}TYING FIXES THE COLLAPSE!{_RESET} "
                      f"Tied system is healthy while free system collapses. "
                      f"The overconstrained free system has no non-trivial solution — "
                      f"tying is the correct search strategy for dim={dim}.")
            elif tied_norm["failure_mode"] != "none" and norm_bad:
                print(f"\n  {_RED}Tying does NOT fix the collapse.{_RESET} "
                      f"The base matrices themselves are degenerate.")
        except Exception as e:
            print(f"  {_RED}Tying failed: {e}{_RESET}")
            result["tying_error"] = str(e)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Standalone runner  (python3 diagnostics.py --dim 10 --depth 300)
# ──────────────────────────────────────────────────────────────────────────────

def _make_demo_fns(dim: int) -> tuple[list, dict]:
    """
    Build a synthetic demo CMF system for standalone testing.
    Uses the same LDU gauge construction as Agent C.
    """
    sys.path.insert(0, str(HERE))
    from agent_c_large import sample_params, build_eval_fns
    rng    = np.random.default_rng(42)
    params = sample_params(dim, rng)
    fns    = build_eval_fns(params)
    return fns, params


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="CMF Structural Collapse Diagnostics (phase-transition debugger)")
    ap.add_argument("--dim",         type=int, default=10,
                    help="CMF dimension to diagnose (default: 10)")
    ap.add_argument("--depth",       type=int, default=400,
                    help="Walk depth for norm trajectory")
    ap.add_argument("--samples",     type=int, default=60,
                    help="Number of lattice-point samples")
    ap.add_argument("--tying",       action="store_true",
                    help="Also run tied-parameter fallback diagnostic")
    ap.add_argument("--store-file",  type=str, default="",
                    help="Load params from a JSONL store file (first record used)")
    ap.add_argument("--tie-mode",    type=str, default="cyclic_shift",
                    choices=["cyclic_shift", "symmetric_permutation", "identity_extension"])
    args = ap.parse_args()

    dim = args.dim

    if args.store_file:
        # Load real params from a store file
        import json
        sys.path.insert(0, str(HERE))
        from agent_c_large import build_eval_fns

        store_path = Path(args.store_file)
        if not store_path.exists():
            store_path = HERE / args.store_file
        lines = [l for l in store_path.read_text().splitlines() if l.strip()]
        if not lines:
            print(f"ERROR: {store_path} is empty."); sys.exit(1)
        rec = json.loads(lines[0])
        raw = rec.get("params", {})
        # Deserialise tuple keys
        params = {
            "dim":      raw["dim"],
            "D_params": [tuple(x) for x in raw["D_params"]],
            "L_off":    {eval(k): v for k, v in raw["L_off"].items()},
            "U_off":    {eval(k): v for k, v in raw["U_off"].items()},
        }
        dim = params["dim"]
        fns = build_eval_fns(params)
        label = f"stored_{store_path.stem}"
        print(f"  Loaded {label} (dim={dim}) from {store_path.name}")
    else:
        print(f"  Building synthetic dim={dim} CMF system for demonstration...")
        fns, params = _make_demo_fns(dim)
        label = f"synthetic_dim{dim}"

    run_full_diagnostic(
        fns, dim,
        label=label,
        walk_depth=args.depth,
        n_samples=args.samples,
        run_tying=args.tying,
        tying_base_params=params if args.tying else None,
    )
