#!/usr/bin/env python3
"""
asymptotic_filter.py — 3-Step Asymptotic Fast-Fail Filter
==========================================================
Kills diverging / collapsing CMF candidates in milliseconds before
any expensive T2/T3 walk is attempted.

Step 1 — Degree Balance (Determinant Asymptotics)
  Estimate k where det(M_0(n)) ~ C * n^k by sampling at n = 200, 400, 800.
  For LDU gauge bootstrap we expect k ≈ dim (linear D_i entries).
  Reject if k < -1  (collapse to 0)  or k > 3·dim + 2  (explosive growth).

Step 2 — Spectral Condition of the Limit Matrix
  Evaluate M_0 at n=500 (moderate, avoids float overflow).
  Normalize by Frobenius norm, compute SVD.
  Reject if condition number > 10^dim  (one direction dominates → walk diverges).

Step 3 — Early-Stopping Norm Ratio
  Walk 20 steps along axis 0 with running re-normalisation.
  R = norm(step 20) / norm(step 10).
  Reject if R > 1e6 (diverging) or R < 1e-6 (collapsing).

Usage:
    from asymptotic_filter import asymptotic_filter
    ok, reason = asymptotic_filter(eval_fns, dim)
    if not ok:
        continue   # fast reject
"""
from __future__ import annotations
import math
from typing import Callable

import numpy as np

__all__ = ["asymptotic_filter"]


def asymptotic_filter(
    eval_fns: list[Callable],
    dim: int,
    n_vars: int | None = None,
) -> tuple[bool, str]:
    """
    Run all 3 filter steps.

    Parameters
    ----------
    eval_fns : list of callables
        eval_fns[i](*coords) -> dim×dim array-like
    dim      : matrix size
    n_vars   : number of coordinate arguments each fn takes (default = len(eval_fns))

    Returns
    -------
    (True, "ok")              if candidate passes all steps
    (False, "<reason>")       if rejected with explanation
    """
    if n_vars is None:
        n_vars = len(eval_fns)
    fn0 = eval_fns[0]

    # ── Step 1: Degree Balance ─────────────────────────────────────────────────
    try:
        base = 200
        results: list[tuple[float, float]] = []
        for mult in (1, 2, 4):
            n = base * mult
            coords = [n] + [5] * (n_vars - 1)
            M = np.asarray(fn0(*coords), dtype=float)
            if not np.all(np.isfinite(M)):
                return False, "step1_nonfinite"
            sign, logd = np.linalg.slogdet(M)
            if sign == 0:
                return False, "step1_singular"
            results.append((math.log(n), float(logd)))

        # Fit k: log|det| = k·log(n) + c
        log_ns = [r[0] for r in results]
        log_ds = [r[1] for r in results]
        n_pts = len(log_ns)
        sx  = sum(log_ns)
        sy  = sum(log_ds)
        sxx = sum(x * x for x in log_ns)
        sxy = sum(x * y for x, y in zip(log_ns, log_ds))
        denom = n_pts * sxx - sx * sx
        if abs(denom) < 1e-12:
            return False, "step1_degenerate_fit"
        k = (n_pts * sxy - sx * sy) / denom

        if k < -1.0:
            return False, f"step1_collapse(k={k:.2f})"
        if k > 3.0 * dim + 2.0:
            return False, f"step1_explosion(k={k:.2f})"

    except Exception as exc:
        return False, f"step1_error({exc})"

    # ── Step 2: Spectral Condition of Normalised Limit Matrix ─────────────────
    try:
        n_mid  = 500
        coords = [n_mid] + [5] * (n_vars - 1)
        M_raw  = np.asarray(fn0(*coords), dtype=float)
        if not np.all(np.isfinite(M_raw)):
            return False, "step2_nonfinite"

        frob = np.linalg.norm(M_raw, "fro")
        if frob < 1e-200:
            return False, "step2_zero_matrix"

        sv    = np.linalg.svd(M_raw / frob, compute_uv=False)
        cond  = float(sv[0]) / (float(sv[-1]) + 1e-300)
        limit = 10.0 ** min(dim + 3, 14)   # generous: 10^(dim+3), max 10^14
        if cond > limit:
            return False, f"step2_ill_conditioned(cond={cond:.2e},lim={limit:.2e})"

    except Exception as exc:
        return False, f"step2_error({exc})"

    # ── Step 3: Early-Stopping Norm Ratio ─────────────────────────────────────
    try:
        coords = [2] * n_vars
        v      = np.zeros(dim, dtype=float)
        v[0]   = 1.0
        norm10 = None

        for step in range(20):
            coords[0] += 1
            M = np.asarray(fn0(*coords), dtype=float)
            if not np.all(np.isfinite(M)):
                return False, "step3_nonfinite_walk"
            v = M @ v
            norm = float(np.linalg.norm(v))
            if norm < 1e-150:
                return False, f"step3_collapse_walk(step={step})"
            v /= norm
            if step == 9:
                norm10 = norm

        norm20 = norm
        if norm10 is None or norm10 < 1e-150:
            return False, "step3_zero_norm10"

        R = norm20 / norm10
        if R > 1e6:
            return False, f"step3_diverge(R={R:.2e})"
        if R < 1e-6:
            return False, f"step3_collapse(R={R:.2e})"

    except Exception as exc:
        return False, f"step3_error({exc})"

    return True, "ok"
