#!/usr/bin/env python3
"""
reward_engine.py — Multi-Objective CMF Reward Calculator
=========================================================
Score = w1*conv_rate + w2*ray_stability + w3*identifiability
      + w4*simplicity + w5*proofability - w6*pole_penalty
"""
from __future__ import annotations
import math, hashlib, json
from typing import Optional
import numpy as np
import mpmath as mp
import sympy as sp
from sympy import symbols, det, Rational, Poly, degree, cancel, factor

# ── Weights ───────────────────────────────────────────────────────────────────
W = dict(conv_rate=0.35, ray_stability=0.20, identifiability=0.20,
         simplicity=0.15, proofability=0.10, pole_penalty=1000.0)

# ── Constant bank for ISC matching ───────────────────────────────────────────
_BANK: list[dict] = []

def _build_bank(dps: int = 80) -> list[dict]:
    mp.mp.dps = dps
    raw = [
        ("pi",      mp.pi),   ("pi^2",    mp.pi**2),  ("pi^3", mp.pi**3),
        ("zeta(2)", mp.zeta(2)), ("zeta(3)", mp.zeta(3)), ("zeta(5)", mp.zeta(5)),
        ("ln2",     mp.log(2)),  ("ln3",    mp.log(3)),   ("sqrt2", mp.sqrt(2)),
        ("sqrt3",   mp.sqrt(3)), ("Catalan", mp.catalan), ("euler", mp.euler),
        ("1/pi",    1/mp.pi),    ("4/pi",   4/mp.pi),
        ("pi/sqrt3", mp.pi/mp.sqrt(3)),
    ]
    bank = []
    for name, v in raw:
        for sign, sn in [(1, ""), (-1, "-")]:
            bank.append({"name": sn+name, "val": sign*v, "fval": float(sign*v)})
    # rationals
    for n in range(1, 8):
        for d in range(1, 8):
            if math.gcd(n, d) == 1:
                bank.append({"name": f"{n}/{d}", "val": mp.mpf(n)/d,
                              "fval": n/d})
    return bank

def get_bank() -> list[dict]:
    global _BANK
    if not _BANK:
        _BANK = _build_bank()
    return _BANK


# ── Core walkers ─────────────────────────────────────────────────────────────

def _walk_product_numeric(matrices_fn, dim: int, direction: tuple,
                          depth: int = 800, dps: int = 50) -> Optional[float]:
    """Vector walk along a lattice ray — O(d²) per step vs O(d³) for matrix walk.
    Tracks v = M_k * ... * M_1 * e_0; returns v[0]/v[dim-1].
    """
    mp.mp.dps = dps
    v = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)
    pos = [1, 1, 1]
    for step in range(1, depth + 1):
        ax = step % len(direction)
        pos[ax] += direction[ax]
        try:
            M = matrices_fn(tuple(pos), ax)
        except Exception:
            return None
        v = mp.matrix(M) * v
        if step % 100 == 0:
            scale = max(abs(v[i]) for i in range(dim))
            if scale > mp.mpf(10)**40:
                v /= scale
    if abs(v[dim - 1]) < mp.mpf(10)**(-(dps - 5)):
        return None
    return float(v[0] / v[dim - 1])


def _estimate_delta(matrices_fn, dim: int, direction: tuple, dps: int = 50) -> float:
    r1 = _walk_product_numeric(matrices_fn, dim, direction, depth=200, dps=dps)
    r2 = _walk_product_numeric(matrices_fn, dim, direction, depth=1000, dps=dps)
    if r1 is None or r2 is None:
        return 0.0
    diff = abs(r2 - r1)
    if diff < 1e-100:
        return 60.0
    return min(60.0, -math.log10(diff + 1e-120))


# ── Sub-scores ────────────────────────────────────────────────────────────────

def score_conv_rate(deltas: list[float]) -> float:
    if not deltas:
        return 0.0
    best = max(deltas)
    return min(1.0, best / 30.0)


def score_ray_stability(deltas: list[float]) -> float:
    if len(deltas) < 2:
        return 0.0
    valid = [d for d in deltas if d > 0.5]
    frac  = len(valid) / len(deltas)
    spread = np.std(deltas) if len(deltas) > 1 else 0.0
    return frac * max(0.0, 1.0 - spread / 20.0)


def score_identifiability(ratios: list[float], dps: int = 50) -> float:
    bank = get_bank()
    best = 0.0
    mp.mp.dps = dps
    for r in ratios:
        if not math.isfinite(r) or abs(r) > 1e12:
            continue
        r_mp = mp.mpf(r)
        for c in bank:
            err = abs(r_mp - c["val"])
            if err < mp.mpf(10)**(-6):
                digits = min(60.0, float(-mp.log10(err + mp.mpf(10)**-120)))
                best = max(best, digits)
        # nsimplify: catches algebraic/transcendental constants the bank misses
        try:
            ns = sp.nsimplify(
                float(r_mp),
                [sp.pi, sp.E, sp.log(2), sp.zeta(3), sp.Catalan,
                 sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)],
                rational=False, tolerance=1e-8,
            )
            if (ns not in (sp.nan, sp.zoo) and ns.is_number
                    and not ns.is_rational and not ns.is_integer):
                best = max(best, 20.0)
        except Exception:
            pass
    return min(1.0, best / 40.0)


def score_simplicity(sympy_matrices: list) -> float:
    """Penalise high degree, large integer coefficients, dense expressions."""
    total_terms = 0
    max_coeff   = 0
    max_deg     = 0
    for M in sympy_matrices:
        for expr in M:
            try:
                e = sp.expand(expr)
                total_terms += sp.count_ops(e)
                for c in e.atoms(sp.Integer, sp.Rational):
                    max_coeff = max(max_coeff, int(abs(c)))
                p = Poly(e)
                max_deg = max(max_deg, p.total_degree())
            except Exception:
                total_terms += 20
    penalty = total_terms / 200.0 + max_deg / 10.0 + math.log10(max_coeff + 2) / 4.0
    return max(0.0, 1.0 - penalty / 5.0)


def score_proofability(sympy_matrices: list) -> float:
    """
    Heuristic: reward Pochhammer-looking entries (linear factors in numerator/denominator),
    penalise irreducible high-degree factors.
    """
    linear_frac_count = 0
    total = 0
    for M in sympy_matrices:
        for expr in M:
            total += 1
            try:
                f = sp.factor(expr)
                num, den = sp.fraction(f)
                def count_linear(e):
                    c = 0
                    for arg in sp.preorder_traversal(e):
                        if isinstance(arg, sp.Mul) or isinstance(arg, sp.Pow):
                            continue
                        if isinstance(arg, sp.Add) and sp.degree(sp.Poly(arg)) == 1:
                            c += 1
                    return c
                linear_frac_count += count_linear(num) + count_linear(den)
            except Exception:
                pass
    if total == 0:
        return 0.0
    return min(1.0, linear_frac_count / (total * 3))


def check_poles(matrices_fn, dim: int, n_samples: int = 200) -> bool:
    """Return True (has poles) if any sample point gives a singular or exploding matrix."""
    import itertools
    rng = np.random.default_rng(0)
    coords_list = [tuple(rng.integers(1, 30, size=3).tolist()) for _ in range(n_samples)]
    for pos in coords_list:
        for ax in range(3):
            try:
                M = matrices_fn(pos, ax)
                M_np = np.array(M, dtype=complex)
                if not np.all(np.isfinite(M_np)):
                    return True
                if abs(np.linalg.det(M_np)) < 1e-12:
                    return True
            except (ZeroDivisionError, ValueError, OverflowError):
                return True
    return False


# ── Fingerprint / canonicalization ───────────────────────────────────────────

def fingerprint(matrices_fn, dim: int, probe_points: int = 12) -> str:
    """Numerical fingerprint for deduplication."""
    mp.mp.dps = 30
    vals = []
    rng = np.random.default_rng(777)
    for _ in range(probe_points):
        pos  = tuple(rng.integers(1, 10, size=3).tolist())
        ax   = int(rng.integers(0, 3))
        try:
            M = matrices_fn(pos, ax)
            for i in range(dim):
                for j in range(dim):
                    vals.append(round(float(M[i][j].real), 6))
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals, sort_keys=True).encode()).hexdigest()[:16]


# ── Master scorer ─────────────────────────────────────────────────────────────

def evaluate(matrices_fn, dim: int,
             sympy_matrices: Optional[list] = None,
             n_rays: int = 6, dps: int = 50) -> dict:
    """
    Main entry point.
    matrices_fn(pos: tuple, axis: int) -> dim×dim list-of-lists of floats/complex.
    Returns score dict.
    """
    # Pole check first — catastrophic penalty
    if check_poles(matrices_fn, dim):
        return {"score": -W["pole_penalty"], "pole": True,
                "conv_rate": 0, "ray_stability": 0,
                "identifiability": 0, "simplicity": 0, "proofability": 0,
                "deltas": [], "ratios": [], "best_delta": 0.0}

    # Build rays
    rays = []
    for ax in range(min(3, len(range(3)))):
        v = [0, 0, 0]; v[ax] = 1; rays.append(tuple(v))
    for ax1 in range(3):
        for ax2 in range(ax1+1, 3):
            v = [0, 0, 0]; v[ax1] = 1; v[ax2] = 1; rays.append(tuple(v))
    rays = rays[:n_rays]

    deltas, ratios = [], []
    for ray in rays:
        d  = _estimate_delta(matrices_fn, dim, ray, dps=dps)
        r  = _walk_product_numeric(matrices_fn, dim, ray, dps=dps)
        deltas.append(d)
        if r is not None and math.isfinite(r):
            ratios.append(r)

    s_conv   = score_conv_rate(deltas)
    s_ray    = score_ray_stability(deltas)
    s_ident  = score_identifiability(ratios, dps=dps)
    s_simp   = score_simplicity(sympy_matrices) if sympy_matrices else 0.5
    s_proof  = score_proofability(sympy_matrices) if sympy_matrices else 0.3

    score = (W["conv_rate"]      * s_conv   +
             W["ray_stability"]  * s_ray    +
             W["identifiability"]* s_ident  +
             W["simplicity"]     * s_simp   +
             W["proofability"]   * s_proof)

    return {
        "score": round(score, 6),
        "pole":  False,
        "conv_rate":       round(s_conv, 4),
        "ray_stability":   round(s_ray, 4),
        "identifiability": round(s_ident, 4),
        "simplicity":      round(s_simp, 4),
        "proofability":    round(s_proof, 4),
        "deltas":  [round(d, 3) for d in deltas],
        "ratios":  [round(r, 8) for r in ratios[:4]],
        "best_delta": round(max(deltas) if deltas else 0.0, 3),
    }
