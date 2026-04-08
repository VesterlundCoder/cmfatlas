#!/usr/bin/env python3
"""
reward_engine.py — Multi-Objective CMF Reward Calculator
=========================================================
Score = w1*conv_rate + w2*ray_stability + w3*identifiability
      + w4*simplicity + w5*proofability - w6*pole_penalty
"""
from __future__ import annotations
import math, hashlib, json, re
from fractions import Fraction
from typing import Optional
import numpy as np
import mpmath as mp
import sympy as sp
from sympy import symbols, det, Rational, Poly, degree, cancel, factor

# ── Weights ───────────────────────────────────────────────────────────────────
W = dict(conv_rate=0.22, ray_stability=0.12, identifiability=0.12,
         simplicity=0.08, proofability=0.04, positive_structure=0.10,
         pole_penalty=1000.0,
         irrationality_bonus=3.0, trivial_rational_penalty=0.05,
         massive_irrational_bonus=5.0, fatal_penalty=0.02)

RATIONAL_DENOM_THRESHOLD = 100_000

# ── Gate constants ────────────────────────────────────────────────────────────
# Gate 1: triviality bounds
GATE1_ABS_MIN   = 1e-4      # |L| below this → Zero Trap
GATE1_ABS_MAX   = 1e4       # |L| above this → Divergence Trap
GATE1_NEAR_PM1  = 1e-6      # ||L|±1| below this → ±1 Trivial

# Gate 2: algebraic purge — transcendental keywords that MUST appear in identify()
_TRANSCENDENTAL_RE = re.compile(
    r'\b(pi|zeta|log|catalan|euler|gamma|Li)\b|exp\(',
    re.IGNORECASE
)

# Gate 3: PSLQ strict basis and coefficient cap
GATE3_MAX_COEFF = 50
GATE3_DPS       = 100
_GATE3_BASIS_NAMES = ["1", "pi", "pi^2", "log2", "zeta2", "zeta3", "catalan"]

# Gate 4: residual threshold for MASSIVE bonus
GATE4_RESIDUAL  = 1e-50

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


# ── Irrationality check — 4-Gate strict filter ───────────────────────────────

def fast_rationality_check(r: float, max_denominator: int = RATIONAL_DENOM_THRESHOLD) -> bool:
    """
    Returns True if r is a TRIVIALLY RATIONAL number (expressible as p/q with q ≤ max_denominator).
    """
    if not math.isfinite(r) or abs(r) > 1e12:
        return False
    if abs(r) < 1e-15:
        return True
    frac = Fraction(r).limit_denominator(max_denominator)
    approx = float(frac)
    if abs(approx - r) / (abs(r) + 1e-30) < 1e-8:
        return True
    x = abs(r)
    for _ in range(5):
        a = int(x); x -= a
        if x < 1e-9:
            return True
        x = 1.0 / x
    return False


def _gate3_pslq_basis():
    """Build the strict 7-element transcendental PSLQ basis at GATE3_DPS."""
    mp.mp.dps = GATE3_DPS + 10
    return [
        mp.mpf(1),
        mp.pi,
        mp.pi ** 2,
        mp.log(2),
        mp.zeta(2),
        mp.zeta(3),
        mp.catalan,
    ]


def classify_limit_strict(val: float, hp_val=None, verbose: bool = True) -> dict:
    """
    4-gate strict irrationality classifier.  Returns a result dict with keys:
      gate_passed (int 1-4 = highest gate cleared, 0 = pre-gate fail)
      label       (str)   — FATAL_* or IRRATIONAL_* or TRUE_TRANSCENDENTAL
      score_mult  (float) — multiply base score by this
      reason      (str)   — human-readable explanation

    hp_val: optional mpmath.mpf at high precision (e.g. 60 dps from scout walk).
            If provided, it is used for PSLQ in Gates 3/4 instead of float(val).
    """
    def _log(msg):
        if verbose:
            print(f"      [IrrGate] {msg}")

    # ── Gate 1: Triviality / Zero Trapdoor ─────────────────────────────────────
    absval = abs(val)
    if absval < GATE1_ABS_MIN:
        msg = (f"REJECTED: Zero Trap (|L|={absval:.2e} < {GATE1_ABS_MIN})")
        _log(msg)
        return {"gate_passed": 0, "label": "FATAL_ZERO_TRAP",
                "score_mult": W["fatal_penalty"], "reason": msg}
    if absval > GATE1_ABS_MAX:
        msg = (f"REJECTED: Divergence Trap (|L|={absval:.2e} > {GATE1_ABS_MAX})")
        _log(msg)
        return {"gate_passed": 0, "label": "FATAL_DIVERGENCE_TRAP",
                "score_mult": W["fatal_penalty"], "reason": msg}
    if abs(val - 1.0) < GATE1_NEAR_PM1:
        msg = (f"REJECTED: Near +1 Trap (|L-1|={abs(val-1):.2e})")
        _log(msg)
        return {"gate_passed": 0, "label": "FATAL_NEAR_ONE",
                "score_mult": W["fatal_penalty"], "reason": msg}
    if abs(val + 1.0) < GATE1_NEAR_PM1:
        msg = (f"REJECTED: Near -1 Trap (|L+1|={abs(val+1):.2e})")
        _log(msg)
        return {"gate_passed": 0, "label": "FATAL_NEAR_MINUS_ONE",
                "score_mult": W["fatal_penalty"], "reason": msg}

    # Also reject trivial rationals (Gate 1 extension)
    if fast_rationality_check(val, 10_000):
        msg = (f"REJECTED: Trivial Rational (L={val!r})")
        _log(msg)
        return {"gate_passed": 0, "label": "FATAL_TRIVIAL_RATIONAL",
                "score_mult": W["fatal_penalty"], "reason": msg}

    _log(f"Gate 1 PASS  (|L|={absval:.6g})")

    # ── Gate 2: Algebraic Purge ────────────────────────────────────────────────
    identify_str: str | None = None
    try:
        mp.mp.dps = 50
        identify_str = mp.identify(mp.mpf(val))
    except Exception:
        identify_str = None

    if identify_str is not None:
        # Check for transcendental keywords
        has_transcendental = bool(_TRANSCENDENTAL_RE.search(identify_str))
        if not has_transcendental:
            msg = (f"REJECTED: Algebraic Trap — identify()='{identify_str}' "
                   f"(no transcendental keyword)")
            _log(msg)
            return {"gate_passed": 1, "label": "FATAL_ALGEBRAIC_ESCAPE",
                    "score_mult": W["fatal_penalty"], "reason": msg,
                    "identify_str": identify_str}
        _log(f"Gate 2 PASS  (identify='{identify_str[:60]}')")
    else:
        _log("Gate 2 PASS  (mpmath.identify returned None — unknown constant)")

    # ── Gate 3: Strict PSLQ Coefficient Cap ───────────────────────────────────
    pslq_relation: list | None  = None
    pslq_max_coeff: int         = 0
    pslq_residual: float        = float('inf')

    try:
        mp.mp.dps = GATE3_DPS + 10
        v_hp = (mp.mpf(mp.nstr(hp_val, GATE3_DPS))
                if hp_val is not None else mp.mpf(str(val)))
        basis    = [v_hp] + _gate3_pslq_basis()
        coeffs   = mp.pslq(basis, tol=mp.mpf(10) ** (-GATE3_DPS // 2),
                           maxcoeff=GATE3_MAX_COEFF + 10)
        if coeffs is not None and coeffs[0] != 0:
            pslq_relation  = list(coeffs)
            pslq_max_coeff = int(max(abs(c) for c in coeffs))
            pslq_residual  = float(abs(sum(
                c * b for c, b in zip(coeffs, basis)
            )))

            if pslq_max_coeff > GATE3_MAX_COEFF:
                msg = (f"REJECTED: PSLQ Overfit "
                       f"(max coeff {pslq_max_coeff} > {GATE3_MAX_COEFF})")
                _log(msg)
                return {"gate_passed": 2, "label": "FATAL_PSLQ_OVERFIT",
                        "score_mult": W["fatal_penalty"], "reason": msg,
                        "pslq_max_coeff": pslq_max_coeff,
                        "identify_str": identify_str}

            names = ["L"] + _GATE3_BASIS_NAMES
            rel_str = " + ".join(
                f"({c})*{n}" for c, n in zip(coeffs, names) if c != 0
            ) + " = 0"
            _log(f"Gate 3 PASS  (max_coeff={pslq_max_coeff}, "
                 f"residual={pslq_residual:.2e}, relation={rel_str[:70]})")
        else:
            _log(f"Gate 3 PASS  (PSLQ: no relation involving L found with maxcoeff={GATE3_MAX_COEFF})")
    except Exception as exc:
        _log(f"Gate 3 PASS  (PSLQ exception: {exc})")

    # ── Gate 4: True Transcendental Reward ────────────────────────────────────
    if pslq_relation is not None and pslq_residual < GATE4_RESIDUAL:
        names = ["L"] + _GATE3_BASIS_NAMES
        rel_str = " + ".join(
            f"({c})*{n}" for c, n in zip(pslq_relation, names) if c != 0
        ) + " = 0"
        msg = (f"TRUE_TRANSCENDENTAL: PSLQ residual={pslq_residual:.2e} "
               f"< {GATE4_RESIDUAL}, relation={rel_str}")
        _log(msg)
        return {"gate_passed": 4, "label": "TRUE_TRANSCENDENTAL",
                "score_mult": W["massive_irrational_bonus"], "reason": msg,
                "pslq_relation": pslq_relation, "pslq_residual": pslq_residual,
                "identify_str": identify_str}

    # Passed all gates but not positively identified → still a strong candidate
    msg = (f"IRRATIONAL_UNKNOWN: passed all 4 gates, no PSLQ match in strict basis")
    _log(msg)
    return {"gate_passed": 3, "label": "IRRATIONAL_UNKNOWN",
            "score_mult": W["irrationality_bonus"], "reason": msg,
            "pslq_residual": pslq_residual if pslq_relation else None,
            "identify_str": identify_str}


def score_irrationality(ratios: list[float],
                        max_denominator: int = RATIONAL_DENOM_THRESHOLD
                        ) -> tuple[float, bool, dict]:
    """
    Returns (irrationality_score, is_fatal, gate_result).
    Applies the 4-gate strict filter to the median of valid ratios.
    """
    valid = [r for r in ratios if math.isfinite(r) and 1e-15 < abs(r) < 1e15]
    if not valid:
        return 0.5, False, {"gate_passed": -1, "label": "NO_DATA",
                            "score_mult": 1.0, "reason": "no valid ratios"}

    # Use the median value as the canonical limit
    median_val = float(sorted(valid)[len(valid) // 2])
    gate_result = classify_limit_strict(median_val, verbose=True)

    is_fatal   = gate_result["label"].startswith("FATAL")
    score      = 0.0 if is_fatal else (1.0 if gate_result["gate_passed"] >= 3 else 0.5)
    return score, is_fatal, gate_result


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


def score_positive_structure(params: Optional[dict]) -> float:
    """
    Reward all-positive D_params slopes and half-integer shift values.

    Mathematical basis:
      * Positive slopes a_k > 0  →  no sign oscillation along any walk axis.
      * Half-integer shifts b_k ∈ {±½, ±3/2, …}  →  products involve
        Γ(n+½) / Γ(n+3/2) etc., which carry √π factors and can produce
        transcendental (irrational) limits via Γ-function ratios.

    Returns: 1.0  — all positive slopes AND all half-integer shifts
             0.6  — all positive slopes only
             0.2  — any negative slope (oscillation risk)
    """
    if params is None:
        return 0.3
    d_params = params.get("D_params", [])
    if not d_params:
        return 0.3

    slopes = [a for a, b in d_params]
    shifts = [b for a, b in d_params]

    all_positive = all(a > 0 for a in slopes)
    _is_half_int = lambda b: abs((abs(b) % 1.0) - 0.5) < 1e-9
    all_half_int = all(_is_half_int(b) for b in shifts)

    if all_positive and all_half_int:
        return 1.0
    elif all_positive:
        return 0.6
    else:
        return 0.2


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
             params: Optional[dict] = None,
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
    s_pos    = score_positive_structure(params)
    s_irrat, is_fatal, gate_result = score_irrationality(ratios)

    base = (W["conv_rate"]         * s_conv   +
            W["ray_stability"]     * s_ray    +
            W["identifiability"]   * s_ident  +
            W["simplicity"]        * s_simp   +
            W["proofability"]      * s_proof  +
            W["positive_structure"]* s_pos)

    # Apply 4-gate multiplier — only if CMF actually converges
    irrat_label = gate_result["label"]
    if s_conv > 0.1:
        score = base * gate_result["score_mult"]
    else:
        score = base
        irrat_label = "NO_CONVERGENCE"

    return {
        "score": round(score, 6),
        "pole":  False,
        "conv_rate":            round(s_conv, 4),
        "ray_stability":        round(s_ray, 4),
        "identifiability":      round(s_ident, 4),
        "simplicity":           round(s_simp, 4),
        "proofability":         round(s_proof, 4),
        "irrationality_score":  round(s_irrat, 4),
        "irrationality_label":  irrat_label,
        "is_trivial_rational":  is_fatal,
        "positive_structure":   round(s_pos, 4),
        "gate_result":          gate_result,
        "deltas":  [round(d, 3) for d in deltas],
        "ratios":  [round(r, 8) for r in ratios[:4]],
        "best_delta": round(max(deltas) if deltas else 0.0, 3),
    }
