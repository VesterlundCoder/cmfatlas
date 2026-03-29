"""Smart gap-filling: telescope construction + 2D promotion + 3D extension.

Strategy:
  1. For each dfinite gap (total_degree, max_var_degree), generate conjugate
     polynomials f(x,y) of exactly that degree with multiple conjugacies.
  2. Build telescope CMFs — these are GUARANTEED flat by construction.
  3. Attempt 2D promotion: check if a(k,m) truly depends on m.
  4. Attempt 3D extension for any truly-2D CMFs found.
  5. Evaluate limits numerically + run PSLQ recognition.
  6. For PCF gaps: generate Apéry-like PCFs with exact target degrees.
"""

import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mpmath
import numpy as np
import sympy as sp
from sympy import Rational, expand, simplify, symbols, Poly, Integer

from cmf_atlas.db.session import get_engine, get_session, init_db
from cmf_atlas.db.models import (
    CMF, EvalRun, Features as FeaturesModel, Project, Representation, Series,
)
from cmf_atlas.density.gaps import build_gap_atlas
from cmf_atlas.canonical import canonicalize_and_fingerprint
from cmf_atlas.features.compute import compute_features
from cmf_atlas.recognition.pslq import run_pslq
from cmf_atlas.util.json import dumps
from cmf_atlas.util.logging import get_logger

log = get_logger("smart_fill")

DB_PATH = Path(__file__).parent / "data" / "atlas.db"
RESULTS_PATH = Path(__file__).parent / "data" / "smart_gap_fill_results.json"

x, y, z = symbols("x y z")
k, m, n_sym = symbols("k m n", integer=True)

# ── Conjugacies ──────────────────────────────────────────────────────────
CONJUGACIES = {
    "neg":      lambda f: f.subs(x, -x).subs(y, -y),
    "negx":     lambda f: f.subs(x, -x),
    "negy":     lambda f: f.subs(y, -y),
    "neg_negx": lambda f: -f.subs(x, -x),
    "neg_negy": lambda f: -f.subs(y, -y),
}


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1: Generate polynomials of target degree
# ══════════════════════════════════════════════════════════════════════════
def generate_polynomials(total_deg: int, max_var_deg: int, coeff_range: int = 3) -> list[sp.Expr]:
    """Generate all monomials up to total_degree=total_deg with each variable
    degree <= max_var_deg, then form polynomials with small integer coefficients."""
    monoms = []
    for dx in range(min(total_deg, max_var_deg) + 1):
        for dy in range(min(total_deg - dx, max_var_deg) + 1):
            monoms.append(x**dx * y**dy)

    polys = []
    # Strategy 1: single monomials with coefficient
    for monom in monoms:
        p = Poly(monom, x, y)
        if p.total_degree() == total_deg:
            for c in range(-coeff_range, coeff_range + 1):
                if c != 0:
                    polys.append(c * monom)

    # Strategy 2: sum of 2 monomials (one at max degree, one lower)
    high_monoms = [m for m in monoms if Poly(m, x, y).total_degree() == total_deg]
    low_monoms = [m for m in monoms if Poly(m, x, y).total_degree() < total_deg]

    for hm in high_monoms:
        for lm in low_monoms[:10]:  # limit combinations
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    polys.append(c1 * hm + c2 * lm)

    # Strategy 3: dense polynomials (all monomials up to degree)
    if total_deg <= 3:
        relevant = [m for m in monoms if Poly(m, x, y).total_degree() <= total_deg]
        # Random subsets of 3-5 monomials (must include at least one at max degree)
        import random
        random.seed(42 + total_deg * 100 + max_var_deg)
        for _ in range(50):
            n_terms = random.randint(2, min(5, len(relevant)))
            must_have = random.choice(high_monoms) if high_monoms else relevant[0]
            others = random.sample([m for m in relevant if m != must_have], min(n_terms - 1, len(relevant) - 1))
            terms = [must_have] + others
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in terms]
            polys.append(sum(c * t for c, t in zip(coeffs, terms)))

    # Deduplicate
    seen = set()
    unique = []
    for p in polys:
        p_expanded = expand(p)
        key = str(p_expanded)
        if key not in seen and p_expanded != 0:
            seen.add(key)
            unique.append(p_expanded)

    return unique


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2: Build telescope CMF from f(x,y) + conjugacy
# ══════════════════════════════════════════════════════════════════════════
def build_telescope(f_expr, conjugacy_name: str):
    """Build telescope CMF matrices from f(x,y) and conjugacy.

    Returns (a_km, b_k, g_km, gbar_km, flat_ok) where:
      K1(k,m) = [[a(k,m), 1], [b(k+1), 0]]
      K2(k,m) = [[gbar(k,m), 1], [b(k), g(k,m)]]
    and a(k,m) = g(k,m) - gbar(k+1,m).

    Flatness is guaranteed by construction for telescope CMFs.
    """
    conj_fn = CONJUGACIES.get(conjugacy_name)
    if conj_fn is None:
        return None

    fbar_expr = conj_fn(f_expr)

    # g(k,m) = f(k,m), gbar(k,m) = fbar(k,m)
    g_km = f_expr.subs([(x, k), (y, m)])
    gbar_km = fbar_expr.subs([(x, k), (y, m)])

    # b(k) = g(k,0) * gbar(k,0)
    b_k = expand(g_km.subs(m, 0) * gbar_km.subs(m, 0))

    # a(k,m) = g(k,m) - gbar(k+1,m)
    a_km = expand(g_km - gbar_km.subs(k, k + 1))

    return {
        "f_poly": str(f_expr),
        "fbar_poly": str(fbar_expr),
        "conjugacy": conjugacy_name,
        "g_km": g_km,
        "gbar_km": gbar_km,
        "b_k": b_k,
        "a_km": a_km,
    }


def verify_flatness_numeric(tel, n_test: int = 10) -> tuple[bool, float]:
    """Numerically verify flatness of a telescope CMF."""
    g_fn = sp.lambdify((k, m), tel["g_km"], "mpmath")
    gbar_fn = sp.lambdify((k, m), tel["gbar_km"], "mpmath")
    b_fn = sp.lambdify(k, tel["b_k"], "mpmath")
    a_fn = sp.lambdify((k, m), tel["a_km"], "mpmath")

    mpmath.mp.dps = 30
    max_diff = 0.0

    for kv in range(2, 2 + n_test):
        for mv in range(1, 4):
            try:
                MX = np.array([
                    [0, 1],
                    [float(b_fn(kv + 1)), float(a_fn(kv, mv))]
                ], dtype=float)
                MY = np.array([
                    [float(gbar_fn(kv, mv)), 1],
                    [float(b_fn(kv)), float(g_fn(kv, mv))]
                ], dtype=float)
                MY_k1 = np.array([
                    [float(gbar_fn(kv + 1, mv)), 1],
                    [float(b_fn(kv + 1)), float(g_fn(kv + 1, mv))]
                ], dtype=float)
                MX_m1 = np.array([
                    [0, 1],
                    [float(b_fn(kv + 1)), float(a_fn(kv, mv + 1))]
                ], dtype=float)

                LHS = MX @ MY_k1
                RHS = MY @ MX_m1
                diff = np.abs(LHS - RHS)
                for i in range(2):
                    for j in range(2):
                        scale = max(abs(LHS[i, j]), abs(RHS[i, j]), 1.0)
                        max_diff = max(max_diff, diff[i, j] / scale)
            except Exception:
                return False, float("inf")

    return max_diff < 1e-10, max_diff


def check_truly_2d(tel) -> bool:
    """Check if a(k,m) genuinely depends on m (not just k)."""
    a_km = tel["a_km"]
    # Differentiate w.r.t. m — if nonzero, it's truly 2D
    da_dm = sp.diff(a_km, m)
    return da_dm != 0


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3: Evaluate limits numerically
# ══════════════════════════════════════════════════════════════════════════
def evaluate_telescope_limit(tel, m_val: int = 0, N: int = 500, dps: int = 50) -> dict:
    """Evaluate the limit of a telescope CMF at a specific m value.

    Computes the product walk: prod_{k=start}^{N} K1(k, m_val) applied to [1, 0]^T.
    """
    mpmath.mp.dps = dps + 20

    a_fn = sp.lambdify((k, m), tel["a_km"], "mpmath")
    b_fn = sp.lambdify(k, tel["b_k"], "mpmath")

    try:
        # Product walk: accumulate K1(k,m) = [[a(k,m), 1], [b(k+1), 0]]
        # Using the standard convergent ratio p_n/q_n
        p_prev = mpmath.mpf(1)
        p_curr = mpmath.mpf(a_fn(1, m_val))
        q_prev = mpmath.mpf(0)
        q_curr = mpmath.mpf(1)

        estimates = []
        for kv in range(2, N + 1):
            a_val = mpmath.mpf(a_fn(kv, m_val))
            b_val = mpmath.mpf(b_fn(kv))

            p_new = a_val * p_curr + b_val * p_prev
            q_new = a_val * q_curr + b_val * q_prev
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new

            if q_curr != 0 and kv % 10 == 0:
                estimates.append(float(p_curr / q_curr))

        if not estimates:
            return {"limit": None, "conv_score": 0.0, "error": "no estimates"}

        limit = estimates[-1]
        if len(estimates) >= 3:
            tail = estimates[-5:]
            error = max(abs(tail[i] - tail[i - 1]) for i in range(1, len(tail)))
        else:
            error = float("inf")

        if error > 0 and error < 1e10:
            stable_digits = max(0, -math.log10(max(error, 1e-50)))
            conv_score = min(1.0, stable_digits / dps)
        else:
            conv_score = 0.0

        return {
            "limit": mpmath.nstr(mpmath.mpf(limit), dps - 5) if conv_score > 0.1 else str(limit),
            "limit_float": limit,
            "conv_score": conv_score,
            "error": error,
            "m_val": m_val,
            "N": N,
        }

    except Exception as e:
        return {"limit": None, "conv_score": 0.0, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4: 3D extension attempts
# ══════════════════════════════════════════════════════════════════════════
def attempt_3d_extension(tel) -> list[dict]:
    """For a truly-2D telescope CMF, attempt to find K3(k,m,n).

    Known approach: for f(x,y,z) = f(x,y) + δ*z, try different δ values
    and check if the 3-direction flatness equations hold.
    Also try f(x,y,z) with z-dependence in higher-degree terms.
    """
    results = []
    f_str = tel["f_poly"]
    f_expr = sp.sympify(f_str.replace("^", "**"))
    conjugacy = tel["conjugacy"]
    conj_fn = CONJUGACIES.get(conjugacy)
    if not conj_fn:
        return results

    # For each extension f_3d = f(x,y) + δ*z, check 3D flatness
    for delta in range(-3, 4):
        if delta == 0:
            continue

        f_3d = f_expr + delta * z
        fbar_3d = conj_fn(f_3d)

        # Build 3D telescope
        g_3d = f_3d.subs([(x, k), (y, m), (z, n_sym)])
        gbar_3d = fbar_3d.subs([(x, k), (y, m), (z, n_sym)])

        # b(k) from 3D: g(k,0,0)*gbar(k,0,0)
        b_3d = expand(g_3d.subs([(m, 0), (n_sym, 0)]) * gbar_3d.subs([(m, 0), (n_sym, 0)]))

        # K1(k,m,n) = [[a_1(k,m,n), 1], [b(k+1), 0]] where a_1 = g - gbar(k+1)
        a_1 = expand(g_3d - gbar_3d.subs(k, k + 1))

        # K2(k,m,n) = [[gbar, 1], [b(k), g]]
        # K3: need new direction. Try K3 based on n-shift.
        # For K3(k,m,n): flatness requires
        #   K1(k,m,n) * K3(k+1,m,n) = K3(k,m,n) * K1(k,m,n+1)
        #   K2(k,m,n) * K3(k,m+1,n) = K3(k,m,n) * K2(k,m,n+1)

        # For the z-linear extension, K3 has the same telescope structure
        # but with n playing the role m played in K2.
        g_3d_n = f_3d.subs([(x, k), (y, n_sym), (z, m)])  # swap roles
        gbar_3d_n = fbar_3d.subs([(x, k), (y, n_sym), (z, m)])

        # Actually, K3 should use the same f but in a different "direction"
        # The standard approach: K3(k,m,n) = [[gbar_n(k,m,n), 1], [b(k), g_n(k,m,n)]]
        # where g_n uses n-dependence

        # Simple attempt: K3 = K2 with m→n (degenerate check first)
        K1 = sp.Matrix([[a_1, 1], [b_3d.subs(k, k + 1), 0]])
        K2 = sp.Matrix([[gbar_3d, 1], [b_3d, g_3d]])

        # K3 with n-direction: same structure but shift in n instead of m
        # g_for_K3 = f_3d(k, m, n), but K3 uses n-shift
        gbar_for_k3 = fbar_3d.subs([(x, k), (y, m), (z, n_sym)])
        g_for_k3 = f_3d.subs([(x, k), (y, m), (z, n_sym)])
        b_k3 = expand(g_for_k3.subs([(m, 0), (n_sym, 0)]) * gbar_for_k3.subs([(m, 0), (n_sym, 0)]))

        K3 = sp.Matrix([[gbar_for_k3, 1], [b_k3, g_for_k3]])

        # Check flatness condition 1: K1*K3(k+1) = K3*K1(n+1)
        try:
            LHS1 = K1 * K3.subs(k, k + 1)
            RHS1 = K3 * K1.subs(n_sym, n_sym + 1)
            diff1 = sp.expand(LHS1 - RHS1)

            if diff1 == sp.zeros(2, 2):
                # Check condition 2: K2*K3(m+1) = K3*K2(n+1)
                LHS2 = K2 * K3.subs(m, m + 1)
                RHS2 = K3 * K2.subs(n_sym, n_sym + 1)
                diff2 = sp.expand(LHS2 - RHS2)

                if diff2 == sp.zeros(2, 2):
                    results.append({
                        "delta": delta,
                        "f_3d": str(f_3d),
                        "fbar_3d": str(fbar_3d),
                        "a_3d": str(a_1),
                    })
        except Exception:
            continue

    return results


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5: Fill PCF gaps with Apéry-like structures
# ══════════════════════════════════════════════════════════════════════════
def generate_apery_pcfs(deg_a: int, deg_b: int, n_each: int = 30) -> list[dict]:
    """Generate PCFs with Apéry-like structure for given degrees.

    Uses structured templates that are known to converge well:
    - a(n) = (2n+1)*P(n), b(n) = -n^{deg_b}*Q(n)  (Apéry-type)
    - a(n) = combinatorial sums, b(n) = factored products
    """
    import random
    random.seed(100 * deg_a + deg_b)
    n = sp.Symbol("n")
    candidates = []

    for i in range(n_each):
        # Template 1: Apéry-like a(n) = (2n+1)*R(n), b(n) = -n^s * S(n)
        if deg_a >= 1:
            # a(n) has factor (2n+1), remaining degree = deg_a - 1
            rem_a = deg_a - 1
            r_coeffs = [random.randint(-3, 3) for _ in range(rem_a + 1)]
            if all(c == 0 for c in r_coeffs):
                r_coeffs[0] = 1
            R = sum(c * n**j for j, c in enumerate(r_coeffs))
            a_expr = (2 * n + 1) * R

            # b(n) = -(n+c)^s or -n*(n-1)*...
            if deg_b >= 2:
                shift = random.randint(0, 2)
                base = n + shift
                power = min(deg_b, 3)
                remaining = deg_b - power
                S = base**power
                if remaining > 0:
                    extra = sum(random.randint(-2, 2) * n**j for j in range(remaining + 1))
                    if extra != 0:
                        S = S * (1 + extra) if random.random() > 0.5 else S + extra
                b_expr = -S
            elif deg_b == 1:
                b_expr = -(random.randint(1, 5) * n + random.randint(-3, 3))
            else:
                b_expr = sp.Integer(-random.randint(1, 5))
        else:
            # deg_a == 0: constant a
            a_expr = sp.Integer(random.randint(1, 10))
            b_expr = -sp.Integer(random.randint(1, 5))

        a_poly = sp.Poly(sp.expand(a_expr), n)
        b_poly = sp.Poly(sp.expand(b_expr), n)

        a_coeffs = [int(c) for c in reversed(a_poly.all_coeffs())]
        b_coeffs = [int(c) for c in reversed(b_poly.all_coeffs())]

        # Pad to exact degree
        while len(a_coeffs) < deg_a + 1:
            a_coeffs.append(0)
        while len(b_coeffs) < deg_b + 1:
            b_coeffs.append(0)

        # Truncate to exact degree
        a_coeffs = a_coeffs[:deg_a + 1]
        b_coeffs = b_coeffs[:deg_b + 1]

        # Ensure leading coefficients nonzero
        if a_coeffs[-1] == 0:
            a_coeffs[-1] = random.choice([-1, 1, 2])
        if b_coeffs[-1] == 0:
            b_coeffs[-1] = random.choice([-1, -2])

        candidates.append({
            "a_coeffs": a_coeffs,
            "b_coeffs": b_coeffs,
            "gap": (deg_a, deg_b),
            "provenance": f"apery_pcf(deg_a={deg_a},deg_b={deg_b},idx={i})",
        })

    return candidates


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    engine = get_engine(DB_PATH)
    init_db(engine)
    session = get_session(engine)

    gaps = build_gap_atlas(session, max_density=2)
    dfinite_gaps = gaps[gaps["group"] == "dfinite"]
    pcf_gaps = gaps[gaps["group"] == "pcf"]
    log.info(f"Gaps: {len(dfinite_gaps)} dfinite, {len(pcf_gaps)} pcf")

    all_results = {
        "telescope_cmfs": [],
        "truly_2d": [],
        "cmf_3d": [],
        "pcf_converging": [],
        "pcf_recognized": [],
    }

    # ═══════════════════════════════════════════════════════════════════
    #  DFINITE GAPS: Telescope construction
    # ═══════════════════════════════════════════════════════════════════
    n_telescope = 0
    n_truly_2d = 0
    n_3d = 0
    n_flat_verified = 0

    for _, gap in dfinite_gaps.iterrows():
        total_deg = int(gap["x_coord"])
        max_var_deg = int(gap["y_coord"])

        if total_deg < 1 or max_var_deg < 1:
            continue

        polys = generate_polynomials(total_deg, max_var_deg, coeff_range=3)
        log.info(f"  Gap ({total_deg},{max_var_deg}): {len(polys)} polynomials")

        for f_expr in polys:
            for conj_name in CONJUGACIES:
                tel = build_telescope(f_expr, conj_name)
                if tel is None:
                    continue

                # Quick numeric flatness check
                flat_ok, residual = verify_flatness_numeric(tel, n_test=5)
                if not flat_ok:
                    continue  # Should not happen for telescope, but safety check

                n_flat_verified += 1

                # Check if truly 2D
                is_2d = check_truly_2d(tel)

                # Evaluate limit at m=0 and m=1
                eval_m0 = evaluate_telescope_limit(tel, m_val=0, N=300, dps=40)
                eval_m1 = evaluate_telescope_limit(tel, m_val=1, N=300, dps=40)

                result = {
                    "f_poly": tel["f_poly"],
                    "fbar_poly": tel["fbar_poly"],
                    "conjugacy": conj_name,
                    "a_km": str(tel["a_km"]),
                    "b_k": str(tel["b_k"]),
                    "gap": (total_deg, max_var_deg),
                    "total_degree": total_deg,
                    "flatness_residual": residual,
                    "truly_2d": is_2d,
                    "limit_m0": eval_m0.get("limit_float"),
                    "limit_m1": eval_m1.get("limit_float"),
                    "conv_m0": eval_m0.get("conv_score", 0),
                    "conv_m1": eval_m1.get("conv_score", 0),
                }

                all_results["telescope_cmfs"].append(result)
                n_telescope += 1

                if is_2d:
                    n_truly_2d += 1
                    all_results["truly_2d"].append(result)

                    # Attempt 3D extension
                    ext_3d = attempt_3d_extension(tel)
                    if ext_3d:
                        n_3d += len(ext_3d)
                        for e in ext_3d:
                            e["parent_2d"] = result
                            all_results["cmf_3d"].append(e)

    log.info(f"Telescope: {n_telescope} flat, {n_truly_2d} truly-2D, {n_3d} 3D extensions")

    # ═══════════════════════════════════════════════════════════════════
    #  PSLQ recognition on truly-2D CMFs
    # ═══════════════════════════════════════════════════════════════════
    n_pslq_2d = 0
    for result in all_results["truly_2d"]:
        if result.get("conv_m0", 0) < 0.2:
            continue
        limit_str = result.get("limit_m0")
        if limit_str is None:
            continue

        try:
            # Check if it's a simple rational
            limit_f = float(limit_str)
            # Skip obvious rationals
            from fractions import Fraction
            frac = Fraction(limit_f).limit_denominator(1000)
            if abs(limit_f - float(frac)) < 1e-10:
                result["recognized_m0"] = f"{frac.numerator}/{frac.denominator}"
                n_pslq_2d += 1
                continue

            # High-precision re-evaluation for PSLQ
            hp_eval = evaluate_telescope_limit(
                build_telescope(
                    sp.sympify(result["f_poly"]),
                    result["conjugacy"]
                ),
                m_val=0, N=1000, dps=120,
            )
            if hp_eval.get("conv_score", 0) > 0.3 and hp_eval.get("limit"):
                rec = run_pslq(str(hp_eval["limit"]), dps=100)
                if rec.get("success"):
                    result["recognized_m0"] = rec["identified_as"]
                    n_pslq_2d += 1
        except Exception:
            pass

    log.info(f"PSLQ on 2D: {n_pslq_2d} recognized")

    # ═══════════════════════════════════════════════════════════════════
    #  PCF GAPS: Apéry-like generation + evaluation
    # ═══════════════════════════════════════════════════════════════════
    n_pcf_converging = 0
    n_pcf_recognized = 0

    for _, gap in pcf_gaps.iterrows():
        deg_a = int(gap["x_coord"])
        deg_b = int(gap["y_coord"])

        cands = generate_apery_pcfs(deg_a, deg_b, n_each=40)

        for c in cands:
            a, b = c["a_coeffs"], c["b_coeffs"]
            if all(v == 0 for v in a) or all(v == 0 for v in b):
                continue

            # Quick eval
            mpmath.mp.dps = 50
            try:
                def a_fn(nn):
                    return sum(mpmath.mpf(coeff) * mpmath.power(nn, i) for i, coeff in enumerate(a))
                def b_fn(nn):
                    return sum(mpmath.mpf(coeff) * mpmath.power(nn, i) for i, coeff in enumerate(b))

                p_prev, p_curr = mpmath.mpf(1), a_fn(0)
                q_prev, q_curr = mpmath.mpf(0), mpmath.mpf(1)

                estimates = []
                for nn in range(1, 301):
                    an, bn = a_fn(nn), b_fn(nn)
                    p_new = an * p_curr + bn * p_prev
                    q_new = an * q_curr + bn * q_prev
                    p_prev, p_curr = p_curr, p_new
                    q_prev, q_curr = q_curr, q_new
                    if q_curr != 0 and nn % 20 == 0:
                        estimates.append(float(p_curr / q_curr))

                if not estimates or len(estimates) < 3:
                    continue

                limit = estimates[-1]
                tail = estimates[-5:]
                error = max(abs(tail[i] - tail[i - 1]) for i in range(1, len(tail)))

                if error > 0 and error < 1e-5:
                    stable_digits = -math.log10(max(error, 1e-50))
                    conv = min(1.0, stable_digits / 40)
                else:
                    conv = 0.0

                if conv < 0.2:
                    continue

                n_pcf_converging += 1
                c["limit"] = limit
                c["conv_score"] = conv
                c["error"] = error

                all_results["pcf_converging"].append(c)

                # PSLQ on best ones
                if conv > 0.4:
                    try:
                        # High-precision
                        mpmath.mp.dps = 150
                        p_prev, p_curr = mpmath.mpf(1), a_fn(0)
                        q_prev, q_curr = mpmath.mpf(0), mpmath.mpf(1)
                        for nn in range(1, 1001):
                            an, bn = a_fn(nn), b_fn(nn)
                            p_new = an * p_curr + bn * p_prev
                            q_new = an * q_curr + bn * q_prev
                            p_prev, p_curr = p_curr, p_new
                            q_prev, q_curr = q_curr, q_new

                        if q_curr != 0:
                            hp_limit = mpmath.nstr(p_curr / q_curr, 100)
                            rec = run_pslq(hp_limit, dps=100)
                            if rec.get("success"):
                                c["recognized"] = rec["identified_as"]
                                n_pcf_recognized += 1
                                all_results["pcf_recognized"].append(c)
                    except Exception:
                        pass

            except Exception:
                continue

    log.info(f"PCF gaps: {n_pcf_converging} converging, {n_pcf_recognized} recognized")

    # ═══════════════════════════════════════════════════════════════════
    #  INGEST INTO DB
    # ═══════════════════════════════════════════════════════════════════
    log.info("Ingesting results into DB...")
    project = session.query(Project).filter_by(name="Smart Gap Fill").first()
    if not project:
        project = Project(name="Smart Gap Fill")
        session.add(project)
        session.flush()

    n_stored = 0
    for tel in all_results["telescope_cmfs"]:
        try:
            payload = {
                "operator": [],
                "source_type": "telescope",
                "f_poly": tel["f_poly"],
                "fbar_poly": tel["fbar_poly"],
                "conjugacy": tel["conjugacy"],
                "total_degree": tel["total_degree"],
                "dimension": 2 if tel["truly_2d"] else 1,
            }
            fp, canonical_json = canonicalize_and_fingerprint("dfinite", payload)

            existing = session.query(Representation).filter_by(
                primary_group="dfinite", canonical_fingerprint=fp
            ).first()
            if existing:
                continue

            series = Series(
                project_id=project.id,
                name=f"tel_{tel['gap'][0]}_{tel['gap'][1]}_{tel['conjugacy']}",
                definition=tel["f_poly"],
                generator_type="gap_fill_telescope",
                provenance=f"gap_fill:telescope:{tel['gap']}:{tel['conjugacy']}",
            )
            session.add(series)
            session.flush()

            rep = Representation(
                series_id=series.id,
                primary_group="dfinite",
                canonical_fingerprint=fp,
                canonical_payload=canonical_json,
            )
            session.add(rep)
            session.flush()

            feat = compute_features("dfinite", canonical_json)
            feat["conv_score"] = tel.get("conv_m0", 0)
            feat_obj = FeaturesModel(
                representation_id=rep.id,
                feature_json=dumps(feat),
                feature_version="1.0.0",
            )
            session.add(feat_obj)

            cmf = CMF(
                representation_id=rep.id,
                cmf_payload=dumps(tel),
                dimension=2 if tel["truly_2d"] else 1,
            )
            session.add(cmf)
            n_stored += 1

        except Exception:
            session.rollback()
            project = session.query(Project).filter_by(name="Smart Gap Fill").first()
            continue

    for c in all_results["pcf_converging"]:
        try:
            payload = {"a_coeffs": c["a_coeffs"], "b_coeffs": c["b_coeffs"]}
            fp, canonical_json = canonicalize_and_fingerprint("pcf", payload)

            existing = session.query(Representation).filter_by(
                primary_group="pcf", canonical_fingerprint=fp
            ).first()
            if existing:
                continue

            series = Series(
                project_id=project.id,
                name=f"pcf_{c['gap'][0]}_{c['gap'][1]}",
                definition=c["provenance"],
                generator_type="gap_fill_pcf",
                provenance=f"gap_fill:pcf:{c['gap']}",
            )
            session.add(series)
            session.flush()

            rep = Representation(
                series_id=series.id,
                primary_group="pcf",
                canonical_fingerprint=fp,
                canonical_payload=canonical_json,
            )
            session.add(rep)
            session.flush()

            feat = compute_features("pcf", canonical_json)
            feat["conv_score"] = c.get("conv_score", 0)
            if c.get("recognized"):
                feat["recognized"] = 1
            feat_obj = FeaturesModel(
                representation_id=rep.id,
                feature_json=dumps(feat),
                feature_version="1.0.0",
            )
            session.add(feat_obj)

            cmf = CMF(
                representation_id=rep.id,
                cmf_payload=dumps(c),
                dimension=1,
            )
            session.add(cmf)
            n_stored += 1

        except Exception:
            session.rollback()
            project = session.query(Project).filter_by(name="Smart Gap Fill").first()
            continue

    session.commit()
    log.info(f"Stored {n_stored} items in DB")

    # ═══════════════════════════════════════════════════════════════════
    #  SAVE & PRINT RESULTS
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(elapsed, 1),
        "dfinite_gaps_processed": len(dfinite_gaps),
        "pcf_gaps_processed": len(pcf_gaps),
        "telescope_cmfs_flat": n_telescope,
        "telescope_flat_verified": n_flat_verified,
        "truly_2d_cmfs": n_truly_2d,
        "cmf_3d_extensions": n_3d,
        "pslq_2d_recognized": n_pslq_2d,
        "pcf_converging": n_pcf_converging,
        "pcf_recognized": n_pcf_recognized,
        "stored_in_db": n_stored,
        "truly_2d_details": [
            {
                "f_poly": r["f_poly"],
                "fbar_poly": r["fbar_poly"],
                "conjugacy": r["conjugacy"],
                "a_km": r["a_km"],
                "gap": r["gap"],
                "limit_m0": r.get("limit_m0"),
                "limit_m1": r.get("limit_m1"),
                "recognized_m0": r.get("recognized_m0"),
            }
            for r in all_results["truly_2d"]
        ],
        "cmf_3d_details": all_results["cmf_3d"],
        "top_pcf": [
            {
                "gap": c["gap"],
                "a_coeffs": c["a_coeffs"],
                "b_coeffs": c["b_coeffs"],
                "limit": c.get("limit"),
                "conv_score": c.get("conv_score"),
                "recognized": c.get("recognized"),
            }
            for c in sorted(
                all_results["pcf_converging"],
                key=lambda c: -c.get("conv_score", 0)
            )[:30]
        ],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  SMART GAP-FILLING — RESULTS")
    print("=" * 70)
    print(f"  D-finite gaps processed:    {len(dfinite_gaps)}")
    print(f"  PCF gaps processed:         {len(pcf_gaps)}")
    print(f"  Telescope CMFs (flat):      {n_telescope}")
    print(f"  Truly 2D CMFs:              {n_truly_2d}")
    print(f"  3D extensions found:        {n_3d}")
    print(f"  2D limits recognized:       {n_pslq_2d}")
    print(f"  PCF converging:             {n_pcf_converging}")
    print(f"  PCF recognized (PSLQ):      {n_pcf_recognized}")
    print(f"  Stored in DB:               {n_stored}")
    print(f"  Runtime:                    {elapsed:.1f}s")
    print("=" * 70)

    if all_results["truly_2d"]:
        print(f"\n  TRULY 2D CMFs ({n_truly_2d}):")
        for r in all_results["truly_2d"][:20]:
            print(f"    f={r['f_poly']}, conj={r['conjugacy']}, gap={r['gap']}")
            print(f"      a(k,m) = {r['a_km']}")
            lim0 = r.get("limit_m0")
            lim1 = r.get("limit_m1")
            rec = r.get("recognized_m0", "?")
            print(f"      limit(m=0) = {lim0:.10g if lim0 else '?'}, limit(m=1) = {lim1:.10g if lim1 else '?'}")
            if rec and rec != "?":
                print(f"      → Recognized: {rec}")

    if all_results["cmf_3d"]:
        print(f"\n  3D EXTENSIONS ({n_3d}):")
        for e in all_results["cmf_3d"][:10]:
            print(f"    f_3d={e['f_3d']}, delta={e['delta']}")

    if all_results["pcf_recognized"]:
        print(f"\n  RECOGNIZED PCFs ({n_pcf_recognized}):")
        for c in all_results["pcf_recognized"][:20]:
            print(f"    gap={c['gap']} a={c['a_coeffs']} b={c['b_coeffs']}")
            print(f"      → {c['recognized']} (conv={c.get('conv_score', 0):.2f})")

    session.close()
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
