#!/usr/bin/env python3
"""
cmf_deep_identify_v2.py
========================
Extended identification pass for all telescope/CMF-Hunter CMFs.

NEW vs v1:
  • Polygamma / digamma at rationals  ψ(p/q),  q ≤ 8
  • Hurwitz zeta  ζ(s, p/q)  for  s ∈ {2,3}  and  q ≤ 6
  • Dirichlet L-functions  L(χ, s)  for conductors 3,4,5,7,8  and  s ∈ {1,2,3}
  • Products / sums of the above
  • Analytic partial-fraction route:  derive the expected value directly from
    the roots of  f(k, m_val)  via  Σ_k 1/P(k) = −Σ_i ψ(1−rᵢ)/P'(rᵢ)
  • Writes updated `primary_constant` back to atlas.db cmf_payload JSON
  • Produces  walk_identify_results_v2.jsonl  and  walk_identify_report_v2.txt

Usage:
    python3 cmf_deep_identify_v2.py
    python3 cmf_deep_identify_v2.py --source telescope --depth 2000 --dps 80
    python3 cmf_deep_identify_v2.py --update-db   # write hits to atlas.db
"""
import argparse, json, math, sqlite3, time
from fractions import Fraction
from math import gcd
from pathlib import Path

import mpmath
import sympy as sp
from sympy import symbols as _sym, sympify as _sp, lambdify as _lam, expand as _exp, Poly

DB_PATH   = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "walk_identify_results_v2.jsonl"
OUT_TXT   = Path(__file__).parent / "walk_identify_report_v2.txt"

# ─────────────────────────────────────────────────────────────────────────────
#  Extended constant basis
# ─────────────────────────────────────────────────────────────────────────────

def _rationals_up_to(max_denom: int):
    """All reduced fractions p/q with 1 ≤ p < q ≤ max_denom."""
    seen = set()
    result = []
    for q in range(2, max_denom + 1):
        for p in range(1, q):
            if gcd(p, q) == 1:
                key = Fraction(p, q)
                if key not in seen:
                    seen.add(key)
                    result.append((p, q))
    return result


def build_extended_basis(dps: int = 80):
    """
    Build a (value, name) list covering:
      ψ(p/q)        — polygamma at rationals
      ζ(2, p/q)     — Hurwitz zeta s=2
      ζ(3, p/q)     — Hurwitz zeta s=3
      L(χ, s)       — Dirichlet L at conductors 3,4,5,7,8
      Products / mixtures of the above with π, ln2, γ, √n
    """
    mpmath.mp.dps = dps
    m = mpmath
    pi  = m.pi;   ln2 = m.log(2);  ln3 = m.log(3)
    z2  = m.zeta(2);  z3 = m.zeta(3);  z4 = m.zeta(4);  z5 = m.zeta(5)
    G   = m.catalan;  gamma_em = m.euler   # Euler-Mascheroni

    consts = []
    _add = consts.append

    # ── Fundamentals ──
    for v, n in [
        (1,        "1"),
        (pi,       "π"),
        (pi**2,    "π²"),
        (pi**3,    "π³"),
        (pi**4,    "π⁴"),
        (pi**5,    "π⁵"),
        (ln2,      "ln2"),
        (ln3,      "ln3"),
        (z2,       "ζ(2)"),
        (z3,       "ζ(3)"),
        (z4,       "ζ(4)"),
        (z5,       "ζ(5)"),
        (G,        "G"),
        (gamma_em, "γ"),
        (m.sqrt(2),"√2"),
        (m.sqrt(3),"√3"),
        (m.sqrt(5),"√5"),
        (m.log(m.sqrt(2)), "ln√2"),
    ]:
        _add((v, n))

    # ── ψ(p/q) for q ≤ 8 ──
    rats = _rationals_up_to(8)
    for p, q in rats:
        r = m.mpf(p) / q
        try:
            psi_r = m.digamma(r)
            _add((psi_r, f"ψ({p}/{q})"))
            # also 1 + integer shifts (these differ by 1/r, 1/(r+1), ...)
            psi_r1 = m.digamma(r + 1)
            _add((psi_r1, f"ψ(1+{p}/{q})"))
        except Exception:
            pass

    # ── Hurwitz ζ(s, p/q) for s=2,3 and q ≤ 6 ──
    for p, q in _rationals_up_to(6):
        r = m.mpf(p) / q
        for s in (2, 3):
            try:
                hz = m.zeta(s, r)
                _add((hz, f"ζ({s},{p}/{q})"))
            except Exception:
                pass

    # ── Dirichlet L-functions L(χ,s) for conductors 3,4,5,7,8 ──
    # Primitive characters; L-values computed via mpmath.altzeta / known formulae
    # conductor 4:  χ₄(1)=1, χ₄(3)=-1  (the Catalan character)
    # L(χ₄,1) = π/4   L(χ₄,3) = π³/32
    _add((pi/4,         "L(χ₄,1)"))
    _add((pi**3/32,     "L(χ₄,3)"))
    _add((G,            "L(χ₄,2)"))   # =Catalan

    # conductor 3:  χ₃(1)=1, χ₃(2)=-1  (Eisenstein)
    # L(χ₃,1) = π/(3√3)   L(χ₃,2) = π²/(9√3)·??  actually π²/9·1/... use mpmath
    try:
        L31 = m.mpf(0)
        for n in range(1, 20000):
            chi = 1 if n % 3 == 1 else (-1 if n % 3 == 2 else 0)
            L31 += m.mpf(chi) / n
        _add((L31, "L(χ₃,1)"))
    except Exception:
        pass
    try:
        L32 = m.nsum(lambda n: (1 if n%3==1 else (-1 if n%3==2 else 0)) / n**2, [1, m.inf])
        _add((L32, "L(χ₃,2)"))
        L33 = m.nsum(lambda n: (1 if n%3==1 else (-1 if n%3==2 else 0)) / n**3, [1, m.inf])
        _add((L33, "L(χ₃,3)"))
    except Exception:
        pass

    # conductor 5
    try:
        L51 = m.nsum(lambda n: [0,1,-1,-1,1][int(n)%5] / n,   [1, m.inf])
        L52 = m.nsum(lambda n: [0,1,-1,-1,1][int(n)%5] / n**2,[1, m.inf])
        L53 = m.nsum(lambda n: [0,1,-1,-1,1][int(n)%5] / n**3,[1, m.inf])
        _add((L51, "L(χ₅,1)")); _add((L52, "L(χ₅,2)")); _add((L53, "L(χ₅,3)"))
    except Exception:
        pass

    # ── Products: ψ × π, ψ × ln, etc. ──
    # Grab all digamma values added so far
    psi_vals = [(v, n) for v, n in consts if n.startswith("ψ(")]
    for psi_v, psi_n in psi_vals[:20]:   # first 20 to limit size
        _add((psi_v * pi,  f"{psi_n}·π"))
        _add((psi_v * ln2, f"{psi_n}·ln2"))
        _add((psi_v**2,    f"{psi_n}²"))

    # ── Mixed products of basic constants ──
    for v1, n1 in [(z3, "ζ(3)"), (z2, "ζ(2)"), (pi, "π"), (G, "G"), (ln2, "ln2")]:
        for v2, n2 in [(pi, "π"), (ln2, "ln2"), (z2, "ζ(2)"), (G, "G")]:
            if n1 != n2:
                _add((v1 * v2, f"{n1}·{n2}"))

    # ── Critical product-constants (PSLQ needs these as atoms) ──
    sq3 = m.sqrt(3); sq2 = m.sqrt(2); sq5 = m.sqrt(5)
    for sv, sn in [(sq3, "√3"), (sq2, "√2"), (sq5, "√5")]:
        _add((pi / sv,       f"π/{sn}"))
        _add((pi * sv,       f"π·{sn}"))
        _add((ln2 / sv,      f"ln2/{sn}"))
        _add((ln3 / sv,      f"ln3/{sn}"))
        _add((gamma_em / sv, f"γ/{sn}"))
    _add((gamma_em * pi,     "γ·π"))
    _add((gamma_em * ln2,    "γ·ln2"))
    _add((gamma_em * ln3,    "γ·ln3"))
    _add((gamma_em**2,       "γ²"))
    _add((pi**2 / sq3,       "π²/√3"))
    _add((pi**2 * sq3,       "π²·√3"))
    _add((z3 / sq3,          "ζ(3)/√3"))
    _add((z3 * sq3,          "ζ(3)·√3"))
    _add((z3 / pi,           "ζ(3)/π"))
    _add((G / sq3,           "G/√3"))
    _add((G * sq3,           "G·√3"))
    _add((ln2**2,            "ln2²"))
    _add((ln3**2,            "ln3²"))
    _add((ln2 * ln3,         "ln2·ln3"))
    _add((m.log(2) * m.log(3), "ln2·ln3"))  # deduplicated via pslq anyway

    # ── Clausen Cl₂(π/3) and related log-sine at rational angles ──
    try:
        # Cl₂(θ) = -∫₀^θ ln|2sin(t/2)| dt  = Im Li₂(e^{iθ})
        cl2_pi3  = m.clsin(2, m.pi/3)   # Clausen at π/3 ≈ 1.01494
        cl2_pi2  = m.clsin(2, m.pi/2)   # = G (Catalan)
        cl2_2pi3 = m.clsin(2, 2*m.pi/3)
        _add((cl2_pi3,  "Cl₂(π/3)"))
        _add((cl2_2pi3, "Cl₂(2π/3)"))
    except Exception:
        pass

    # ── ln(Γ(p/q)) — Chowla-Selberg type ──
    for p, q in _rationals_up_to(6)[:15]:
        try:
            lgv = m.log(m.gamma(m.mpf(p)/q))
            _add((lgv, f"ln Γ({p}/{q})"))
        except Exception:
            pass

    return consts


# ─────────────────────────────────────────────────────────────────────────────
#  Partial-fraction analytic route
# ─────────────────────────────────────────────────────────────────────────────

def analytic_euler_sum(f_poly_str: str, m_val: int = 0, k_start: int = 1,
                       dps: int = 80) -> tuple:
    """
    Given P(k) = f_poly_str evaluated at m=m_val, compute
        S = Σ_{k=k_start}^∞  1 / P(k)
    analytically via partial fractions and digamma:
        1/P(k) = Σ_i  Aᵢ / (k − rᵢ)
        Σ_{k=k_start}^∞ 1/(k−r) = ψ(k_start − r) − ψ(k_start) ... → ψ(1) − ψ(k_start − r)
    Returns (value_mpmath, formula_str) or (None, None) on failure.
    """
    mpmath.mp.dps = dps
    k = sp.Symbol('k')
    m_sym = sp.Symbol('m')

    try:
        f_expr = _sp(f_poly_str).subs(m_sym, m_val)
        poly = Poly(f_expr, k)
        if poly.degree() < 1:
            return None, None

        # Get roots (numerical, via mpmath for reliability)
        coeffs_sp  = poly.all_coeffs()
        coeffs_mp  = [complex(float(sp.re(c)), float(sp.im(c))) for c in coeffs_sp]
        roots_mp   = mpmath.polyroots(coeffs_mp)

        lead = complex(float(sp.re(coeffs_sp[0])), float(sp.im(coeffs_sp[0])))
        if abs(lead) < 1e-300:
            return None, None

        # Partial fraction coefficients:  Aᵢ = 1 / (lead * Π_{j≠i} (rᵢ − rⱼ))
        total = mpmath.mpf(0)
        terms = []
        for i, ri in enumerate(roots_mp):
            denom = mpmath.mpf(1)
            for j, rj in enumerate(roots_mp):
                if j != i:
                    denom *= (ri - rj)
            Ai = 1 / (mpmath.mpf(lead) * denom)

            # Σ_{k=k_start}^∞ Ai/(k−ri) = −Ai * [ψ(k_start − ri)]  + finite terms
            # More precisely: Σ_{k=k0}^∞ 1/(k+a) = ψ(∞) − ψ(k0+a) → diverges
            # But for finite sum from the partial fraction:
            # Σ_{k=k0}^∞ Ai/(k−ri)  converges only if the Ai sum to 0 (which they must for deg≥2)
            # Use: Σ_{k=k0}^∞ Ai/(k−ri) = −Ai * ψ(k0−ri)  [polygamma regularisation]
            # This is valid when Σ Ai = 0 (ensured by deg P ≥ 2).
            try:
                shift = mpmath.mpf(k_start) - ri
                psi_val = mpmath.digamma(shift)
                contrib = -Ai * psi_val
                total   += contrib
                terms.append(f"−({Ai:.4g}·ψ({k_start}−{ri:.4g}))")
            except Exception:
                return None, None

        formula = "Σ " + " + ".join(terms)
        return total, formula

    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Walk machinery (mirrors v1 / api.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_fns(fp: str, fb: str):
    k, m, x, y = _sym("k m x y")
    fe = _sp(fp); fb_e = _sp(fb)
    is3d = 'z' in str(fe.free_symbols)

    if is3d:
        _fr = fe.free_symbols
        xs = next(s for s in _fr if s.name == 'x')
        ys = next(s for s in _fr if s.name == 'y')
        zs = next(s for s in _fr if s.name == 'z')
        gkm  = fe.subs([(xs, k), (ys, m)])
        gbkm = fb_e.subs([(xs, k), (ys, m)])
        be   = _exp(gkm.subs(m, 0) * gbkm.subs(m, 0))
        ae   = _exp(gkm - gbkm.subs(k, k + 1))
        ev   = lambda e, kv, mv, nv: complex(e.subs([(k, kv), (m, mv), (zs, nv)]))
        evb  = lambda e, kv, nv:    complex(e.subs([(k, kv), (zs, nv)]))
        Kx = lambda kv, mv, nv: mpmath.matrix([[0, 1], [evb(be, kv+1, nv), ev(ae, kv, mv, nv)]])
        Ky = lambda kv, mv, nv: mpmath.matrix([[ev(gbkm, kv, mv, nv), 1], [evb(be, kv, nv), ev(gkm, kv, mv, nv)]])
        return Kx, Ky, Ky, True

    gkm  = fe.subs([(x, k), (y, m)])
    gbkm = fb_e.subs([(x, k), (y, m)])
    be   = _exp(gkm.subs(m, 0) * gbkm.subs(m, 0))
    ae   = _exp(gkm - gbkm.subs(k, k + 1))
    gf   = _lam([k, m], gkm,  modules="mpmath")
    gbf  = _lam([k, m], gbkm, modules="mpmath")
    bf   = _lam([k],    be,   modules="mpmath")
    af   = _lam([k, m], ae,   modules="mpmath")
    Kx = lambda kv, mv, nv=0: mpmath.matrix([[0, 1], [bf(kv+1), af(kv, mv)]])
    Ky = lambda kv, mv, nv=0: mpmath.matrix([[gbf(kv, mv), 1], [bf(kv), gf(kv, mv)]])
    return Kx, Ky, None, False


def delta_from_walk(Kfn, k0, depth):
    m1, m2, m3 = depth//3, 2*depth//3, depth
    P = mpmath.eye(2); ests = {}
    for it in range(1, depth + 1):
        try:
            P = P * Kfn(k0 + it - 1)
            if it % 20 == 0:
                sc = max(abs(float(mpmath.re(P[0,0]))), abs(float(mpmath.re(P[0,1]))), 1e-300)
                P  = P / sc
            if it in (m1, m2, m3):
                d = P[1,1]; n = P[0,1]
                ests[it] = mpmath.re(n/d) if mpmath.fabs(d) > 1e-200 else None
        except Exception:
            if it in (m1, m2, m3): ests[it] = None
    e1, e2, e3 = ests.get(m1), ests.get(m2), ests.get(m3)
    if None in (e1, e2, e3): return None, ests.get(m3)
    d12 = abs(float(e2 - e1)); d23 = abs(float(e3 - e2))
    if d12 == 0: return 0.0, e3
    if d23 == 0: return 1e6, e3
    return (math.log2(d12) - math.log2(d23)) / (m2 - m1), e3


def best_trajectory(is3d, Kx, Ky, Kz, depth_scan=200):
    candidates = []
    for mv in [0, 1, 2, 3, 5]:
        candidates.append((f"Kx_m{mv}", lambda s, mv=mv: Kx(s, mv, 0), 0))
    for kv in [1, 2, 3, 5]:
        candidates.append((f"Ky_k{kv}", lambda s, kv=kv: Ky(kv, s, 0), 1))
    if is3d and Kz:
        for kv in [1, 2]:
            for mv in [0, 1]:
                k0 = 1
                for s in range(1, 20):
                    try:
                        if abs(float(mpmath.re(Kz(kv, mv, s)[1,0]))) > 1e-10:
                            k0 = s; break
                    except Exception: pass
                candidates.append((f"Kz_k{kv}_m{mv}", lambda s, kv=kv, mv=mv: Kz(kv, mv, s), k0))

    best_d = None; best = None
    for lbl, Kfn, k0 in candidates:
        try:
            d, _ = delta_from_walk(Kfn, k0, depth_scan)
            if d is not None and (best_d is None or d > best_d):
                best_d = d; best = (lbl, Kfn, k0)
        except Exception: pass
    return best, best_d


# ─────────────────────────────────────────────────────────────────────────────
#  PSLQ identification with extended basis
# ─────────────────────────────────────────────────────────────────────────────

def pslq_identify_extended(val, basis: list, tol_digits: int = 10) -> list:
    """
    Try to express `val` as an integer-linear combination of basis constants
    divided by an integer coefficient.
    Returns list of dicts {method, expr, rel_err, digits}.
    """
    if val is None: return []
    fv = float(mpmath.re(val))
    if not math.isfinite(fv) or fv == 0: return []

    hits = []

    # ── 1. Direct float match  (sign × integer/small-denom multiple) ──
    for bv, bn in basis:
        try:
            bvf = float(mpmath.re(bv))
        except Exception:
            continue
        if bvf == 0: continue
        for num, den in [(1,1),(1,2),(1,3),(1,4),(2,1),(3,1),(4,1),(1,6),(5,1)]:
            for sign in (1, -1):
                cand = sign * (num / den) * bvf
                if abs(fv) < 1e-200: break
                rel = abs(fv - cand) / abs(fv)
                if rel < 1e-10:
                    hits.append({"method": "float",
                                 "expr": f"{'−' if sign<0 else ''}{'' if num==1 and den==1 else f'({num}/{den})·'}{bn}",
                                 "rel_err": rel, "digits": int(-math.log10(rel+1e-300))})
        if hits:
            hits.sort(key=lambda x: x["rel_err"])
            return hits[:3]

    # ── 2. PSLQ pass 1: [1, val] + first 30 basis ──
    for n_basis in [30, 60, min(120, len(basis))]:
        sub = basis[:n_basis]
        try:
            vec = [mpmath.mpf(1), mpmath.re(mpmath.mpf(fv))] + \
                  [mpmath.re(mpmath.mpf(float(v))) for v, _ in sub]
            rel = mpmath.pslq(vec, maxcoeff=200, tol=mpmath.mpf(10)**(-tol_digits+2),
                               maxsteps=2000)
            if rel is not None and rel[1] != 0:
                c0 = rel[0]; c1 = rel[1]
                parts = []
                if c0 != 0: parts.append(str(c0))
                for co, (_, bn) in zip(rel[2:], sub):
                    if co != 0: parts.append(f"{co}·{bn}")
                numer = " + ".join(parts) if parts else "0"
                expr = f"({numer}) / {-c1}"
                # verify
                approx = (sum(co * float(mpmath.re(v))
                              for co, (v, _) in zip(rel[2:], sub))
                          + c0) / (-c1)
                rel_err = abs(fv - approx) / (abs(fv) + 1e-300)
                digits = int(-math.log10(rel_err + 1e-300))
                if digits >= tol_digits - 2:
                    hits.append({"method": "pslq", "expr": expr,
                                 "rel_err": rel_err, "digits": digits})
                    break
        except Exception:
            pass

    # ── 3. mpmath.identify (ISC) ──
    try:
        s = mpmath.identify(mpmath.mpf(fv), tol=mpmath.mpf(10)**(-tol_digits))
        if s and '**' not in s:   # skip noisy power expressions
            hits.append({"method": "isc", "expr": s, "rel_err": 0.0, "digits": tol_digits})
    except Exception:
        pass

    hits.sort(key=lambda x: (-x.get("digits", 0), x["rel_err"]))
    return hits[:5]


# ─────────────────────────────────────────────────────────────────────────────
#  Atlas DB update
# ─────────────────────────────────────────────────────────────────────────────

def update_atlas_db(cmf_id: int, identified_expr: str, method: str, digits: int):
    """Write the identified constant back into cmf_payload in atlas.db."""
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT cmf_payload FROM cmf WHERE id=?", (cmf_id,)).fetchone()
        if row is None:
            con.close(); return False
        payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        payload["identified_constant"]        = identified_expr
        payload["identification_method"]      = method
        payload["identification_digits"]      = digits
        payload["identification_updated_at"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        con.execute("UPDATE cmf SET cmf_payload=? WHERE id=?",
                    (json.dumps(payload), cmf_id))
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"    DB update error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",      default="all", choices=["all","telescope","cmf_hunter"])
    ap.add_argument("--depth",       type=int, default=1500)
    ap.add_argument("--dps",         type=int, default=80)
    ap.add_argument("--scan-depth",  type=int, default=200, dest="scan_depth")
    ap.add_argument("--tol-digits",  type=int, default=10,  dest="tol_digits")
    ap.add_argument("--out",         default=str(OUT_JSONL))
    ap.add_argument("--update-db",   action="store_true", dest="update_db",
                    help="Write confirmed identifications back to atlas.db")
    ap.add_argument("--min-digits",  type=int, default=8, dest="min_digits",
                    help="Minimum matching digits to count as a hit")
    args = ap.parse_args()

    mpmath.mp.dps = args.dps

    src_filter = {
        "all":        ("telescope", "cmf_hunter"),
        "telescope":  ("telescope",),
        "cmf_hunter": ("cmf_hunter",),
    }[args.source]

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT id, cmf_payload FROM cmf
        WHERE json_extract(cmf_payload,'$.source') IN ({','.join('?'*len(src_filter))})
        ORDER BY json_extract(cmf_payload,'$.source'), id
    """, src_filter).fetchall()
    conn.close()

    print("══════════════════════════════════════════════════════════════════")
    print(f"  CMF Deep Identify v2  —  {len(rows)} CMFs")
    print(f"  source={args.source}  depth={args.depth}  dps={args.dps}")
    print(f"  update_db={args.update_db}  min_digits={args.min_digits}")
    print("══════════════════════════════════════════════════════════════════")

    print("Building extended constant basis …", end="", flush=True)
    basis = build_extended_basis(args.dps)
    print(f" {len(basis)} constants ready.\n")

    out_path = Path(args.out)
    results = []
    t0 = time.time()
    n_identified = 0

    with open(out_path, "w") as fout:
        for ci, (cid, praw) in enumerate(rows):
            p      = json.loads(praw) if isinstance(praw, str) else praw
            fp     = p.get("f_poly", "");  fb = p.get("fbar_poly", "")
            src    = p.get("source", "");   cert = p.get("certification_level", "")
            pconst = p.get("primary_constant", None)
            elapsed = time.time() - t0

            print(f"[{ci+1:3d}/{len(rows)}] #{cid} {src:12s} [{elapsed:5.0f}s]", end="  ", flush=True)

            if not fp or not fb:
                print("skip"); continue

            try:
                Kx, Ky, Kz, is3d = build_fns(fp, fb)
            except Exception as e:
                print(f"BUILD ERR: {e}"); continue

            # ── Best trajectory ──
            best, scan_delta = best_trajectory(is3d, Kx, Ky, Kz, args.scan_depth)
            if best is None:
                print("no trajectory"); continue

            lbl, Kfn, k0 = best
            print(f"traj={lbl} Δ={scan_delta:.3f}", end="  ", flush=True)

            # ── Deep walk ──
            try:
                deep_delta, deep_est = delta_from_walk(Kfn, k0, args.depth)
                if deep_est is None:
                    print("degenerate"); continue
            except Exception as e:
                print(f"WALK ERR: {e}"); continue

            est_str = mpmath.nstr(deep_est, 40)
            print(f"est={est_str[:28]}", end="  ", flush=True)

            # ── Partial-fraction analytic route ──
            # Determine m_val from the trajectory label
            m_val = int(lbl.split("_m")[1]) if "_m" in lbl else 0
            pf_val, pf_formula = analytic_euler_sum(fp, m_val=m_val, k_start=k0+1,
                                                     dps=args.dps)

            pf_match = None
            if pf_val is not None:
                pf_val_f = float(mpmath.re(pf_val))
                est_f    = float(mpmath.re(deep_est))
                if abs(est_f) > 1e-200:
                    pf_rel = abs(pf_val_f - est_f) / abs(est_f)
                    pf_digits = int(-math.log10(pf_rel + 1e-300))
                    pf_match = {"pf_val": pf_val_f, "pf_rel_err": pf_rel,
                                "pf_digits": pf_digits, "pf_formula": pf_formula}
                    if pf_digits >= args.min_digits:
                        print(f"PF✓{pf_digits}d", end="  ", flush=True)

            # ── PSLQ / extended identification ──
            hits = pslq_identify_extended(deep_est, basis, tol_digits=args.tol_digits)

            # ── pconst sanity check ──
            pconst_match = None
            if pconst and str(pconst) not in ("None", ""):
                try:
                    pconst_val = float(mpmath.mpf(str(sp.sympify(str(pconst)).evalf(60))))
                    rel = abs(float(deep_est) - pconst_val) / max(abs(pconst_val), 1e-300)
                    pconst_match = {"matches": rel < 1e-6, "rel_err": rel}
                except Exception:
                    pconst_match = {"matches": False, "rel_err": None}

            # ── Report ──
            best_hit = hits[0] if hits else None
            if best_hit and best_hit.get("digits", 0) >= args.min_digits:
                n_identified += 1
                print(f"★ {best_hit['expr'][:50]}  [{best_hit['digits']}d]")
                if args.update_db:
                    ok = update_atlas_db(cid, best_hit["expr"],
                                         best_hit["method"], best_hit["digits"])
                    if ok: print(f"    ✓ DB updated #{cid}")
            elif pconst_match and pconst_match.get("matches"):
                n_identified += 1
                print(f"★ matches pconst={pconst}")
            elif pf_match and pf_match["pf_digits"] >= args.min_digits:
                # Partial-fraction gave a numerical match; now identify pf_val via PSLQ
                pf_hits = pslq_identify_extended(mpmath.mpf(pf_match["pf_val"]),
                                                  basis, tol_digits=args.tol_digits)
                if pf_hits and pf_hits[0].get("digits", 0) >= args.min_digits:
                    n_identified += 1
                    print(f"★(pf) {pf_hits[0]['expr'][:50]}  [{pf_hits[0]['digits']}d]")
                    hits = pf_hits
                    if args.update_db:
                        update_atlas_db(cid, pf_hits[0]["expr"],
                                        "pf_pslq", pf_hits[0]["digits"])
                else:
                    print(f"(pf match, unidentified  pf={pf_match['pf_val']:.8g})")
            else:
                print("(unidentified)")

            rec = {
                "cmf_id": cid, "source": src, "cert": cert,
                "f_poly": fp, "fbar_poly": fb, "is_3d": is3d,
                "primary_constant": str(pconst) if pconst else None,
                "best_traj": lbl, "scan_delta": round(float(scan_delta), 4) if scan_delta else None,
                "deep_delta": round(float(deep_delta), 4) if deep_delta is not None else None,
                "deep_depth": args.depth,
                "estimate": est_str,
                "identified": hits,
                "pconst_match": pconst_match,
                "partial_fraction": pf_match,
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            results.append(rec)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_t = time.time() - t0
    hits_strong = [r for r in results
                   if r.get("identified") and r["identified"][0].get("digits", 0) >= args.min_digits]
    unid = [r for r in results if not r.get("identified") or
            r["identified"][0].get("digits", 0) < args.min_digits]

    lines = [
        "CMF Deep Identification v2 — Report",
        "=====================================",
        f"Sources: {src_filter}  |  CMFs: {len(results)}  |  Identified: {n_identified}",
        f"Depth: {args.depth}  |  dps: {args.dps}  |  Elapsed: {total_t:.1f}s",
        f"Basis: {len(basis)} constants (polygamma+Hurwitz+L-fn+products)",
        "",
        f"HITS (≥{args.min_digits} digits):",
        "-" * 60,
    ]
    for r in sorted(hits_strong, key=lambda x: -(x["identified"][0].get("digits", 0))):
        h = r["identified"][0]
        lines.append(
            f"  #{r['cmf_id']:4d} ({r['source']:12s})  "
            f"Δ={r.get('deep_delta') or 0:6.4f}  "
            f"f={r['f_poly'][:35]:35s}  "
            f"→ {h['expr'][:60]}  [{h.get('digits','?')}d]"
        )

    lines += ["", "STRONG DELTA, UNIDENTIFIED:", "-" * 60]
    for r in sorted([r for r in unid if (r.get("deep_delta") or 0) >= 0.05],
                    key=lambda x: -(x.get("deep_delta") or 0)):
        lines.append(
            f"  #{r['cmf_id']:4d} ({r['source']:12s})  "
            f"Δ={r.get('deep_delta') or 0:6.4f}  "
            f"f={r['f_poly'][:35]:35s}  "
            f"est={r['estimate'][:40]}"
        )

    report = "\n".join(lines)
    print("\n" + report)
    OUT_TXT.write_text(report)
    print(f"\nJSONL → {out_path}")
    print(f"Report → {OUT_TXT}")


if __name__ == "__main__":
    main()
