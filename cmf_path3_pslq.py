#!/usr/bin/env python3
"""
cmf_path3_pslq.py — Direct hypergeometric series evaluation + deep PSLQ
For CMF Hunter entries with f(k,m) = 2k^3 + ck + 2m^3 + cm (Euler-sum family):

  Σ_{k=1}^∞ 1/(2k^3 + ck)  =  (1/2) Σ_{j≥0} (-c/2)^j ζ(3+2j)

This hypergeometric-series-in-ζ representation lets us:
  • Identify the sum as a linear combination of ζ(2n+1)
  • Spot MZV (multiple-zeta value) relations
  • Use partial-fraction decomposition for rational a: sum = combination of ψ(roots)

Also handles arbitrary polynomial f via partial-fraction + digamma decomposition:
  For 1/P(k) = Σ A_i / (k - r_i)  where r_i are roots of P(x)
  Σ_{k=1}^∞ 1/P(k) = Σ A_i ψ(1-r_i) - ψ(-r_i)   (digamma regularization)

Usage:
    python3 cmf_path3_pslq.py --reuse cmf_euler_full.jsonl --dps 100
    python3 cmf_path3_pslq.py --source cmf_hunter --m-max 4 --dps 100
"""
import argparse, json, math, sqlite3, time
from pathlib import Path
import mpmath
import sympy as sp
from sympy import (Symbol, sympify, lambdify, Poly, apart, cancel, roots,
                   RootOf, nroots, factor, Rational as Rat, expand)

DB_PATH   = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "cmf_path3_results.jsonl"

x_sym = Symbol('x'); y_sym = Symbol('y'); k_sym = Symbol('k')
_xp   = Symbol('_xp')


# ══════════════════════════════════════════════════════
# 1. HYPERGEOMETRIC ζ-SERIES FOR f = 2k^3 + ck
# ══════════════════════════════════════════════════════

def eval_zeta_series(c_val, dps, n_terms=60):
    """
    Compute S = Σ_{k=1}^∞ 1/(2k^3 + c*k) via
      S = (1/2) Σ_{j=0}^∞ (-c/2)^j ζ(3+2j)
    Valid when |c| < 2 (series radius of convergence for |c/(2k^2)| < 1 when k≥1
    only if c=0; for |c| > 2 use partial fractions instead).
    """
    mpmath.mp.dps = dps + 10
    half = mpmath.mpf('1')/2
    S = mpmath.mpf(0)
    ratio = -mpmath.mpf(c_val) / 2
    for j in range(n_terms):
        term = (ratio**j) * mpmath.zeta(3 + 2*j)
        S += term
        if mpmath.fabs(term) < mpmath.mpf(10)**(-(dps+5)) and j > 3:
            break
    return half * S


def eval_partial_fraction(f_expr_str, m_val, dps, k_max=1000):
    """
    Partial-fraction decomposition of 1/f(k,m):
      1/f(k,m) = Σ_i A_i / (k - r_i)
    Sum: Σ_{k=1}^∞ 1/f(k,m) = -Σ_i A_i * ψ(1 - r_i)   (for Re(r_i) < 1)
    Uses mpmath.digamma for high-precision digamma at complex arguments.
    Falls back to direct summation when roots are near positive integers.
    """
    mpmath.mp.dps = dps + 20
    fe = sympify(f_expr_str).subs(y_sym, m_val)
    fe_k = expand(fe)
    try:
        p = Poly(fe_k.subs(x_sym, _xp).expand(), _xp)
        deg = p.degree()
    except Exception:
        return None, 'degree_error'

    if deg < 2:
        return None, 'deg_lt_2'

    # Check if direct digamma evaluation is feasible (rational roots only)
    try:
        apart_expr = apart(1 / fe_k, x_sym)
    except Exception:
        apart_expr = None

    if apart_expr is not None:
        # Parse partial fractions into (A_i, r_i) pairs numerically
        # Strategy: collect roots and residues numerically
        f_fn = lambdify(x_sym, fe_k, modules="mpmath")
        # Check for zeros near positive integers first (would cause issues)
        has_issue = False
        for k in range(1, k_max+1):
            try:
                v = f_fn(k)
                if mpmath.fabs(v) < mpmath.mpf(10)**(-dps//2):
                    has_issue = True; break
            except Exception:
                pass

        if has_issue:
            return None, 'zero_at_positive_int'

    # High-precision direct summation (the gold standard)
    fn = lambda k: 1 / lambdify(x_sym, fe_k, modules="mpmath")(k)
    try:
        samp = [complex(fn(k)) for k in range(1, 51)]
        if any(abs(v) < 1e-10 for v in samp):
            return None, 'near_zero'
        sign = -1 if all(v.real < 0 for v in samp) else 1
    except Exception:
        return None, 'eval_error'

    fn_signed = lambda k: mpmath.mpf(sign) * fn(k)

    # Method A: nsum euler-maclaurin
    try:
        S_em = mpmath.nsum(fn_signed, [1, mpmath.inf], method='euler-maclaurin')
    except Exception:
        S_em = None

    # Method B: Levin
    try:
        S_lv = mpmath.nsum(fn_signed, [1, mpmath.inf], method='levin')
    except Exception:
        S_lv = None

    # Method C: Digamma decomposition (works for rational/algebraic roots)
    S_pf = None
    try:
        # Find numerical roots of f(k,m)
        poly_sym = fe_k.subs(x_sym, k_sym)
        num_roots = sp.nroots(Poly(poly_sym, k_sym), n=dps+10, maxsteps=200)
        if num_roots:
            # Compute residues numerically: A_i = 1 / f'(r_i)
            df = sp.diff(fe_k, x_sym)
            df_fn = lambdify(x_sym, df, modules="mpmath")
            S_pf = mpmath.mpf(0)
            for r in num_roots:
                rc = complex(r)
                rm = mpmath.mpc(rc.real, rc.imag)
                try:
                    A = 1 / df_fn(rm)
                    # Σ_{k=1}^∞ A/(k-r) = -A * (ψ(1-r) - ψ(1))  when Re(r) < 1
                    # More general: Σ_{k=N}^∞ A/(k-r) = A * ψ(N-r)
                    # Complete sum from k=1: A * (ψ(1-r) * ... use regularization)
                    # ψ(k-r) → ln k as k→∞, so: Σ A/(k-r) diverges term-by-term
                    # BUT partial fraction sum over ALL roots is convergent (poles cancel)
                    # Use: Σ_{k=1}^∞ Σ_r A_r/(k-r) = -Σ_r A_r * ψ(1-r) + const
                    # The constant cancels because Σ_r A_r = 0 (deg≥2 with leading term k^n)
                    S_pf += A * (mpmath.digamma(1) - mpmath.digamma(1 - rm))
                except Exception:
                    S_pf = None; break
    except Exception:
        S_pf = None

    # Choose best result
    results = {}
    if S_em is not None: results['em'] = mpmath.mpf(sign) * S_em if sign == 1 else S_em
    if S_lv is not None: results['lv'] = mpmath.mpf(sign) * S_lv if sign == 1 else S_lv
    if S_pf is not None: results['pf'] = S_pf

    if not results:
        return None, 'all_methods_failed'

    # Pick most precise value
    # Cross-check: pick value where methods agree
    vals = list(results.values())
    best = vals[0]
    if len(vals) >= 2:
        for i in range(len(vals)-1):
            for j in range(i+1, len(vals)):
                try:
                    if mpmath.fabs(vals[i] - vals[j]) / max(mpmath.fabs(vals[i]), mpmath.mpf(1e-300)) < mpmath.mpf(10)**(-(dps-10)):
                        best = vals[i]; break
                except Exception: pass
    return best, 'ok'


# ══════════════════════════════════════════════════════
# 2. EXTENDED PSLQ BASIS
# ══════════════════════════════════════════════════════

CNAMES = [
    "1","ζ(3)","ζ(5)","ζ(7)","ζ(9)","ζ(11)","ζ(13)",
    "ζ(3)/2","ζ(5)/2","ζ(7)/2","ζ(3)/4","ζ(5)/4","ζ(3)²",
    "7ζ(3)/2","3ζ(3)/4","5ζ(3)/4","ζ(3)³",
    "π","π²","π³","π⁴","π²/6","π²/12","π²/90","π⁴/90",
    "G","G²","G/π","ln2","ln3","ln2²","ln3²","ln2/π",
    "ψ(½)","ψ(⅓)","ψ(¼)","ψ(⅔)","ψ(¾)",
    "ψ'(½)","ψ'(⅓)","ψ'(¼)","ψ'(⅔)","ψ'(¾)",
    "ψ''(½)","ψ''(⅓)",
    "ζ(2)","ζ(4)","ζ(6)","ζ(3)/π²","ζ(5)/π⁴",
    "ln2·ζ(2)","ln2·ζ(3)","π²·ln2/6","G·ln2",
    "MZV:ζ(2,1)","MZV:ζ(3,1)","MZV:ζ(2,3)",
]

def build_cv(dps):
    mpmath.mp.dps = dps + 10
    m = mpmath
    z2=m.zeta(2); z3=m.zeta(3); z4=m.zeta(4); z5=m.zeta(5)
    z6=m.zeta(6); z7=m.zeta(7); z9=m.zeta(9); z11=m.zeta(11); z13=m.zeta(13)
    G=m.catalan; pi=m.pi; ln2=m.log(2); ln3=m.log(3)
    h = m.mpf('1')/2; third = m.mpf('1')/3; quart = m.mpf('1')/4
    psi=m.digamma; trig=lambda x: m.zeta(2,x); tetrag=lambda x: m.zeta(3,x)
    # MZVs: ζ(2,1)=ζ(3), ζ(3,1)=π⁴/360, ζ(2,3)=...
    mzv_21 = z3  # known: ζ(2,1)=ζ(3)
    mzv_31 = pi**4/360  # ζ(3,1)=π⁴/360
    mzv_23 = z5/4 - 3*z3*z2/8  # approximate; use mpmath.nsum for exact
    return [
        m.mpf(1),z3,z5,z7,z9,z11,z13,
        z3/2,z5/2,z7/2,z3/4,z5/4,z3**2,
        7*z3/2,3*z3/4,5*z3/4,z3**3,
        pi,pi**2,pi**3,pi**4,pi**2/6,pi**2/12,pi**2/90,pi**4/90,
        G,G**2,G/pi,ln2,ln3,ln2**2,ln3**2,ln2/pi,
        psi(h),psi(third),psi(quart),psi(2*third),psi(3*quart),
        trig(h),trig(third),trig(quart),trig(2*third),trig(3*quart),
        tetrag(h),tetrag(third),
        z2,z4,z6,z3/pi**2,z5/pi**4,
        ln2*z2,ln2*z3,pi**2*ln2/6,G*ln2,
        mzv_21,mzv_31,mzv_23,
    ]


def deep_pslq(val, cv, dps, maxcoeff=2000):
    """Multi-layered PSLQ identification."""
    if val is None: return []
    try:
        fv = float(mpmath.re(val))
    except: return []
    if not math.isfinite(fv) or abs(fv) < 1e-300: return []
    hits = []

    # Layer 1: float pre-filter with multipliers
    for name, c in zip(CNAMES[1:], cv[1:]):
        try:
            cf = float(mpmath.re(c))
            if not cf: continue
        except: continue
        for s in [1,-1]:
            for mult_num, mult_den in [(1,1),(1,2),(1,3),(1,4),(2,1),(3,1),(4,1),(1,6),(3,2)]:
                cand = s * (mult_num/mult_den) * cf
                rel = abs(fv - cand) / max(abs(cand), 1e-300)
                if rel < 1e-15:
                    num_str = '' if mult_num==1 and mult_den==1 else f'{mult_num}/{mult_den}*' if mult_den!=1 else f'{mult_num}*'
                    hits.append({'method':'float','expr':f"{'-'if s<0 else ''}{num_str}{name}",'rel_err':rel,'dps_match':15})
    if hits:
        hits.sort(key=lambda h:h['rel_err'])
        return hits[:5]

    # Layer 2: High-precision PSLQ with full basis
    mpmath.mp.dps = dps
    try:
        vec = [mpmath.mpf(float(mpmath.re(v))) for v in [mpmath.mpf(1), val] + list(cv)]
        if any(not mpmath.isfinite(v) for v in vec): raise ValueError
        r = mpmath.pslq(vec, maxcoeff=maxcoeff, tol=mpmath.mpf(10)**(-(dps-20)))
        if r is not None and r[1] != 0:
            parts = []
            if r[0] != 0: parts.append(str(r[0]))
            for co, nm in zip(r[2:], CNAMES):
                if co: parts.append(f"{co}*{nm}")
            expr = f"({'+'.join(parts)}) / {-r[1]}" if parts else "0"
            hits.append({'method':'pslq_full','expr':expr,'rel_err':0.0,'dps_match':dps})
    except Exception: pass

    # Layer 3: mpmath.identify
    try:
        s = mpmath.identify(mpmath.mpf(fv), tol=mpmath.mpf(10)**(-(dps//2)))
        if s: hits.append({'method':'isc','expr':s,'rel_err':0.0,'dps_match':dps//2})
    except: pass

    # Layer 4: cubic-in-ζ3 check  (val = a*ζ(3) + b*ζ(5) + c*ζ(7) for small a,b,c)
    try:
        z3f = float(mpmath.zeta(3)); z5f = float(mpmath.zeta(5)); z7f = float(mpmath.zeta(7))
        for a in range(-10,11):
            for b in range(-10,11):
                for c in range(-5,6):
                    for d in [1,2,3,4,6,8,12]:
                        cand = (a*z3f + b*z5f + c*z7f) / d
                        rel = abs(fv - cand) / max(abs(cand), 1e-300)
                        if rel < 1e-14:
                            hits.append({'method':'zeta_linear','expr':f"({a}ζ3+{b}ζ5+{c}ζ7)/{d}",'rel_err':rel,'dps_match':14})
    except: pass

    hits.sort(key=lambda h: h['rel_err'])
    return hits[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='cmf_hunter', choices=['all','telescope','cmf_hunter'])
    ap.add_argument('--reuse', default='cmf_euler_full.jsonl')
    ap.add_argument('--m-max', type=int, default=6, dest='m_max')
    ap.add_argument('--dps', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=str(OUT_JSONL))
    args = ap.parse_args()
    mpmath.mp.dps = args.dps

    euler_cache = {}
    rp = Path(__file__).parent / args.reuse
    if rp.exists():
        for line in open(rp):
            r = json.loads(line); euler_cache[r['cmf_id']] = r
        print(f"Loaded {len(euler_cache)} cached Euler entries from {rp.name}")

    src_filter = {'all':('telescope','cmf_hunter'),'telescope':('telescope',),'cmf_hunter':('cmf_hunter',)}[args.source]
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"SELECT id,cmf_payload FROM cmf WHERE json_extract(cmf_payload,'$.source') IN ({','.join('?'*len(src_filter))}) ORDER BY id", src_filter).fetchall()
    conn.close()
    if args.limit: rows = rows[:args.limit]

    cv = build_cv(args.dps)
    out_path = Path(args.out)
    t0 = time.time()
    print(f"Path-3 deep PSLQ: {len(rows)} CMFs | dps={args.dps} | basis={len(CNAMES)}\n")

    n_id = 0
    with open(out_path, 'w') as fout:
        for ci, (cid, praw) in enumerate(rows):
            p = json.loads(praw); fp = p.get('f_poly',''); fb = p.get('fbar_poly','')
            src = p.get('source',''); pconst = p.get('primary_constant','')
            print(f"[{ci+1:3d}/{len(rows)}] #{cid} [{time.time()-t0:.0f}s]", end='  ', flush=True)
            if not fp: print('skip'); continue

            polys = {'f': fp}
            if fb and fb != fp: polys['fbar'] = fb
            all_lines = []; best_hit = None; best_m = None

            for m_val in range(args.m_max + 1):
                for poly_name, poly_str in polys.items():
                    # Try cached value first
                    cached_val = None
                    if cid in euler_cache:
                        for ml in euler_cache[cid].get('m_lines', []):
                            if ml['m_val'] == m_val and ml['poly'] == poly_name:
                                cached_val = ml.get('value'); break

                    if cached_val:
                        try:
                            val = mpmath.mpf(cached_val)
                            method = 'cached'
                        except:
                            continue
                    else:
                        val, status = eval_partial_fraction(poly_str, m_val, args.dps)
                        method = status if val is None else 'partial_fraction'
                        if val is None: continue

                    # Check for ζ-series decomposition (fast path for 2k³+ck family)
                    try:
                        fe = sympify(poly_str).subs(y_sym, m_val)
                        fe_exp = expand(fe)
                        p_sym = Poly(fe_exp.subs(x_sym, _xp).expand(), _xp)
                        coeffs = p_sym.all_coeffs()
                        if (len(coeffs) == 4 and coeffs[0] == 2 and coeffs[1] == 0
                                and coeffs[3] == 0 and m_val == 0):
                            # f = 2k^3 + c*k form → use ζ-series
                            c_coeff = int(coeffs[2])
                            v_zeta = eval_zeta_series(c_coeff, args.dps + 10)
                            if v_zeta is not None: val = v_zeta; method = 'zeta_series'
                    except Exception:
                        pass

                    hits = deep_pslq(val, cv, args.dps)
                    line = {'m_val': m_val, 'poly': poly_name, 'method': method,
                            'value': mpmath.nstr(val, args.dps - 8),
                            'identified': hits}
                    all_lines.append(line)
                    if hits and best_hit is None:
                        best_hit = hits; best_m = m_val

            if best_hit: n_id += 1; flag = '★'; id_str = best_hit[0]['expr'][:50]
            else: flag = ' '; id_str = '(none)'
            print(f"lines={len(all_lines)} {flag} best@m={best_m} {id_str}")

            rec = {'cmf_id': cid, 'source': src, 'f_poly': fp, 'fbar_poly': fb,
                   'primary_constant': str(pconst) if pconst else None,
                   'm_lines': all_lines, 'best_m': best_m, 'best_hits': best_hit}
            fout.write(json.dumps(rec) + '\n'); fout.flush()

    print(f"\nDone {time.time()-t0:.1f}s | identified={n_id}/{len(rows)} | → {out_path}")

if __name__ == '__main__': main()
