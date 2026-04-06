#!/usr/bin/env python3
"""
cmf_euler_scan.py
=================
Euler-sum trajectory scan for all CMF Hunter (and telescope) Atlas entries.

The native readout for conjugate-polynomial CMFs is the EULER SUM:
    L(m) = Σ_{k=1}^∞  1 / f(k, m)

At each fixed m, this gives a (usually different) constant.
At m=0, f=2x³+2y³ → Σ 1/(2k³) = ζ(3)/2  ✓

This is the readout used by CMF Hunter and cmf_microscope.py.
It is NOT a matrix walk; it is a direct series sum computed with mpmath.nsum
to arbitrary precision — fast and highly accurate.

Why this is better than the 2×2 companion walk:
  - 60+ digit precision in seconds (vs ~14 digits after depth 3000 walk)
  - The limit is exactly the certified primary_constant at m=0
  - Each m-line can give a DIFFERENT constant (new formula harvest)
  - No convergence-ratio issues (the sum converges as 1/k^{deg-1})

Usage:
    python3 cmf_euler_scan.py --source cmf_hunter
    python3 cmf_euler_scan.py --source all --m-max 20 --dps 65
    python3 cmf_euler_scan.py --limit 10   # smoke test
"""
import argparse, json, math, sqlite3, time
from pathlib import Path

import mpmath
import sympy as sp
from sympy import Symbol, sympify, lambdify, Poly, degree

DB_PATH   = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "cmf_euler_results.jsonl"

CONST_NAMES = ["π","e","ln2","ζ(3)","ζ(5)","π²/6","G","4/π","1/π","π²","ln3","√2","φ","√3","√5",
               "2/π","ζ(2)","ζ(3)/2","ζ(5)/2","ζ(3)/π²","G/π","ln2/π",
               "7ζ(3)/2","3ζ(3)/4","ζ(3)²","π³/32","π²·ln2/8",
               "ζ(2)/2","ζ(4)","ζ(4)/4"]

def build_const_vec():
    m = mpmath
    z3=m.zeta(3); z5=m.zeta(5); G=m.catalan; pi=m.pi; z2=m.zeta(2)
    return [pi, m.e, m.log(2), z3, z5, pi**2/6, G, 4/pi, 1/pi, pi**2,
            m.log(3), m.sqrt(2), (1+m.sqrt(5))/2, m.sqrt(3), m.sqrt(5),
            2/pi, z2, z3/2, z5/2, z3/pi**2, G/pi, m.log(2)/pi,
            7*z3/2, 3*z3/4, z3**2, pi**3/32, pi**2*m.log(2)/8,
            z2/2, m.zeta(4), m.zeta(4)/4]

x_sym = Symbol('x')    # plain symbol for lambdify
_xp   = Symbol('_xp')  # plain symbol for Poly degree detection
y_sym = Symbol('y')

def euler_sum(f_poly_str, m_val, dps=65, N_check=500):
    """Compute Σ_{k=1}^∞ 1/f(k, m_val) using mpmath.nsum.

    Returns (value, deg, is_convergent).
    deg is the degree of f(k, m_val) in k — convergent iff deg >= 2.
    """
    fe = sympify(f_poly_str)
    f_m = fe.subs(y_sym, m_val)
    # Degree detection: substitute a plain symbol to avoid positive=True quirk
    try:
        f_m_plain = f_m.subs(x_sym, _xp)
        p = Poly(f_m_plain.expand(), _xp)
        deg = p.degree()
    except Exception:
        deg = 0
    if deg < 2:
        return None, deg, False

    f_fn = lambdify(x_sym, f_m, modules="mpmath")

    # Check for zeros / sign changes in first 50 terms; also detect all-negative
    try:
        sample_vals = [complex(f_fn(kk)) for kk in range(1, 51)]
        # Skip if any zero
        if any(abs(v) < 1e-10 for v in sample_vals):
            return None, deg, False
        # All-negative: negate f so sum is positive
        all_neg = all(v.real < 0 for v in sample_vals)
    except Exception:
        return None, deg, False
    
    sign = -1 if all_neg else 1

    old_dps = mpmath.mp.dps
    mpmath.mp.dps = dps + 10
    try:
        val = mpmath.nsum(lambda k: mpmath.mpf(sign) / f_fn(k), [1, mpmath.inf],
                          method="euler-maclaurin", error=True)
        if isinstance(val, tuple):
            val, err = val
            if float(abs(err)) > 10**(-dps//2):
                val = mpmath.nsum(lambda k: mpmath.mpf(sign) / f_fn(k), [1, mpmath.inf],
                                  method="richardson")
    except Exception:
        try:
            val = mpmath.nsum(lambda k: mpmath.mpf(sign) / f_fn(k), [1, mpmath.inf])
        except Exception:
            return None, deg, False
    finally:
        mpmath.mp.dps = old_dps

    return val, deg, True

def identify(val, cv):
    """Identification: float match, PSLQ, ISC."""
    if val is None: return []
    fv = float(val)
    if not math.isfinite(fv) or fv == 0: return []
    hits = []
    for name, c in zip(CONST_NAMES, cv):
        cf = float(c)
        if not cf: continue
        for s in [1, -1]:
            for mult in [mpmath.mpf(1), mpmath.mpf(2), mpmath.mpf(3), mpmath.mpf(4),
                         mpmath.mpf("1")/2, mpmath.mpf("1")/3, mpmath.mpf("1")/4,
                         mpmath.mpf("1")/6, mpmath.mpf("1")/8]:
                cand = float(s * mult * c)
                rel = abs(fv - cand) / max(abs(cand), 1e-300)
                if rel < 1e-12:
                    lbl = f"{'-' if s<0 else ''}{'' if float(mult)==1.0 else str(float(mult))+'*'}{name}"
                    hits.append({"method": "float", "expr": lbl, "rel_err": rel})
    if hits:
        hits.sort(key=lambda h: h["rel_err"])
        return hits[:4]

    try:
        v = mpmath.mpf(fv)
        vec = [mpmath.mpf(1), v] + [mpmath.mpf(float(c)) for c in cv]
        rel = mpmath.pslq(vec, maxcoeff=500, tol=mpmath.mpf("1e-25"))
        if rel and rel[1] != 0:
            parts = [str(rel[0])] + [f"{co}*{nm}" for co,nm in zip(rel[2:], CONST_NAMES) if co]
            numer = " + ".join(p for p in parts if p != "0") or "0"
            hits.append({"method": "pslq", "expr": f"({numer}) / {-rel[1]}", "rel_err": 0.0})
    except Exception: pass

    try:
        s = mpmath.identify(mpmath.mpf(fv), tol=1e-15)
        if s: hits.append({"method": "isc", "expr": s, "rel_err": 0.0})
    except Exception: pass

    hits.sort(key=lambda h: h["rel_err"])
    return hits[:4]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="all", choices=["all","telescope","cmf_hunter"])
    ap.add_argument("--m-max",   type=int, default=15, dest="m_max")
    ap.add_argument("--dps",     type=int, default=65)
    ap.add_argument("--limit",   type=int, default=0)
    ap.add_argument("--out",     type=str, default=str(OUT_JSONL))
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

    if args.limit: rows = rows[:args.limit]

    cv = build_const_vec()
    out_path = Path(args.out)
    t0 = time.time()

    print(f"CMF Euler Scan: {len(rows)} CMFs | m ∈ [0,{args.m_max}] | dps={args.dps}")
    print(f"  Readout: Σ_k 1/f(k,m_val)  (Euler-sum, native CMF readout)")
    print(f"  Output: {out_path}\n")

    n_identified = 0; n_new = 0

    with open(out_path, "w") as fout:
        for ci, (cid, praw) in enumerate(rows):
            p = json.loads(praw)
            fp  = p.get("f_poly", "");  fb = p.get("fbar_poly", "")
            src = p.get("source", "");   pconst = p.get("primary_constant", "")
            cert = p.get("certification_level", "")
            elapsed = time.time() - t0

            print(f"[{ci+1:3d}/{len(rows)}] #{cid} ({src}) [{elapsed:.0f}s]", end="  ", flush=True)

            if not fp:
                print("skip (no f_poly)"); continue

            # For fbar, run both f and fbar Euler sums (different formulas)
            polys = {"f": fp}
            if fb and fb != fp: polys["fbar"] = fb

            all_lines = []   # list of {m_val, poly_name, value, deg, hits}
            best_hit = None; best_m = None; best_poly = None

            for poly_name, poly_str in polys.items():
                for m_val in range(args.m_max + 1):
                    val, deg, converges = euler_sum(poly_str, m_val, dps=args.dps)
                    if not converges or val is None:
                        continue
                    val_str = mpmath.nstr(val, 35)
                    hits = identify(val, cv)

                    line = {
                        "m_val":     m_val,
                        "poly":      poly_name,
                        "value":     val_str,
                        "deg":       deg,
                        "identified": hits,
                    }
                    all_lines.append(line)

                    if hits and best_hit is None:
                        best_hit = hits; best_m = m_val; best_poly = poly_name

            # Check primary_constant match at m=0
            pconst_match = None
            if pconst and str(pconst) not in ("None", ""):
                try:
                    pconst_val = float(mpmath.mpf(str(sp.sympify(str(pconst)).evalf(60))))
                    m0_line = next((l for l in all_lines if l["m_val"]==0 and l["poly"]=="f"), None)
                    if m0_line:
                        m0_val = float(mpmath.mpf(m0_line["value"]))
                        rel = abs(m0_val - pconst_val) / max(abs(pconst_val), 1e-300)
                        pconst_match = {"matches": rel < 1e-10, "rel_err": rel}
                except Exception:
                    pconst_match = None

            converging_lines = len(all_lines)
            identified_lines = [l for l in all_lines if l["identified"]]

            if best_hit:
                n_identified += 1
                flag = "★"
                id_str = best_hit[0]["expr"][:45]
                if not pconst:
                    n_new += 1
            else:
                flag = " "; id_str = "(none)"

            pconst_ok = pconst_match.get("matches") if pconst_match else None
            pconst_str = f"pconst_ok={pconst_ok}" if pconst_match else ""
            print(f"lines={converging_lines}  id={len(identified_lines)}  {flag}  best@m={best_m}  {id_str}  {pconst_str}")

            rec = {
                "cmf_id":     cid,
                "source":     src,
                "cert":       cert,
                "f_poly":     fp,
                "fbar_poly":  fb,
                "primary_constant": str(pconst) if pconst else None,
                "pconst_match": pconst_match,
                "m_lines":    all_lines,
                "best_m":     best_m,
                "best_poly":  best_poly,
                "best_hits":  best_hit,
                "n_identified_lines": len(identified_lines),
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()

    total = time.time() - t0
    print(f"\nDone in {total:.1f}s")
    print(f"  CMFs with ≥1 identified m-line: {n_identified}/{len(rows)}")
    print(f"  Of those, no primary_constant (new?): {n_new}")
    print(f"  Results → {out_path}")

if __name__ == "__main__":
    main()
