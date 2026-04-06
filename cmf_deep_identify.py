#!/usr/bin/env python3
"""
cmf_deep_identify.py
====================
Deep walk + ISC identification for all telescope (gauge) and CMF Hunter CMFs.

Strategy:
 - Telescope CMFs: unknown limits → deep walk at high dps, try mpmath.identify + PSLQ
 - CMF Hunter CMFs: compare 2x2 walk limit vs certified primary_constant
 - Output: walk_identify_results.jsonl  +  walk_identify_report.txt

Usage:
    python3 cmf_deep_identify.py
    python3 cmf_deep_identify.py --source telescope   # telescope only
    python3 cmf_deep_identify.py --depth 2000 --dps 80
"""
import argparse, json, math, sqlite3, time
from pathlib import Path
import mpmath
from sympy import symbols as _sym, sympify as _sp, lambdify as _lam, expand as _exp

DB_PATH  = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "walk_identify_results.jsonl"
OUT_TXT   = Path(__file__).parent / "walk_identify_report.txt"

# ── Constant library ──────────────────────────────────────────────────────────
CONST_NAMES = [
    "π", "e", "ln2", "ζ(3)", "ζ(5)", "π²/6", "G",
    "4/π", "1/π", "π²", "ln3", "√2", "φ", "√3", "√5",
    "2/π", "ζ(2)", "ζ(3)/π²", "ln2/π", "G/π",
    "Γ(1/3)", "Γ(1/4)", "ζ(3)²",
]

def build_const_vec():
    m = mpmath
    z3 = m.zeta(3); z5 = m.zeta(5); z2 = m.zeta(2)
    G  = m.catalan; pi = m.pi
    return [
        pi, m.e, m.log(2), z3, z5, pi**2/6, G,
        4/pi, 1/pi, pi**2, m.log(3), m.sqrt(2), (1+m.sqrt(5))/2, m.sqrt(3), m.sqrt(5),
        2/pi, z2, z3/pi**2, m.log(2)/pi, G/pi,
        m.gamma(mpmath.mpf("1")/3), m.gamma(mpmath.mpf("1")/4), z3**2,
    ]

# ── Walk builder (mirrors api.py) ─────────────────────────────────────────────
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
        evb  = lambda e, kv, nv: complex(e.subs([(k, kv), (zs, nv)]))
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

def do_walk(Kfn, k0: int, depth: int):
    P = mpmath.eye(2)
    for it in range(1, depth + 1):
        try:
            P = P * Kfn(k0 + it - 1)
            if it % 20 == 0:
                sc = max(abs(float(mpmath.re(P[0,0]))), abs(float(mpmath.re(P[0,1]))), 1e-300)
                P  = P / sc
        except Exception:
            pass
    d = P[1,1]; n = P[0,1]
    return mpmath.re(n/d) if mpmath.fabs(d) > 1e-200 else None

def delta_from_walk(Kfn, k0, depth):
    """Compute convergence delta using milestones at depth//3, 2//3, depth."""
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
    """Try ~10 trajectories at moderate depth, return best (label, Kfn, k0)."""
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

# ── PSLQ + ISC identification ─────────────────────────────────────────────────
def identify_value(val, cv):
    """Full identification suite: float match, PSLQ, mpmath.identify."""
    if val is None: return []
    fv = float(val)
    if not math.isfinite(fv) or fv == 0: return []
    hits = []

    # 1. Float match against known constants and simple multiples
    for name, c in zip(CONST_NAMES, cv):
        cf = float(c)
        if cf == 0: continue
        for sign in [1, -1]:
            for mult in [1, 2, 3, 4, mpmath.mpf("1")/2, mpmath.mpf("1")/3, mpmath.mpf("1")/4]:
                candidate = float(sign * mult * cf)
                rel = abs(fv - candidate) / max(abs(candidate), 1e-300)
                if rel < 1e-10:
                    label = f"{'-' if sign<0 else ''}{'' if mult==1 else str(mult)+'*'}{name}"
                    hits.append({"method": "float", "expr": label, "rel_err": rel})

    if hits:
        hits.sort(key=lambda x: x["rel_err"])
        return hits[:3]

    # 2. PSLQ against [1, val, c1, c2, ...]
    try:
        mpmath.mp.dps = max(mpmath.mp.dps, 60)
        v = mpmath.mpf(fv)
        vec = [mpmath.mpf(1), v] + [mpmath.mpf(float(c)) for c in cv]
        rel = mpmath.pslq(vec, maxcoeff=500, tol=mpmath.mpf("1e-35"))
        if rel is not None and rel[1] != 0:
            parts = []
            if rel[0] != 0: parts.append(str(rel[0]))
            for co, nm in zip(rel[2:], CONST_NAMES):
                if co != 0: parts.append(f"{co}*{nm}")
            numer = " + ".join(parts) if parts else "0"
            hits.append({"method": "pslq", "expr": f"({numer}) / {-rel[1]}", "coeffs": list(rel), "rel_err": 0.0})
    except Exception as e:
        hits.append({"method": "pslq_err", "expr": str(e)[:80], "rel_err": 1.0})

    # 3. mpmath.identify (ISC database lookup)
    try:
        s = mpmath.identify(mpmath.mpf(fv), tol=1e-12)
        if s: hits.append({"method": "isc", "expr": s, "rel_err": 0.0})
    except Exception: pass

    hits.sort(key=lambda x: x["rel_err"])
    return hits[:5]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   default="all", choices=["all","telescope","cmf_hunter"])
    ap.add_argument("--depth",    type=int, default=1500)
    ap.add_argument("--dps",      type=int, default=65)
    ap.add_argument("--scan-depth", type=int, default=200, dest="scan_depth")
    ap.add_argument("--out",        type=str, default=str(OUT_JSONL))
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

    out_path = Path(args.out)
    cv = build_const_vec()
    print(f"Deep identify: {len(rows)} CMFs | source={args.source} | depth={args.depth} | dps={args.dps}")
    print(f"  Output: {out_path}\n")

    results = []
    t0 = time.time()

    with open(out_path, "w") as fout:
        for ci, (cid, praw) in enumerate(rows):
            p      = json.loads(praw) if isinstance(praw, str) else praw
            fp     = p.get("f_poly", "");  fb = p.get("fbar_poly", "")
            src    = p.get("source", "");   cert = p.get("certification_level", "")
            pconst = p.get("primary_constant", None)
            elapsed = time.time() - t0

            print(f"[{ci+1:3d}/{len(rows)}] #{cid} ({src}) [{elapsed:.0f}s]", end="  ", flush=True)

            if not fp or not fb:
                print("skip (no f_poly)")
                continue

            try:
                Kx, Ky, Kz, is3d = build_fns(fp, fb)
            except Exception as e:
                print(f"BUILD ERR: {e}")
                continue

            # Phase 1: find best trajectory at moderate depth
            best, scan_delta = best_trajectory(is3d, Kx, Ky, Kz, args.scan_depth)
            if best is None:
                print("no converging trajectory")
                continue

            lbl, Kfn, k0 = best
            sd_str = f"{scan_delta:.3f}" if scan_delta is not None else "N/A"
            print(f"best={lbl} scan_delta={sd_str}", end="  ", flush=True)

            # Phase 2: deep walk on best trajectory
            try:
                deep_delta, deep_est = delta_from_walk(Kfn, k0, args.depth)
                if deep_est is None:
                    print("degenerate")
                    continue
            except Exception as e:
                print(f"WALK ERR: {e}")
                continue

            est_str = mpmath.nstr(deep_est, 35)
            dd_str = f"{deep_delta:.3f}" if deep_delta is not None else "N/A"
            print(f"delta={dd_str}  est={est_str[:30]}", end="  ", flush=True)

            # Phase 3: identification
            hits = identify_value(deep_est, cv)

            # Phase 4: check against primary_constant if known
            pconst_match = None
            if pconst and str(pconst) not in ("None", ""):
                try:
                    import sympy as _sym_pc
                    pconst_val = float(mpmath.mpf(str(_sym_pc.sympify(str(pconst)).evalf(60))))
                    rel = abs(float(deep_est) - pconst_val) / max(abs(pconst_val), 1e-300)
                    pconst_match = {"matches": rel < 1e-6, "rel_err": rel, "pconst_float": pconst_val}
                except Exception:
                    pconst_match = {"matches": False, "rel_err": None}

            if hits:
                print(f"→ {hits[0]['expr'][:50]}")
            elif pconst_match and pconst_match["matches"]:
                print(f"→ matches pconst {pconst}")
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
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            results.append(rec)

    # ── Summary report ─────────────────────────────────────────────────────
    total = time.time() - t0
    identified   = [r for r in results if r["identified"]]
    strong_delta = [r for r in results if r.get("deep_delta") and r["deep_delta"] >= 0.05]
    unidentified = [r for r in results if not r["identified"]]

    report_lines = [
        f"CMF Deep Identification Report",
        f"================================",
        f"Sources: {src_filter}  |  CMFs processed: {len(results)}",
        f"Depth: {args.depth}  |  dps: {args.dps}  |  Wall-time: {total:.1f}s",
        f"Output: {out_path}",
        f"",
        f"IDENTIFIED ({len(identified)}):",
        f"-----------------------------",
    ]
    for r in sorted(identified, key=lambda x: -(x.get("deep_delta") or 0)):
        h = r["identified"][0]
        dval = r.get('deep_delta')
        dstr = f"{dval:.3f}" if dval is not None else 'N/A'
        report_lines.append(
            f"  #{r['cmf_id']:4d} ({r['source']:12s}) delta={dstr:>7}  "
            f"f={r['f_poly'][:35]:35s}  → {h['expr'][:60]}"
        )

    report_lines += [
        f"",
        f"STRONG CONVERGENCE but UNIDENTIFIED ({len([r for r in unidentified if r.get('deep_delta') and r['deep_delta']>=0.05])}):",
        f"-----------------------------------------------------",
    ]
    for r in sorted([r for r in unidentified if r.get("deep_delta") and r["deep_delta"] >= 0.05],
                    key=lambda x: -(x.get("deep_delta") or 0)):
        report_lines.append(
            f"  #{r['cmf_id']:4d} ({r['source']:12s}) delta={r['deep_delta']:>7.3f}  "
            f"f={r['f_poly'][:35]:35s}  est={r['estimate'][:40]}"
        )

    report_lines += [
        f"",
        f"ALL RESULTS ({len(results)}):",
        f"------------------",
    ]
    for r in results:
        h_str = r["identified"][0]["expr"][:50] if r["identified"] else "(none)"
        report_lines.append(
            f"  #{r['cmf_id']:4d} ({r['source']:12s}) delta={str(r.get('deep_delta') or 'N/A'):>8}  "
            f"id={h_str}"
        )

    report = "\n".join(report_lines)
    print()
    print(report)
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(report)
    print(f"\nReport saved to {txt_path}")
    print(f"JSONL saved to {out_path}")

if __name__ == "__main__":
    main()
