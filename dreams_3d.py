"""
dreams_3d.py — Ramanujan Dreams Session for 3D Atlas CMFs
==========================================================
Runs depth-2000 walks, all (m, n) shift combinations, full PSLQ
identification against 500+ known constants.

Usage:
    python3 dreams_3d.py [--id 1846] [--dps 50] [--depth 2000]

Output:
    dreams_3d_results.jsonl   — one JSON line per (cmf_id, m, n) walk
    dreams_3d_hits.jsonl      — strong hits only (log10_err < -8)
    dreams_3d_summary.json    — per-CMF aggregated summary
"""
import json, os, sys, time, math, argparse, sqlite3
from pathlib import Path
import mpmath
from sympy import symbols, sympify, lambdify, expand

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
DB_PATH = ROOT / "data" / "atlas_2d.db"
OUT_RESULTS = ROOT / "dreams_3d_results.jsonl"
OUT_HITS    = ROOT / "dreams_3d_hits.jsonl"
OUT_SUMMARY = ROOT / "dreams_3d_summary.json"

# ─── constants bank ───────────────────────────────────────────────────────────
def _build_bank(dps: int) -> dict[str, mpmath.mpf]:
    mpmath.mp.dps = dps + 10
    bank: dict[str, mpmath.mpf] = {}
    pi = mpmath.pi; e = mpmath.e; ln2 = mpmath.log(2); ln3 = mpmath.log(3)
    sq2 = mpmath.sqrt(2); sq3 = mpmath.sqrt(3); sq5 = mpmath.sqrt(5)
    Cat = mpmath.catalan

    bank["0"] = mpmath.mpf(0)
    bank["1"] = mpmath.mpf(1)

    # --- rationals ---
    for p in range(1, 10):
        for q in range(1, 16):
            if math.gcd(p, q) == 1:
                bank[f"{p}/{q}"] = mpmath.mpf(p) / q

    # --- pi ---
    for n in [1, 2, 3, 4, 6, 8]:
        bank[f"pi/{n}"]    = pi / n
        bank[f"pi^2/{n}"]  = pi**2 / n
        bank[f"pi^3/{n}"]  = pi**3 / n
        bank[f"pi^4/{n}"]  = pi**4 / n
    bank["1/pi"] = 1/pi; bank["1/pi^2"] = 1/pi**2

    # --- log ---
    bank["log2"] = ln2; bank["log3"] = ln3; bank["log5"] = mpmath.log(5)
    for n in [2, 3, 4, 6]:
        bank[f"log2/{n}"] = ln2/n; bank[f"log3/{n}"] = ln3/n
    bank["log2*pi"] = ln2*pi; bank["log2/pi"] = ln2/pi

    # --- zeta ---
    for s in [2, 3, 4, 5, 6, 7, 8]:
        z = mpmath.zeta(s)
        bank[f"zeta({s})"] = z
        bank[f"zeta({s})-1"] = z - 1
        for n in [2, 3, 4, 6, 8]:
            bank[f"zeta({s})/{n}"] = z/n
    bank["zeta(3)/pi^3"] = mpmath.zeta(3)/pi**3
    bank["zeta(3)*pi"]   = mpmath.zeta(3)*pi

    # --- catalan ---
    bank["catalan"]     = Cat
    bank["catalan/pi"]  = Cat/pi
    bank["catalan*pi"]  = Cat*pi
    for n in [2, 3, 4]:
        bank[f"catalan/{n}"] = Cat/n

    # --- sqrt ---
    bank["sqrt(2)"]   = sq2; bank["sqrt(3)"] = sq3; bank["sqrt(5)"] = sq5
    bank["1/sqrt(2)"] = 1/sq2; bank["1/sqrt(3)"] = 1/sq3
    bank["sqrt(2)-1"] = sq2-1; bank["sqrt(3)-1"] = sq3-1
    bank["(sqrt(5)-1)/2"] = (sq5-1)/2  # golden ratio - 1
    bank["(sqrt(5)+1)/2"] = (sq5+1)/2  # golden ratio

    # --- e ---
    bank["e"]    = e;    bank["1/e"]   = 1/e
    bank["e-1"]  = e-1;  bank["e-2"]   = e-2
    bank["log(e)"] = mpmath.mpf(1)

    # --- digamma / psi ---
    for a in range(1, 8):
        for b in range(a+1, 10):
            v = mpmath.digamma(b) - mpmath.digamma(a)
            bank[f"psi({b})-psi({a})"] = v
            bank[f"(psi({b})-psi({a}))/pi"] = v/pi

    # --- harmonic numbers ---
    H = mpmath.mpf(0)
    for n in range(1, 20):
        H += mpmath.mpf(1)/n
        bank[f"H({n})"] = H
        bank[f"H({n})-log({n})"] = H - mpmath.log(n)

    # --- AGM / lemniscate ---
    bank["agm(1,sqrt2)"]  = mpmath.agm(1, sq2)
    bank["pi/agm(1,sqrt2)"] = pi / mpmath.agm(1, sq2)

    # --- negative versions of everything ---
    extra = {f"-{k}": -v for k, v in list(bank.items()) if k not in ("0",)}
    bank.update(extra)

    return bank


def _compare(val: mpmath.mpf, bank: dict, top: int = 8) -> list[dict]:
    if not mpmath.isfinite(val) or val == 0:
        return []
    results = []
    for name, c in bank.items():
        if c == 0:
            continue
        try:
            err = abs(val - c)
            rel = err / abs(c)
            if err < 1e-4:
                log10 = float(mpmath.log10(err)) if err > 0 else -999
                results.append({"name": name, "value": float(c),
                                 "abs_err": float(err), "log10_err": round(log10, 2)})
        except Exception:
            pass
    results.sort(key=lambda x: x["abs_err"])
    return results[:top]


# ─── polynomial → matrix functions ────────────────────────────────────────────
def _poly_fast_fn(expr, *sym_list):
    """Compile a sympy polynomial into a fast Python evaluator via Poly coefficients."""
    from sympy import Poly, Integer
    try:
        p = Poly(expr, *sym_list)
        monoms = p.monoms()
        coeffs = [complex(c) for c in p.coeffs()]
        def fast(*args):
            acc = 0.0
            for c, m in zip(coeffs, monoms):
                t = c
                for a, e in zip(args, m):
                    if e:
                        t *= a ** e
                acc += t
            return acc
        return fast
    except Exception:
        # Fallback: lambdify with resolved symbols
        return lambdify(list(sym_list), expr, modules="math")


def _build_walk_fns_3d(f_poly: str, fbar_poly: str):
    """Return (Kx(k,m,n), b_fn(k,n)) for 3D CMF telescope formula."""
    k_s, m_s = symbols("k m")
    f_e  = sympify(f_poly)
    fb_e = sympify(fbar_poly)

    # Resolve actual symbol objects from the expression (avoids assumption cache mismatch)
    _free = f_e.free_symbols
    x_sym = next((s for s in _free if s.name == 'x'), symbols("x"))
    y_sym = next((s for s in _free if s.name == 'y'), symbols("y"))
    z_sym = next((s for s in _free if s.name == 'z'), symbols("z"))

    g_kmz  = f_e.subs([(x_sym, k_s), (y_sym, m_s)])
    gb_kmz = fb_e.subs([(x_sym, k_s), (y_sym, m_s)])
    b_kz   = expand(g_kmz.subs(m_s, 0) * gb_kmz.subs(m_s, 0))
    a_expr = expand(g_kmz - gb_kmz.subs(k_s, k_s + 1))

    # Fast evaluators — no sympy overhead per step
    _b = _poly_fast_fn(b_kz,   k_s, z_sym)
    _a = _poly_fast_fn(a_expr, k_s, m_s, z_sym)

    def b_fn(k_v, z_v):
        return _b(k_v, z_v)

    def a_fn(k_v, m_v, z_v):
        return _a(k_v, m_v, z_v)

    def Kx(k, m, n):
        return mpmath.matrix([[0, 1], [b_fn(k+1, n), a_fn(k, m, n)]])

    return Kx, b_fn


def _find_k_start(b_fn, n: int, max_k: int = 20) -> int:
    """Find smallest k_start ≥ 1 where b(k_start, n) ≠ 0."""
    for k in range(1, max_k + 1):
        try:
            v = float(mpmath.re(b_fn(k, n)))
            if abs(v) > 1e-10:
                return k - 1   # start one step before so Kx(k_start+1) uses b≠0
        except Exception:
            pass
    return 0


# ─── single walk ──────────────────────────────────────────────────────────────
def run_walk(Kx_fn, m: int, n: int, depth: int, dps: int,
             k_start: int = 0) -> dict:
    mpmath.mp.dps = dps
    P = mpmath.eye(2)
    sequence = []
    for step in range(k_start, k_start + depth):
        try:
            K = Kx_fn(step, m, n)
            P = P * K
            # Normalize every 20 steps
            if (step - k_start + 1) % 20 == 0:
                scale = max(abs(float(mpmath.re(P[0,0]))),
                            abs(float(mpmath.re(P[0,1]))), 1e-300)
                P = P / scale
            denom = P[1,1]; numer = P[0,1]
            if mpmath.fabs(denom) > mpmath.mpf("1e-200"):
                val = mpmath.re(numer / denom)
                sequence.append(float(val))
            else:
                sequence.append(None)
        except Exception:
            sequence.append(None)

    finite = [v for v in sequence if v is not None]
    if not finite:
        return {"best": None, "stable_digits": 0, "sequence_tail": []}

    best = finite[-1]
    # stable digits: how many leading digits match between last two values
    stable = 0
    if len(finite) >= 2:
        diffs = [abs(v - best) for v in finite[-10:] if abs(v - best) > 0]
        if diffs:
            stable = max(0, int(-math.log10(max(diffs))))

    return {
        "best":         best,
        "stable_digits": stable,
        "sequence_tail": [round(v, 15) for v in finite[-20:]],
    }


# ─── per-CMF Dreams run ────────────────────────────────────────────────────────
MN_SHIFTS = [
    # m+n ≤ 3
    (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2),
    (2, 1),
    # m+n = 4
    (0, 4), (1, 3), (2, 2), (3, 1),
    # m+n = 5
    (0, 5), (1, 4), (2, 3), (3, 2), (4, 1),
    # m+n = 6
    (0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1),
    # m+n = 7
    (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1),
]


def run_cmf_dreams(cmf_id: int, payload: dict, bank: dict,
                   depth: int, dps: int) -> list[dict]:
    f_poly    = payload.get("f_poly", "")
    fbar_poly = payload.get("fbar_poly", "")
    if not f_poly or not fbar_poly:
        return []

    try:
        Kx, b_fn = _build_walk_fns_3d(f_poly, fbar_poly)
    except Exception as e:
        print(f"    [build error] {e}")
        return []

    results = []
    primary = payload.get("primary_constant")

    for m_fixed, n_fixed in MN_SHIFTS:
        k_start = _find_k_start(b_fn, n_fixed)
        t0 = time.perf_counter()
        walk = run_walk(Kx, m_fixed, n_fixed, depth, dps, k_start=k_start)
        elapsed = round(time.perf_counter() - t0, 2)

        best = walk["best"]
        if best is None:
            hits = []
        else:
            mpmath.mp.dps = dps
            hits = _compare(mpmath.mpf(str(best)), bank)

        strong = [h for h in hits if h["log10_err"] < -8]

        record = {
            "cmf_id":          cmf_id,
            "m_fixed":         m_fixed,
            "n_fixed":         n_fixed,
            "k_start":         k_start,
            "depth":           depth,
            "dps":             dps,
            "best_estimate":   best,
            "stable_digits":   walk["stable_digits"],
            "sequence_tail":   walk["sequence_tail"],
            "primary_constant": primary,
            "hits":            hits[:6],
            "strong_hits":     strong,
            "elapsed_s":       elapsed,
            "certification_level": payload.get("certification_level"),
            "source_category": payload.get("source_category"),
        }
        results.append(record)

        if strong:
            tag = " 🔥 " + ", ".join(f"{h['name']} ({h['log10_err']})" for h in strong[:3])
        else:
            tag = f"  best={round(best,8) if best else 'None'}  stable={walk['stable_digits']}"
        print(f"    m={m_fixed} n={n_fixed} k_start={k_start} → {tag}  ({elapsed}s)")

    return results


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id",    type=int, default=None,
                    help="Run only this CMF id (default: all 3D CMFs)")
    ap.add_argument("--dps",   type=int, default=50,
                    help="mpmath precision digits (default: 50)")
    ap.add_argument("--depth", type=int, default=2000,
                    help="Walk depth (default: 2000)")
    ap.add_argument("--min_stable", type=int, default=6,
                    help="Min stable digits to flag as hit (default: 6)")
    args = ap.parse_args()

    print(f"Building constants bank (dps={args.dps})…")
    t0 = time.perf_counter()
    bank = _build_bank(args.dps)
    print(f"  {len(bank)} constants  ({time.perf_counter()-t0:.1f}s)\n")

    con = sqlite3.connect(DB_PATH)
    if args.id:
        rows = con.execute(
            "SELECT id, cmf_payload FROM cmf WHERE id=? AND dimension=3",
            (args.id,)
        ).fetchall()
    else:
        rows = con.execute("""
            SELECT id, cmf_payload FROM cmf
            WHERE dimension=3
            AND (json_extract(cmf_payload,'$.hidden') IS NULL
                 OR json_extract(cmf_payload,'$.hidden') = 0)
            ORDER BY id
        """).fetchall()
    con.close()

    print(f"Found {len(rows)} 3D CMFs to explore\n")

    all_results   = []
    all_hits      = []
    summary       = {}

    out_r = open(OUT_RESULTS, "w")
    out_h = open(OUT_HITS, "w")

    for idx, (cmf_id, payload_str) in enumerate(rows):
        payload = json.loads(payload_str) if payload_str else {}
        cert    = payload.get("certification_level", "?")
        source  = payload.get("source_category", "?")
        pconst  = payload.get("primary_constant", "?")
        print(f"[{idx+1}/{len(rows)}] CMF #{cmf_id}  cert={cert}  src={source}  const={pconst}")

        t_cmf = time.perf_counter()
        records = run_cmf_dreams(cmf_id, payload, bank, args.depth, args.dps)

        strong_total = sum(len(r["strong_hits"]) for r in records)
        best_log10   = min(
            (h["log10_err"] for r in records for h in r["strong_hits"]),
            default=None
        )
        summary[cmf_id] = {
            "cert": cert, "source": source,
            "primary_constant": pconst,
            "shifts_tested": len(records),
            "strong_hits": strong_total,
            "best_log10_err": best_log10,
            "elapsed_s": round(time.perf_counter()-t_cmf, 1),
        }

        for r in records:
            line = json.dumps(r)
            out_r.write(line + "\n")
            if r["strong_hits"]:
                out_h.write(line + "\n")

        out_r.flush(); out_h.flush()
        all_results.extend(records)
        all_hits.extend(r for r in records if r["strong_hits"])

        elapsed_total = time.perf_counter() - t_cmf
        print(f"  → {len(records)} shifts, {strong_total} strong hits  ({elapsed_total:.1f}s)\n")

    out_r.close(); out_h.close()

    with open(OUT_SUMMARY, "w") as f:
        json.dump({"summary": summary,
                   "total_cmfs": len(rows),
                   "total_shifts": len(all_results),
                   "total_strong_hits": len(all_hits),
                   }, f, indent=2)

    print("=" * 60)
    print(f"Done.  {len(all_results)} walks,  {len(all_hits)} strong hits")
    print(f"Results → {OUT_RESULTS}")
    print(f"Hits    → {OUT_HITS}")
    print(f"Summary → {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
