#!/usr/bin/env python3
"""
ramanujan_dreams.py — High-precision PSLQ constant identification scan
=======================================================================
Loads the top-180 CMF candidates from pipeline_out/manual_review_queue.csv,
runs high-precision (dps=120) matrix-product walks on their best rays, then
applies:
  1. mpmath.identify  — fast named-constant lookup via PSLQ under the hood
  2. Custom PSLQ      — integer-relation search against a rich constant basis
  3. Ratio PSLQ       — check if limit / known_constant is rational

Results written to ramanujan_out/

Usage:
    python3 ramanujan_dreams.py
"""
from __future__ import annotations

import csv, json, math, sys, time
from pathlib import Path
from typing import Optional

import mpmath as mp
import numpy as np

HERE    = Path(__file__).parent
OUT_DIR = HERE / "ramanujan_out"

# ── Precision and walk settings ───────────────────────────────────────────────
DPS_MAIN   = 100     # working precision for walks
DPS_PSLQ   = 90      # precision passed to pslq (slightly less for stability)
DEPTH_BASE = 2000    # base walk depth (steps)
DEPTH_LONG = 5000    # for slow-converging rays (delta < 10)

# ── Ray definitions (same convention as reward_engine.py) ─────────────────────
RAYS = [
    (1,0,0), (0,1,0), (0,0,1),
    (1,1,0), (1,0,1), (0,1,1),
]
RAY_NAMES = ["ex","ey","ez","exy","exz","eyz"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Eval-function reconstruction (identical to cmf_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def _G_A(p, xv, yv, zv):
    dim = p["dim"]; coords = [xv, yv, zv]
    G = np.zeros((dim, dim))
    for i, (a, b) in enumerate(p["diag"]):
        G[i, i] = a * (coords[i % 3] + b)
    for entry in p["offdiag"]:
        ii, jj, c, vi = int(entry[0]), int(entry[1]), entry[2], int(entry[3])
        G[ii, jj] = c * coords[vi]
    return G

def _Di_A(dim, axis, coord_val, dp):
    D = np.zeros((dim, dim))
    for k, (a, b) in enumerate(dp[axis]):
        D[k, k] = a * (coord_val + b + k)
    return D

def _build_fns_A(rec):
    p = dict(rec["params"]); p["dim"] = rec["dim"]
    dp = rec["d_params"]; dim = rec["dim"]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]
    fns = []
    for i, (sx,sy,sz) in enumerate(shifts):
        def mk(si=(sx,sy,sz), ax=i):
            def fn(xv, yv, zv):
                Gn = _G_A(p, xv, yv, zv)
                Gs = _G_A(p, xv+si[0], yv+si[1], zv+si[2])
                Di = _Di_A(dim, ax, [xv,yv,zv][ax], dp)
                dG = np.linalg.det(Gn)
                if abs(dG) < 1e-10: raise ValueError("singular G")
                return (Gs @ Di @ np.linalg.inv(Gn)).tolist()
            return fn
        fns.append(mk())
    return fns

def _fix_B_params(raw, dim):
    def pk(s): return tuple(int(x.strip()) for x in s.strip("()").split(","))
    L_off = {pk(k): float(v) for k,v in raw["L_off"].items()}
    U_off = {pk(k): float(v) for k,v in raw["U_off"].items()}
    D_params = [tuple(float(x) for x in pair) for pair in raw["D_params"]]
    return {"dim": dim, "L_off": L_off, "D_params": D_params, "U_off": U_off}

def _G_B(p, xv, yv, zv):
    dim = p["dim"]; coords = [xv, yv, zv]
    L = np.eye(dim)
    for (i,j), v in p["L_off"].items(): L[i,j] = v
    D = np.array([a*(coords[k%3]+b) for k,(a,b) in enumerate(p["D_params"])])
    U = np.eye(dim)
    for (i,j), v in p["U_off"].items(): U[i,j] = v
    return L @ np.diag(D) @ U

def _Di_B(dim, axis, coord_val):
    D = np.zeros((dim, dim))
    for k in range(dim): D[k,k] = coord_val + k
    return D

def _build_fns_B(rec):
    p = _fix_B_params(rec["params"], rec["dim"]); dim = rec["dim"]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]
    fns = []
    for i, (sx,sy,sz) in enumerate(shifts):
        def mk(si=(sx,sy,sz), ax=i):
            def fn(xv, yv, zv):
                Gn = _G_B(p, xv, yv, zv)
                Gs = _G_B(p, xv+si[0], yv+si[1], zv+si[2])
                Di = _Di_B(dim, ax, [xv,yv,zv][ax])
                dG = np.linalg.det(Gn)
                if abs(dG) < 1e-10: raise ValueError("singular G")
                return (Gs @ Di @ np.linalg.inv(Gn)).tolist()
            return fn
        fns.append(mk())
    return fns

def build_fns(rec):
    try:
        return _build_fns_A(rec) if rec["agent"] == "A" else _build_fns_B(rec)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  High-precision walk
# ══════════════════════════════════════════════════════════════════════════════

def hp_walk(fns, dim: int, ray: tuple, depth: int, dps: int) -> Optional[mp.mpf]:
    """
    Vector walk: v ← M_k · v,  k=1..depth.
    Axis cycles mod 3; position incremented by ray[ax].
    Returns v[0]/v[dim-1] as high-precision mpf, or None.
    """
    mp.mp.dps = dps + 20
    v = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)
    pos = [2, 2, 2]

    for step in range(1, depth + 1):
        ax = step % 3
        pos[ax] += ray[ax]
        try:
            M_raw = fns[ax](*pos)
            M = mp.matrix([[mp.mpf(str(M_raw[i][j]))
                            for j in range(dim)] for i in range(dim)])
            v = M * v
        except Exception:
            return None
        if step % 500 == 0:
            scale = max(abs(v[i]) for i in range(dim))
            if scale > mp.power(10, 40):
                v = v / scale

    denom = v[dim - 1]
    if abs(denom) < mp.power(10, -(dps - 15)):
        return None
    return v[0] / denom


def estimate_delta_hp(fns, dim, ray, dps):
    """Delta = digits of agreement between depth-300 and depth-1500 walks."""
    r1 = hp_walk(fns, dim, ray, depth=300,  dps=dps)
    r2 = hp_walk(fns, dim, ray, depth=1500, dps=dps)
    if r1 is None or r2 is None: return 0.0
    diff = abs(r2 - r1)
    if diff < mp.power(10, -dps): return float(dps)
    return min(float(dps), float(-mp.log10(diff + mp.power(10, -dps - 10))))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PSLQ constant basis and identification
# ══════════════════════════════════════════════════════════════════════════════

def build_basis(dps: int) -> dict[str, mp.mpf]:
    """Build labelled constant basis for PSLQ."""
    mp.mp.dps = dps + 30
    pi   = mp.pi
    e    = mp.e
    ln2  = mp.log(2)
    ln3  = mp.log(3)
    sq2  = mp.sqrt(2)
    sq3  = mp.sqrt(3)
    sq5  = mp.sqrt(5)
    z3   = mp.zeta(3)          # Apéry
    cat  = mp.catalan
    gam  = mp.euler            # Euler-Mascheroni γ
    phi  = (1 + sq5) / 2      # golden ratio
    sqpi = mp.sqrt(pi)

    return {
        "1":          mp.mpf(1),
        "pi":         pi,
        "pi^2":       pi**2,
        "pi^3":       pi**3,
        "pi^4":       pi**4,
        "pi/2":       pi/2,
        "pi/4":       pi/4,
        "pi/sqrt3":   pi/sq3,
        "sqrt(pi)":   sqpi,
        "e":          e,
        "e^2":        e**2,
        "log2":       ln2,
        "log3":       ln3,
        "sqrt2":      sq2,
        "sqrt3":      sq3,
        "sqrt5":      sq5,
        "phi":        phi,
        "zeta3":      z3,
        "zeta3/pi^3": z3/pi**3,
        "catalan":    cat,
        "euler_g":    gam,
        "pi^2/6":     pi**2/6,     # ζ(2)
        "pi^2/12":    pi**2/12,
        "pi^4/90":    pi**4/90,    # ζ(4)
        "ln2*pi":     ln2*pi,
        "ln2/pi":     ln2/pi,
        "sqrt2*pi":   sq2*pi,
        "pi*phi":     pi*phi,
    }


# Primary basis for PSLQ (smaller = more reliable)
PSLQ_PRIMARY = [
    "1","pi","pi^2","e","log2","log3","sqrt2","sqrt3","zeta3","catalan","euler_g","phi"
]
# Extended basis for ratio tests
PSLQ_EXTENDED = [
    "pi/2","pi/4","pi^3","pi^4","pi^2/6","pi^2/12","sqrt5",
    "sqrt(pi)","zeta3/pi^3","pi*phi","ln2*pi","ln2/pi",
]

MAX_COEFF = 50   # max integer coefficient magnitude accepted
PSLQ_TOL  = mp.power(10, -DPS_PSLQ + 15)


def run_pslq(x: mp.mpf, basis: dict, labels: list[str], dps: int) -> Optional[dict]:
    """
    Try PSLQ([x] + [basis[l] for l in labels]).
    Returns dict with formula string and coefficients if found, else None.
    """
    mp.mp.dps = dps
    vec = [x] + [basis[l] for l in labels]
    try:
        rel = mp.pslq(vec, tol=PSLQ_TOL, maxcoeff=MAX_COEFF)
    except Exception:
        return None
    if rel is None:
        return None
    coefs = [int(c) for c in rel]
    if abs(coefs[0]) == 0:
        return None
    if max(abs(c) for c in coefs) > MAX_COEFF:
        return None

    # Build formula: coefs[0]*x + coefs[1]*1 + coefs[2]*labels[0] + ... = 0
    # → x = -(coefs[1..] · basis_vals) / coefs[0]
    terms = []
    for i, lbl in enumerate(labels):
        c = coefs[i + 1]
        if c == 0: continue
        sign = "+" if c > 0 else "-"
        terms.append(f"{sign}{abs(c)}*{lbl}")
    if not terms and coefs[1] == 0:
        return None  # trivial

    # Constant term
    if coefs[1] != 0:
        terms.insert(0, f"+{coefs[1]}" if coefs[1]>0 else f"{coefs[1]}")

    formula = f"x = (-({' '.join(terms)})) / {coefs[0]}"
    # Also express as: x = rhs
    rhs_parts = []
    if coefs[1] != 0:
        rhs_parts.append(f"{-coefs[1]/coefs[0]:.6g}")
    for i, lbl in enumerate(labels):
        c = -coefs[i+1]
        a = coefs[0]
        if c == 0: continue
        frac = mp.mpf(c) / mp.mpf(a)
        if frac == int(frac):
            rhs_parts.append(f"{int(frac)}*{lbl}")
        else:
            rhs_parts.append(f"({c}/{a})*{lbl}")
    formula_clean = " + ".join(rhs_parts) if rhs_parts else "0"

    return {
        "formula": formula_clean,
        "coeffs":  coefs,
        "basis_labels": labels,
        "max_coeff": max(abs(c) for c in coefs),
    }


# Characters that indicate a garbage identify() result
_IDENT_GARBAGE = ("sqrt(0)", "log(1", "log(0", "nan", "zoo", "oo")


def identify_limit(x: mp.mpf, basis: dict, dps: int,
                  reliable_digits: float = 40.0) -> dict:
    """
    Full identification pipeline for one limit value x.
    reliable_digits: estimated number of correct decimal digits in x.
    Returns dict with all findings.
    """
    mp.mp.dps = dps
    result = {
        "x_str":        mp.nstr(x, 30),
        "identify":     None,
        "pslq_primary":  None,
        "pslq_extended": None,
        "ratio_hits":    [],
        "is_rational":   False,
        "rational_form": None,
        "hit":           False,
    }

    x_abs = abs(float(x))

    # Skip near-zero and near-inf (these give garbage results)
    if x_abs < 1e-8 or x_abs > 1e10:
        result["x_str"] = f"~0 ({x_abs:.3e})" if x_abs < 1e-8 else f"~inf ({x_abs:.3e})"
        return result

    # Practical tolerance: use min(dps-20, reliable_digits-5)
    tol_digits = max(10, min(dps - 20, int(reliable_digits) - 5))
    tol = mp.power(10, -tol_digits)

    # 1. mpmath.identify (fast, wraps PSLQ internally)
    try:
        ident = mp.identify(x, tol=tol)
        if (ident and ident != "?" and ident not in ("0", "1")
                and not any(g in str(ident) for g in _IDENT_GARBAGE)):
            result["identify"] = str(ident)
            result["hit"] = True
    except Exception:
        pass

    # 2. Rational check: is x close to p/q for small q?
    # Use a relative tolerance based on reliable digits
    rat_tol = max(1e-15, 10 ** (-min(reliable_digits - 3, tol_digits)))
    try:
        for q in range(1, 500):
            p_approx = int(round(float(x) * q))
            if p_approx == 0: continue
            err = abs(float(x) - p_approx / q)
            if err < rat_tol * abs(float(x)):
                result["is_rational"]   = True
                result["rational_form"] = f"{p_approx}/{q}"
                break
    except Exception:
        pass

    # 3. PSLQ with primary basis
    pslq_p = run_pslq(x, basis, PSLQ_PRIMARY, dps)
    if pslq_p:
        result["pslq_primary"] = pslq_p
        result["hit"] = True

    # 4. PSLQ with extended basis (only if primary missed)
    if not pslq_p:
        pslq_e = run_pslq(x, basis, PSLQ_PRIMARY + PSLQ_EXTENDED, dps)
        if pslq_e:
            result["pslq_extended"] = pslq_e
            result["hit"] = True

    # 5. Ratio tests: is x / known_constant rational (small fraction)?
    ratio_hits = []
    for lbl in PSLQ_PRIMARY + PSLQ_EXTENDED:
        if lbl == "1": continue
        c_val = basis.get(lbl)
        if c_val is None or abs(c_val) < 1e-30: continue
        try:
            r = x / c_val
            for q in range(1, 200):
                p_approx = int(round(float(r) * q))
                if p_approx == 0: continue
                err = abs(r - mp.mpf(p_approx) / q)
                if err < mp.power(10, -(dps - 25)):
                    ratio_hits.append({
                        "constant": lbl,
                        "ratio":    f"{p_approx}/{q}",
                        "formula":  f"x = ({p_approx}/{q}) * {lbl}",
                    })
                    result["hit"] = True
                    break
        except Exception:
            pass
    result["ratio_hits"] = ratio_hits

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main scan
# ══════════════════════════════════════════════════════════════════════════════

def load_top180():
    """Load candidate records for the top-180 from manual_review_queue.csv."""
    queue_csv = HERE / "pipeline_out" / "manual_review_queue.csv"
    all_jsonl  = HERE / "pipeline_out" / "candidates_all.jsonl"

    # Build index: candidate_id → record
    print("  Indexing candidates_all.jsonl …", flush=True)
    idx: dict[str, dict] = {}
    with open(all_jsonl) as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            idx[rec["candidate_id"]] = rec

    # Read top-180 from queue
    queue: list[dict] = []
    with open(queue_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["CandidateID"]
            if cid in idx:
                rec = idx[cid]
                rec["queue_rank"]         = int(row["Rank"])
                rec["queue_cluster"]      = int(row["ClusterID"])
                rec["queue_balance"]      = float(row["BalanceScore"])
                rec["stored_best_delta"]  = float(row["Delta"])
                queue.append(rec)

    print(f"  Loaded {len(queue)} candidates for scanning.", flush=True)
    return queue


def scan_candidate(rec, basis):
    """Run full Ramanujan Dreams scan on one candidate. Returns result dict."""
    dim = rec["dim"]
    agent = rec["agent"]
    cid   = rec["candidate_id"]
    stored_deltas = rec.get("deltas", [])

    fns = build_fns(rec)
    if fns is None:
        return {"candidate_id": cid, "error": "reconstruction_failed",
                "ray_results": [], "best_hit": None}

    # Pick rays: use all 6 but deprioritize rays with stored_delta < 1
    ray_order = list(range(len(RAYS)))
    if stored_deltas:
        ray_order.sort(key=lambda i: -stored_deltas[i] if i < len(stored_deltas) else 0)

    ray_results = []
    for ray_idx in ray_order:
        ray  = RAYS[ray_idx]
        name = RAY_NAMES[ray_idx]
        stored_delta = stored_deltas[ray_idx] if ray_idx < len(stored_deltas) else 0.0

        # Skip rays that didn't converge at all in the original scan
        if stored_delta < 1.0:
            ray_results.append({
                "ray": name, "ray_idx": ray_idx,
                "stored_delta": stored_delta,
                "skipped": True, "reason": "delta<1",
            })
            continue

        # Adaptive depth based on stored delta
        depth = DEPTH_LONG if stored_delta < 10 else DEPTH_BASE

        t0 = time.time()
        lim = hp_walk(fns, dim, ray, depth=depth, dps=DPS_MAIN)
        walk_t = time.time() - t0

        if lim is None:
            ray_results.append({
                "ray": name, "ray_idx": ray_idx,
                "stored_delta": stored_delta,
                "walk_failed": True,
                "walk_time_s": round(walk_t, 2),
            })
            continue

        lim_float = float(lim)
        # reliable_digits: conservative estimate from stored delta (no re-walk needed)
        reliable_digits = max(8.0, stored_delta * 0.7)

        # Run identification — calibrate tolerance to actual walk precision
        ident = identify_limit(lim, basis, dps=DPS_PSLQ,
                               reliable_digits=reliable_digits)

        ray_results.append({
            "ray":            name,
            "ray_idx":        ray_idx,
            "stored_delta":   stored_delta,
            "reliable_digits": round(reliable_digits, 1),
            "limit_float":    lim_float,
            "limit_str":      mp.nstr(lim, 40),
            "walk_time_s":    round(walk_t, 2),
            **ident,
        })

    # Best hit across all rays
    hits = [r for r in ray_results if r.get("hit")]
    best_hit = None
    if hits:
        # prefer identify match over ratio match
        for r in hits:
            if r.get("identify"):
                best_hit = {"ray": r["ray"], "formula": r["identify"],
                            "type": "identify", "limit_str": r.get("limit_str","")}
                break
        if best_hit is None:
            # prefer primary PSLQ
            for r in hits:
                if r.get("pslq_primary"):
                    best_hit = {"ray": r["ray"],
                                "formula": r["pslq_primary"]["formula"],
                                "type": "pslq_primary",
                                "limit_str": r.get("limit_str","")}
                    break
        if best_hit is None:
            for r in hits:
                rh = r.get("ratio_hits", [])
                if rh:
                    best_hit = {"ray": r["ray"],
                                "formula": rh[0]["formula"],
                                "type": "ratio",
                                "limit_str": r.get("limit_str","")}
                    break

    return {
        "candidate_id":  cid,
        "agent":         agent,
        "dim":           dim,
        "cluster":       rec.get("queue_cluster"),
        "rank":          rec.get("queue_rank"),
        "stored_best_delta": rec.get("stored_best_delta"),
        "ray_results":   ray_results,
        "best_hit":      best_hit,
        "any_hit":       best_hit is not None,
    }


def run_dreams():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 68, flush=True)
    print("  Ramanujan Dreams — PSLQ Constant Identification Scan", flush=True)
    print("=" * 68, flush=True)
    print(f"  dps={DPS_MAIN}  PSLQ_dps={DPS_PSLQ}  "
          f"depth_base={DEPTH_BASE}  depth_long={DEPTH_LONG}", flush=True)

    candidates = load_top180()
    basis      = build_basis(DPS_MAIN + 30)
    print(f"  PSLQ basis: {len(basis)} constants", flush=True)

    all_results = []
    hits        = []
    t0          = time.time()

    for idx, rec in enumerate(candidates):
        cid = rec["candidate_id"]
        print(f"\n[{idx+1:>3}/{len(candidates)}]  {cid}  "
              f"dim={rec['dim']} agent={rec['agent']} "
              f"Δ={rec.get('stored_best_delta',0):.2f}", flush=True)

        result = scan_candidate(rec, basis)
        all_results.append(result)

        if result.get("any_hit"):
            bh = result["best_hit"]
            hits.append(result)
            print(f"  ★ HIT  ray={bh['ray']}  type={bh['type']}", flush=True)
            print(f"         formula: {bh['formula']}", flush=True)
            print(f"         limit  : {bh.get('limit_str','')[:60]}", flush=True)
        else:
            # Print limits found for the best 2 rays
            valid = [r for r in result.get("ray_results",[]) if r.get("limit_str")]
            for r in valid[:2]:
                rat = r.get("rational_form","")
                tag = f"  ≈ {rat}" if rat else ""
                print(f"  ray={r['ray']}  Δ={r.get('hp_delta',0):.1f}  "
                      f"lim={r.get('limit_float',0):.10g}{tag}", flush=True)

        elapsed = time.time() - t0
        avg = elapsed / max(idx+1, 1)
        eta = avg * (len(candidates) - idx - 1)
        print(f"  {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    # ── Write outputs ──────────────────────────────────────────────────────────
    print(f"\n\nWriting outputs to {OUT_DIR} …", flush=True)

    # dreams_all.jsonl
    with open(OUT_DIR / "dreams_all.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # dreams_hits.jsonl
    with open(OUT_DIR / "dreams_hits.jsonl", "w") as f:
        for r in hits:
            f.write(json.dumps(r) + "\n")

    # dreams_summary.csv
    with open(OUT_DIR / "dreams_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rank","CandidateID","Agent","Dim","Cluster","StoredDelta",
                    "AnyHit","BestRay","HitType","Formula","LimitStr"])
        for r in all_results:
            bh = r.get("best_hit") or {}
            w.writerow([
                r.get("rank",""),
                r["candidate_id"],
                r.get("agent",""),
                r.get("dim",""),
                r.get("cluster",""),
                r.get("stored_best_delta",""),
                r.get("any_hit", False),
                bh.get("ray",""),
                bh.get("type",""),
                bh.get("formula",""),
                bh.get("limit_str","")[:60],
            ])

    # ── Print final summary ────────────────────────────────────────────────────
    total_t = time.time() - t0
    print(f"\n{'='*68}", flush=True)
    print(f"  SCAN COMPLETE  {total_t:.0f}s  |  "
          f"{len(candidates)} candidates  |  {len(hits)} hits", flush=True)
    print(f"{'='*68}", flush=True)

    if hits:
        print(f"\n  ★ HITS ({len(hits)}):", flush=True)
        for r in hits:
            bh = r["best_hit"]
            print(f"    [{r['rank']:>3}] {r['candidate_id']:35s}  "
                  f"dim={r['dim']}  Δ={r.get('stored_best_delta',0):.1f}  "
                  f"{bh['type']}  {bh['formula']}", flush=True)
    else:
        print("\n  No transcendental constant hits found in this scan.", flush=True)
        print("  (Check dreams_summary.csv for rational limits.)", flush=True)

    print(f"\n  Files: dreams_all.jsonl  dreams_hits.jsonl  dreams_summary.csv", flush=True)


if __name__ == "__main__":
    run_dreams()
