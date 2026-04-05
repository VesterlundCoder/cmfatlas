#!/usr/bin/env python3
"""
limit_engine.py — Three-pass limit extraction and asymptotic analysis
======================================================================

Implements the analysis guidelines:

  Pass A  — Coarse limit signatures for ALL 3,083 survivors
              (cheap: numpy walk N=100 & 400, dps=30)
  Pass B  — Deep extraction on ALL 513 family representatives
              (N=25,50,100,200,400,800 at dps=50 & 100; asymptotic fits)
  Pass C  — Fix broken metrics (real simplicity/proofability/dependency)
  Pass D  — Cross-dimensional persistence analysis

Outputs written to  pipeline_out/limit_engine_out/

Usage:
    python3 limit_engine.py
"""
from __future__ import annotations

import csv, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mpmath as mp
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

HERE    = Path(__file__).parent
IN_DIR  = HERE / "pipeline_out"
OUT_DIR = IN_DIR / "limit_engine_out"

# ── Walk config ───────────────────────────────────────────────────────────────
DEPTHS_COARSE  = [100, 400]          # Pass A
DEPTHS_DEEP    = [25, 50, 100, 200, 400, 800]   # Pass B
DPS_COARSE     = 30
DPS_DEEP_LO    = 50
DPS_DEEP_HI    = 100
RAYS           = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
RAY_NAMES      = ["ex","ey","ez","exy","exz","eyz"]
START_POINTS   = [[2,2,2], [3,5,4], [4,2,7]]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Eval-function reconstruction (Agent A and B)
# ══════════════════════════════════════════════════════════════════════════════

def _G_A(p, xv, yv, zv):
    dim = p["dim"]; coords = [xv, yv, zv]
    G = np.zeros((dim, dim))
    for i, (a, b) in enumerate(p["diag"]):
        G[i, i] = a * (coords[i % 3] + b)
    for entry in p["offdiag"]:
        ii, jj, c, vi = int(entry[0]),int(entry[1]),entry[2],int(entry[3])
        G[ii,jj] = c * coords[vi]
    return G

def _Di_A(dim, axis, coord_val, dp):
    D = np.zeros((dim,dim))
    for k,(a,b) in enumerate(dp[axis]): D[k,k] = a*(coord_val+b+k)
    return D

def _build_fns_A(rec):
    p = dict(rec["params"]); p["dim"] = rec["dim"]
    dp = rec["d_params"]; dim = rec["dim"]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]
    fns = []
    for i,(sx,sy,sz) in enumerate(shifts):
        def mk(si=(sx,sy,sz),ax=i):
            def fn(xv,yv,zv):
                Gn = _G_A(p,xv,yv,zv)
                Gs = _G_A(p,xv+si[0],yv+si[1],zv+si[2])
                Di = _Di_A(dim,ax,[xv,yv,zv][ax],dp)
                dG = np.linalg.det(Gn)
                if abs(dG)<1e-10: raise ValueError
                return (Gs@Di@np.linalg.inv(Gn)).tolist()
            return fn
        fns.append(mk())
    return fns

def _fix_B(raw, dim):
    def pk(s): return tuple(int(x.strip()) for x in s.strip("()").split(","))
    return {"dim":dim,
            "L_off":{pk(k):float(v) for k,v in raw["L_off"].items()},
            "D_params":[tuple(float(x) for x in pair) for pair in raw["D_params"]],
            "U_off":{pk(k):float(v) for k,v in raw["U_off"].items()}}

def _G_B(p, xv, yv, zv):
    dim=p["dim"]; coords=[xv,yv,zv]
    L=np.eye(dim)
    for (i,j),v in p["L_off"].items(): L[i,j]=v
    D=np.array([a*(coords[k%3]+b) for k,(a,b) in enumerate(p["D_params"])])
    U=np.eye(dim)
    for (i,j),v in p["U_off"].items(): U[i,j]=v
    return L@np.diag(D)@U

def _Di_B(dim, axis, coord_val):
    D=np.zeros((dim,dim))
    for k in range(dim): D[k,k]=coord_val+k
    return D

def _build_fns_B(rec):
    p=_fix_B(rec["params"],rec["dim"]); dim=rec["dim"]
    shifts=[(1,0,0),(0,1,0),(0,0,1)]
    fns=[]
    for i,(sx,sy,sz) in enumerate(shifts):
        def mk(si=(sx,sy,sz),ax=i):
            def fn(xv,yv,zv):
                Gn=_G_B(p,xv,yv,zv)
                Gs=_G_B(p,xv+si[0],yv+si[1],zv+si[2])
                Di=_Di_B(dim,ax,[xv,yv,zv][ax])
                dG=np.linalg.det(Gn)
                if abs(dG)<1e-10: raise ValueError
                return (Gs@Di@np.linalg.inv(Gn)).tolist()
            return fn
        fns.append(mk())
    return fns

def build_fns(rec):
    try:
        return _build_fns_A(rec) if rec["agent"]=="A" else _build_fns_B(rec)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Walk functions
# ══════════════════════════════════════════════════════════════════════════════

def _walk_np(fns, dim, ray, start, depth) -> Optional[float]:
    """Numpy walk. Returns v[0]/v[dim-1] or None."""
    pos = list(start)
    v = np.zeros(dim); v[0] = 1.0
    for step in range(1, depth+1):
        ax = step % 3
        pos[ax] += ray[ax]
        try:
            M = np.array(fns[ax](*pos), float)
            v = M @ v
        except Exception:
            return None
        norm = np.max(np.abs(v))
        if norm > 1e25: v /= norm
        elif norm < 1e-25: return None
    if abs(v[dim-1]) < 1e-18: return None
    return float(v[0]/v[dim-1])


def _walk_mp_one(fns, dim, ray, start, depth, dps) -> Optional[mp.mpf]:
    """mpmath walk."""
    mp.mp.dps = dps+10
    pos = list(start)
    v = mp.zeros(dim,1); v[0] = mp.mpf(1)
    for step in range(1, depth+1):
        ax = step%3
        pos[ax] += ray[ax]
        try:
            Mr = fns[ax](*pos)
            M = mp.matrix([[mp.mpf(str(Mr[r][c])) for c in range(dim)]
                           for r in range(dim)])
            v = M*v
        except Exception:
            return None
        sc = max(abs(v[i]) for i in range(dim))
        if sc > mp.power(10,30): v /= sc
        elif sc < mp.power(10,-30): return None
    if abs(v[dim-1]) < mp.power(10,-(dps-8)): return None
    return v[0]/v[dim-1]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Asymptotic model fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_asymptotic(Ns: list, vals: list) -> dict:
    """
    Given (N_i, a_Ni) pairs, fit three asymptotic models and return best.
    Models:
      A: a_N = L + c * N^{-alpha}
      B: a_N = L + c * r^N
      C: a_N = L + c * r^N * N^{-beta}
    """
    Ns  = np.array(Ns,  dtype=float)
    vs  = np.array(vals, dtype=float)
    if len(Ns) < 3 or not np.all(np.isfinite(vs)):
        return {"model":"none","L":float(vs[-1]) if len(vs)>0 else 0.0,
                "r2":0.0,"alpha":0.0,"r":0.0,"beta":0.0}

    L0 = float(vs[-1])  # crude estimate

    best = {"model":"none","L":L0,"r2":-1.0,"alpha":0.0,"r":1.0,"beta":0.0,"c":0.0}

    # ── Model A: algebraic ───────────────────────────────────────────────────
    try:
        def mA(N, L, c, alpha): return L + c * N**(-alpha)
        p0 = [L0, float(vs[0]-L0), 1.0]
        pA, _ = curve_fit(mA, Ns, vs, p0=p0, maxfev=2000,
                          bounds=([-1e6,-1e6,0.01],[ 1e6, 1e6,10]))
        res = vs - mA(Ns, *pA)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((vs - vs.mean())**2)
        r2A = 1 - ss_res/max(ss_tot, 1e-30)
        if r2A > best["r2"]:
            best.update({"model":"algebraic","L":float(pA[0]),"c":float(pA[1]),
                         "alpha":float(pA[2]),"r2":float(r2A),"r":0.0,"beta":0.0})
    except Exception:
        pass

    # ── Model B: exponential ─────────────────────────────────────────────────
    try:
        def mB(N, L, c, r): return L + c * r**N
        p0 = [L0, float(vs[0]-L0), 0.5]
        pB, _ = curve_fit(mB, Ns, vs, p0=p0, maxfev=2000,
                          bounds=([-1e6,-1e6,1e-6],[1e6,1e6,0.9999]))
        res = vs - mB(Ns, *pB)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((vs - vs.mean())**2)
        r2B = 1 - ss_res/max(ss_tot, 1e-30)
        if r2B > best["r2"]:
            best.update({"model":"exponential","L":float(pB[0]),"c":float(pB[1]),
                         "alpha":0.0,"r":float(pB[2]),"r2":float(r2B),"beta":0.0})
    except Exception:
        pass

    # ── Model C: mixed ───────────────────────────────────────────────────────
    try:
        def mC(N, L, c, r, beta): return L + c * r**N * N**(-beta)
        p0 = [L0, float(vs[0]-L0), 0.5, 0.5]
        pC, _ = curve_fit(mC, Ns, vs, p0=p0, maxfev=3000,
                          bounds=([-1e6,-1e6,1e-6,0],[1e6,1e6,0.9999,5]))
        res = vs - mC(Ns, *pC)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((vs - vs.mean())**2)
        r2C = 1 - ss_res/max(ss_tot, 1e-30)
        # penalise extra parameter
        r2C_adj = r2C - 0.05
        if r2C_adj > best["r2"]:
            best.update({"model":"mixed","L":float(pC[0]),"c":float(pC[1]),
                         "alpha":0.0,"r":float(pC[2]),"r2":float(r2C),"beta":float(pC[3])})
    except Exception:
        pass

    return best


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Real metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_real_simplicity(rec: dict) -> float:
    """Replaces the constant 0.5 simplicity."""
    deg  = rec.get("max_total_degree", -1)
    ht   = rec.get("coeff_height_max", -1)
    ops  = rec.get("expr_ops_total", -1)
    nnz  = rec.get("nnz_G", 0)
    dim  = rec.get("dim", 3)

    if deg < 0: return 0.5   # no symbolic data

    deg_pen  = deg / 8.0
    ht_pen   = math.log10(max(ht, 1)) / 3.0
    ops_pen  = ops / max(200.0, float(dim**2 * 5))
    nnz_pen  = nnz / max(float(dim**2), 1.0)

    penalty = 0.30*deg_pen + 0.30*ht_pen + 0.20*ops_pen + 0.20*nnz_pen
    return float(max(0.0, 1.0 - penalty))


def compute_real_proofability(rec: dict, limit_rational: bool = False) -> float:
    """Replaces the constant 0.3 proofability."""
    score = 0.0
    deg = rec.get("max_total_degree", -1)
    ht  = rec.get("coeff_height_max", -1)

    if deg < 0:
        score += 0.15   # no symbolic data, mild credit
    else:
        if deg <= 1: score += 0.30   # linear — easiest to prove
        elif deg <= 2: score += 0.20
        elif deg <= 4: score += 0.10

        if ht <= 5:   score += 0.20
        elif ht <= 20: score += 0.10

    # Rational limit is a strong proof signal
    if limit_rational:
        score += 0.30

    # Agent B LDU structure → Pochhammer structure → easier proof
    if rec.get("agent") == "B":
        score += 0.10

    return float(min(1.0, score))


def compute_real_dependency(rec: dict) -> dict:
    """Actual variable participation per matrix entry."""
    G_mat = rec.get("G", [])
    coords = ["x","y","z"]
    dep_per_entry = []
    for row in G_mat:
        for e in row:
            deps = [c for c in coords if c in str(e)]
            dep_per_entry.append(deps)
    n_total = max(len(dep_per_entry), 1)
    n_dep   = sum(1 for d in dep_per_entry if d)
    return {
        "dependency_real": round(n_dep / n_total, 4),
        "n_G_entries_with_coords": n_dep,
        "n_G_entries_total": n_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Cross-dimensional persistence
# ══════════════════════════════════════════════════════════════════════════════

def analyse_cross_dim(all_recs: list) -> dict[str, dict]:
    """
    For each cluster, determine if it spans multiple dimensions.
    Returns: cluster_id → {dims_present, is_persistent, mechanism_signature}
    """
    cluster_dims: dict[int, set] = defaultdict(set)
    cluster_agents: dict[int, set] = defaultdict(set)
    cluster_families: dict[int, set] = defaultdict(set)

    for rec in all_recs:
        if not rec.get("garbage_filter_pass"): continue
        cid = rec.get("cluster_id", -1)
        if cid < 0: continue
        cluster_dims[cid].add(rec["dim"])
        cluster_agents[cid].add(rec["agent"])
        cluster_families[cid].add(rec.get("family_hash",""))

    result = {}
    for cid in sorted(cluster_dims):
        dims = sorted(cluster_dims[cid])
        result[str(cid)] = {
            "dims":              dims,
            "n_dims":            len(dims),
            "is_persistent":     len(dims) >= 2,
            "agents":            sorted(cluster_agents[cid]),
            "n_families":        len(cluster_families[cid]),
            "mechanism_sig":     f"{'_'.join(sorted(cluster_agents[cid]))}_d{'_'.join(str(d) for d in dims)}",
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main engine
# ══════════════════════════════════════════════════════════════════════════════

def load_survivors():
    recs = []
    with open(IN_DIR / "candidates_all.jsonl") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return [r for r in recs if r.get("garbage_filter_pass")]


def pick_family_reps(survivors: list) -> list:
    """One representative per family_hash (highest best_delta)."""
    best: dict[str, dict] = {}
    for rec in survivors:
        fh = rec.get("family_hash","_none")
        if fh not in best or rec.get("best_delta",0) > best[fh].get("best_delta",0):
            best[fh] = rec
    return list(best.values())


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("="*66, flush=True)
    print("  Limit Engine — multi-pass asymptotic analysis", flush=True)
    print("="*66, flush=True)

    print("\nLoading survivors …", flush=True)
    survivors = load_survivors()
    family_reps = pick_family_reps(survivors)
    print(f"  {len(survivors)} survivors  |  {len(family_reps)} family representatives",
          flush=True)

    # ── Load all records for cross-dim analysis ───────────────────────────────
    all_recs = []
    with open(IN_DIR / "candidates_all.jsonl") as f:
        for line in f:
            if line.strip(): all_recs.append(json.loads(line))

    # ══════════════════════════════════════════════════════════════════════════
    # Pass A — Coarse limit signatures for ALL survivors
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nPass A — Coarse limits for {len(survivors)} survivors …", flush=True)
    t_a = time.time()

    for idx, rec in enumerate(survivors):
        if idx % 200 == 0:
            print(f"  [{idx:>4}/{len(survivors)}]  {time.time()-t_a:.0f}s", flush=True)

        fns = build_fns(rec)
        dim = rec["dim"]
        ray_limits = {}   # ray_name → limit_at_400

        for ri, (ray, rname) in enumerate(zip(RAYS, RAY_NAMES)):
            stored_d = rec.get("deltas", [0]*6)
            stored_d_val = stored_d[ri] if ri < len(stored_d) else 0.0
            if fns is None or stored_d_val < 0.5:
                continue
            start = START_POINTS[0]
            l100 = _walk_np(fns, dim, ray, start, depth=DEPTHS_COARSE[0])
            l400 = _walk_np(fns, dim, ray, start, depth=DEPTHS_COARSE[1])
            if l100 is not None and l400 is not None:
                ray_limits[rname] = l400

        # Coarse statistics
        lim_vals = [v for v in ray_limits.values()
                    if math.isfinite(v) and abs(v) > 1e-8 and abs(v) < 1e8]

        if lim_vals:
            lim_mean = float(np.mean(lim_vals))
            lim_var  = float(np.var(lim_vals))
            # Asymptotic class: compare variance to mean magnitude
            rel_var = lim_var / (abs(lim_mean)**2 + 1e-10)
            asym_class = "stable" if rel_var < 0.01 else "unstable"
        else:
            lim_mean = 0.0; lim_var = 0.0; asym_class = "zero_or_nan"

        # Constant fingerprint (rounded to 3 sig figs, hash)
        sig_vals = tuple(round(v, 3) for v in sorted(lim_vals)[:4])
        lim_sig = hashlib.md5(str(sig_vals).encode()).hexdigest()[:12]

        rec["limit_estimate_coarse"]     = round(lim_mean, 8)
        rec["ray_limits_coarse"]         = {k: round(v,8) for k,v in ray_limits.items()}
        rec["ray_variance_coarse"]       = round(lim_var, 10)
        rec["asymptotic_class_coarse"]   = asym_class
        rec["limit_signature_hash"]      = lim_sig
        rec["n_valid_rays_coarse"]        = len(lim_vals)

        # Fix broken metrics
        rec["simplicity_real"]  = round(compute_real_simplicity(rec), 4)
        rec["proofability_real"] = round(compute_real_proofability(rec), 4)
        rec.update(compute_real_dependency(rec))

    print(f"  Pass A done in {time.time()-t_a:.0f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Pass B — Deep extraction on family representatives
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nPass B — Deep extraction on {len(family_reps)} family reps …", flush=True)
    t_b = time.time()

    # Build lookup for Pass A data
    surv_by_id = {r["candidate_id"]: r for r in survivors}

    for idx, rec in enumerate(family_reps):
        if idx % 30 == 0:
            print(f"  [{idx:>3}/{len(family_reps)}]  {time.time()-t_b:.0f}s", flush=True)

        fns  = build_fns(rec)
        dim  = rec["dim"]
        cid  = rec["candidate_id"]
        stored_deltas = rec.get("deltas", [0]*6)

        # Best ray index
        best_ray_idx = max(range(len(RAYS)),
                           key=lambda i: stored_deltas[i] if i < len(stored_deltas) else 0)
        best_ray     = RAYS[best_ray_idx]

        # depth-series walk at dps=50 (main precision)
        depth_series_50: list[tuple[int, float]] = []
        if fns is not None:
            for N in DEPTHS_DEEP:
                lim = _walk_mp_one(fns, dim, best_ray,
                                   START_POINTS[0], N, DPS_DEEP_LO)
                if lim is not None and abs(float(lim)) > 1e-9:
                    depth_series_50.append((N, float(lim)))

        # depth-series at dps=100 (high precision)
        depth_series_100: list[tuple[int, float]] = []
        if fns is not None and len(depth_series_50) >= 3:
            for N in [100, 200, 400, 800]:
                lim = _walk_mp_one(fns, dim, best_ray,
                                   START_POINTS[0], N, DPS_DEEP_HI)
                if lim is not None and abs(float(lim)) > 1e-9:
                    depth_series_100.append((N, float(lim)))

        # Asymptotic fit (use dps=50 series)
        asym = {"model":"none","L":0.0,"r2":0.0,"alpha":0.0,"r":0.0,"beta":0.0}
        if len(depth_series_50) >= 3:
            Ns  = [x[0] for x in depth_series_50]
            vs  = [x[1] for x in depth_series_50]
            asym = fit_asymptotic(Ns, vs)

        # Precision variance (compare dps=50 vs dps=100 at N=400)
        prec_var = 0.0
        lim_50_400  = next((v for N,v in depth_series_50  if N==400), None)
        lim_100_400 = next((v for N,v in depth_series_100 if N==400), None)
        if lim_50_400 is not None and lim_100_400 is not None:
            prec_var = abs(lim_50_400 - lim_100_400)

        # Start-point variance (compare 3 start points at N=200)
        sp_lims = []
        if fns is not None:
            for sp in START_POINTS:
                lim = _walk_mp_one(fns, dim, best_ray, sp, 200, DPS_DEEP_LO)
                if lim is not None and abs(float(lim)) > 1e-9:
                    sp_lims.append(float(lim))
        sp_var = float(np.var(sp_lims)) if len(sp_lims) >= 2 else 0.0

        # Ray variance at N=200, dps=50
        ray_lims_deep = {}
        if fns is not None:
            for ri, (ray, rname) in enumerate(zip(RAYS, RAY_NAMES)):
                d_stored = stored_deltas[ri] if ri < len(stored_deltas) else 0.0
                if d_stored < 0.5: continue
                lim = _walk_mp_one(fns, dim, ray, START_POINTS[0], 200, DPS_DEEP_LO)
                if lim is not None and abs(float(lim)) > 1e-9:
                    ray_lims_deep[rname] = float(lim)
        ray_v_vals = list(ray_lims_deep.values())
        ray_var_deep = float(np.var(ray_v_vals)) if len(ray_v_vals) >= 2 else 0.0

        # Extract individual depth estimates
        def _lim_at(N, series): return next((v for n,v in series if n==N), None)

        # Annotate record
        rec.update({
            "limit_estimate_100":    _lim_at(100, depth_series_50),
            "limit_estimate_200":    _lim_at(200, depth_series_50),
            "limit_estimate_400":    _lim_at(400, depth_series_50),
            "limit_estimate_800":    _lim_at(800, depth_series_50),
            "limit_estimate_deep":   asym.get("L"),
            "limit_error":           round(prec_var, 12),
            "ray_limit_variance":    round(ray_var_deep, 12),
            "startpoint_variance":   round(sp_var, 12),
            "asymptotic_model_best": asym.get("model","none"),
            "asymptotic_alpha":      round(asym.get("alpha",0.0), 4),
            "asymptotic_r":          round(asym.get("r",0.0), 6),
            "asymptotic_beta":       round(asym.get("beta",0.0), 4),
            "asymptotic_r2":         round(asym.get("r2",0.0), 4),
            "asymptotic_L":          round(asym.get("L",0.0), 12),
            "depth_series_50":       depth_series_50,
            "ray_limits_deep":       ray_lims_deep,
            "precision_variance":    round(prec_var, 12),
        })

    print(f"  Pass B done in {time.time()-t_b:.0f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Pass C — Cross-dimensional persistence
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPass C — Cross-dimensional persistence …", flush=True)
    cross = analyse_cross_dim(all_recs)

    # Annotate each survivor
    for rec in survivors:
        cid = str(rec.get("cluster_id", -1))
        cd = cross.get(cid, {})
        rec["dimension_persistence_flag"]   = cd.get("is_persistent", False)
        rec["dims_in_cluster"]              = cd.get("dims", [rec["dim"]])
        rec["mechanism_signature"]          = cd.get("mechanism_sig","")

    # Cross-dim clusters
    persistent_clusters = {cid: v for cid,v in cross.items() if v["is_persistent"]}
    print(f"  {len(persistent_clusters)} cross-dimensional clusters found.", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Pass D — Limit-based re-clustering of family reps
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPass D — Limit-based re-clustering …", flush=True)
    reps_with_lim = [r for r in family_reps if r.get("limit_estimate_deep") is not None]

    if len(reps_with_lim) >= 5:
        def lim_fvec(r):
            L = r.get("limit_estimate_deep", 0.0) or 0.0
            err = r.get("limit_error", 0.0) or 0.0
            rv  = r.get("ray_limit_variance", 0.0) or 0.0
            sp  = r.get("startpoint_variance", 0.0) or 0.0
            alpha = r.get("asymptotic_alpha", 0.0) or 0.0
            rval  = r.get("asymptotic_r", 0.0) or 0.0
            return [L, math.log10(abs(L)+1e-30), err, rv, sp, alpha, rval,
                    1.0 if r.get("asymptotic_model_best")=="algebraic" else 0.0,
                    1.0 if r.get("asymptotic_model_best")=="exponential" else 0.0]

        X = np.nan_to_num(
            np.array([lim_fvec(r) for r in reps_with_lim], float),
            nan=0.0, posinf=1.0, neginf=0.0)
        Xs = StandardScaler().fit_transform(X)
        Z  = linkage(Xs, method="ward")
        n_lim_clust = max(3, min(30, len(reps_with_lim) // 10))
        lbls = fcluster(Z, t=n_lim_clust, criterion="maxclust").tolist()
        for rec, lbl in zip(reps_with_lim, lbls):
            rec["limit_cluster_id"] = int(lbl)
        print(f"  {n_lim_clust} limit-space clusters.", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Write outputs
    # ══════════════════════════════════════════════════════════════════════════
    print("\nWriting outputs …", flush=True)

    # survivors_enriched.jsonl
    with open(OUT_DIR / "survivors_enriched.jsonl", "w") as f:
        for rec in survivors:
            f.write(json.dumps({k:v for k,v in rec.items()
                                if k not in ("X0","X1","X2")}) + "\n")

    # family_reps_deep.jsonl
    with open(OUT_DIR / "family_reps_deep.jsonl", "w") as f:
        for rec in family_reps:
            f.write(json.dumps({k:v for k,v in rec.items()
                                if k not in ("X0","X1","X2")}) + "\n")

    # cross_dim_clusters.json
    with open(OUT_DIR / "cross_dim_clusters.json", "w") as f:
        json.dump(cross, f, indent=2)

    # enriched_features.csv (flat table)
    CSV_FIELDS = [
        "candidate_id","agent","dim","source_file","fingerprint","family_hash",
        "garbage_filter_pass","cluster_id",
        "best_delta","ray_stability","simplicity_real","proofability_real",
        "dependency_real","discovery_score","proof_score","balance_score",
        "nnz_G","max_total_degree","coeff_height_max","expr_ops_total",
        "limit_estimate_coarse","ray_variance_coarse","asymptotic_class_coarse",
        "limit_signature_hash","n_valid_rays_coarse",
        "dimension_persistence_flag","mechanism_signature",
        "novelty_score","near_duplicate_family","timestamp",
    ]
    with open(OUT_DIR / "enriched_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in survivors:
            w.writerow({k: rec.get(k,"") for k in CSV_FIELDS})

    # family_reps_deep.csv
    DEEP_FIELDS = [
        "candidate_id","agent","dim","family_hash","cluster_id",
        "best_delta","simplicity_real","proofability_real",
        "limit_estimate_100","limit_estimate_200","limit_estimate_400",
        "limit_estimate_800","limit_estimate_deep","limit_error",
        "ray_limit_variance","startpoint_variance","precision_variance",
        "asymptotic_model_best","asymptotic_alpha","asymptotic_r",
        "asymptotic_beta","asymptotic_r2","asymptotic_L",
        "dimension_persistence_flag","mechanism_signature","limit_cluster_id",
    ]
    with open(OUT_DIR / "family_reps_deep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEEP_FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in family_reps:
            w.writerow({k: rec.get(k,"") for k in DEEP_FIELDS})

    # cross_dim_summary.csv
    with open(OUT_DIR / "cross_dim_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ClusterID","Dims","N_Dims","Persistent","Agents",
                    "N_Families","MechanismSig"])
        for cid, v in sorted(cross.items(), key=lambda x: -x[1]["n_dims"]):
            w.writerow([cid, ",".join(str(d) for d in v["dims"]),
                        v["n_dims"], v["is_persistent"],
                        ",".join(v["agents"]), v["n_families"],
                        v["mechanism_sig"]])

    # ── Print summary ─────────────────────────────────────────────────────────
    total_t = time.time() - t0
    print(f"\n{'='*66}", flush=True)
    print(f"  LIMIT ENGINE COMPLETE  {total_t:.0f}s", flush=True)
    print(f"{'='*66}", flush=True)

    # Real metrics vs old
    old_simp  = 0.5
    old_proof = 0.3
    new_simp  = float(np.mean([r.get("simplicity_real",0.5) for r in survivors]))
    new_proof = float(np.mean([r.get("proofability_real",0.3) for r in survivors]))
    print(f"\n  Metric fix:")
    print(f"    simplicity:   {old_simp:.2f} (constant) → {new_simp:.3f} (mean, real)")
    print(f"    proofability: {old_proof:.2f} (constant) → {new_proof:.3f} (mean, real)")

    # Cross-dim
    print(f"\n  Cross-dimensional clusters: {len(persistent_clusters)}")
    top_cross = sorted(persistent_clusters.items(),
                       key=lambda x: -x[1]["n_dims"])[:10]
    for cid, v in top_cross:
        print(f"    cluster {cid:>3}  dims={v['dims']}  agents={v['agents']}  "
              f"families={v['n_families']}")

    # Asymptotic models on family reps
    model_counts: dict[str,int] = defaultdict(int)
    for r in family_reps:
        model_counts[r.get("asymptotic_model_best","none")] += 1
    print(f"\n  Asymptotic model distribution (family reps):")
    for m, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:<15} {cnt:>5}")

    # Limit stability (Pass B reps with low error)
    stable = [r for r in family_reps
              if r.get("limit_error",1) is not None
              and r.get("limit_error",1) < 1e-10
              and r.get("asymptotic_r2",0) > 0.99]
    print(f"\n  Family reps with stable limit (error<1e-10, R²>0.99): {len(stable)}")
    for r in sorted(stable, key=lambda x: -x.get("asymptotic_r2",0))[:10]:
        print(f"    {r['candidate_id']:35s}  dim={r['dim']}  "
              f"L={r.get('asymptotic_L',0):.8g}  "
              f"model={r.get('asymptotic_model_best')}  "
              f"R²={r.get('asymptotic_r2',0):.4f}")

    print(f"\n  Outputs: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    run()
