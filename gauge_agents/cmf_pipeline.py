#!/usr/bin/env python3
"""
cmf_pipeline.py — Full post-processing pipeline for CMF candidates
===================================================================

Pipeline:
  Step 1 : Garbage filter — path independence, poles, trivial collapse,
           exact duplicates, kernel coordinate-dependence.
           All candidates kept; failures annotated with reasons.
  Step 2 : Symbolic feature extraction on survivors (sympy, lazily).
  Step 3 : Triple scoring — discovery, proof, balance.
  Step 4 : Family signatures + Ward hierarchical clustering.
  Step 5 : Report generation (JSONL + CSV).

Usage:
    python3 cmf_pipeline.py

Output files written to  pipeline_out/
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import sympy as sp
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
OUT_DIR = HERE / "pipeline_out"

# ── Thresholds ─────────────────────────────────────────────────────────────────
DIMS_POLE_BOX  = {3: 12, 4: 10, 5: 8}
N_PATH_CHECK   = 30          # random test points for path-independence
N_POLE_THOROUGH = 100        # points for thorough pole scan
PATH_TOL        = 1e-5       # relative error tolerance for flatness
TRIVIAL_REL_TOL = 0.01       # X_i variance-ratio below this → trivial collapse


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Eval-function reconstruction from stored params
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


def _build_eval_fns_A(rec):
    p = dict(rec["params"]); p["dim"] = rec["dim"]
    dp = rec["d_params"]; dim = rec["dim"]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]
    fns = []
    for i, (sx,sy,sz) in enumerate(shifts):
        def mk(si=(sx,sy,sz), ax=i):
            def fn(xv,yv,zv):
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
    def parse_key(s):
        return tuple(int(x.strip()) for x in s.strip("()").split(","))
    L_off = {parse_key(k): float(v) for k,v in raw["L_off"].items()}
    U_off = {parse_key(k): float(v) for k,v in raw["U_off"].items()}
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


def _build_eval_fns_B(rec):
    p = _fix_B_params(rec["params"], rec["dim"]); dim = rec["dim"]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]
    fns = []
    for i, (sx,sy,sz) in enumerate(shifts):
        def mk(si=(sx,sy,sz), ax=i):
            def fn(xv,yv,zv):
                Gn = _G_B(p, xv, yv, zv)
                Gs = _G_B(p, xv+si[0], yv+si[1], zv+si[2])
                Di = _Di_B(dim, ax, [xv,yv,zv][ax])
                dG = np.linalg.det(Gn)
                if abs(dG) < 1e-10: raise ValueError("singular G")
                return (Gs @ Di @ np.linalg.inv(Gn)).tolist()
            return fn
        fns.append(mk())
    return fns


def build_eval_fns(rec) -> Optional[list]:
    try:
        return _build_eval_fns_A(rec) if rec["agent"] == "A" else _build_eval_fns_B(rec)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Garbage-filter checks
# ══════════════════════════════════════════════════════════════════════════════

def check_path_independence(fns, dim):
    """Returns (pass: bool, max_rel_err: float).
    Verifies X_i(n+e_j)·X_j(n) = X_j(n+e_i)·X_i(n) at N_PATH_CHECK points."""
    rng = np.random.default_rng(42)
    box = DIMS_POLE_BOX.get(dim, 10)
    max_err = 0.0
    n_ax = min(3, dim)
    for _ in range(N_PATH_CHECK):
        pos = list(map(int, rng.integers(2, box, size=3).tolist()))
        for i in range(n_ax):
            for j in range(i+1, n_ax):
                try:
                    Xi_n   = np.array(fns[i](*pos), float)
                    Xj_n   = np.array(fns[j](*pos), float)
                    pei = list(pos); pei[i] += 1
                    pej = list(pos); pej[j] += 1
                    Xj_nei = np.array(fns[j](*pei), float)
                    Xi_nej = np.array(fns[i](*pej), float)
                    lhs = Xi_nej @ Xj_n
                    rhs = Xj_nei @ Xi_n
                    denom = max(np.max(np.abs(lhs)), 1e-10)
                    err = np.max(np.abs(lhs - rhs)) / denom
                    max_err = max(max_err, float(err))
                except Exception:
                    return False, float("inf")
    return max_err < PATH_TOL, max_err


def check_poles_thorough(fns, dim):
    """Returns (pass: bool, bad_count: int, min_abs_det: float).
    Samples a uniform grid in [1, box_max] and checks finiteness + det."""
    rng = np.random.default_rng(99)
    box = DIMS_POLE_BOX.get(dim, 10)
    pts = rng.integers(1, box+1, size=(N_POLE_THOROUGH, 3)).tolist()
    bad = 0; min_det = float("inf")
    for pos in pts:
        for ax in range(min(3, dim)):
            try:
                M = np.array(fns[ax](*pos), dtype=complex)
                if not np.all(np.isfinite(M)):
                    bad += 1; continue
                d = abs(float(np.linalg.det(M)))
                min_det = min(min_det, d)
                if d < 1e-20: bad += 1
            except Exception:
                bad += 1
    return bad == 0, bad, (min_det if math.isfinite(min_det) else 0.0)


def check_trivial_collapse(fns, dim):
    """Returns (collapsed: bool, variance_ratio: float).
    Collapses if X_0(n,2,2) does not vary as n changes 1→10."""
    try:
        mats = [np.array(fns[0](n, 2, 2), float) for n in range(1, 11)]
        base = mats[0]; denom = max(np.max(np.abs(base)), 1e-10)
        spread = max(np.max(np.abs(m - base)) / denom for m in mats[1:])
        return float(spread) < TRIVIAL_REL_TOL, float(spread)
    except Exception:
        return False, 999.0   # can't evaluate → not trivially collapsed


def check_kernel_dep(rec):
    """Returns (ok: bool, dep_count: int).
    At least (dim-1) diagonal entries must have non-zero coordinate slope."""
    if rec["agent"] == "B":
        return True, rec["dim"] * 3   # canonical D_i always coordinate-dependent
    dp = rec["d_params"]
    dep = sum(1 for axis in dp for (a, b) in axis if abs(float(a)) > 1e-10)
    return dep >= max(2, rec["dim"] - 1), dep


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def structural_features(rec):
    """Fast features from stored strings — no sympy, no eval_fns."""
    dim   = rec["dim"]
    G_mat = rec.get("G", [])

    nnz_G = sum(1 for row in G_mat for e in row
                if str(e).strip() not in ("0", "0.0", ""))

    # Which coordinates appear in each kernel axis
    dep_sig = []
    for dk in ["D0", "D1", "D2"]:
        for coord in ("x", "y", "z"):
            dep_sig.append(int(any(coord in str(e) for e in rec.get(dk, []))))

    # Sign of leading coefficient in each kernel entry
    k_signs = []
    for dk in ["D0", "D1", "D2"]:
        for e in rec.get(dk, []):
            s = str(e).strip()
            k_signs.append(-1 if s.startswith("-") else (0 if s == "0" else 1))

    ldu_sig = None
    if rec["agent"] == "B":
        p = rec["params"]
        ldu_sig = f"L{len(p.get('L_off',{}))}U{len(p.get('U_off',{}))}"

    ops_proxy = sum(len(str(e)) for row in G_mat for e in row)

    return {
        "nnz_G":                          nnz_G,
        "variable_dependency_signature":  dep_sig,
        "kernel_sign_pattern":            k_signs,
        "ldu_signature":                  ldu_sig,
        "expr_len_proxy":                 ops_proxy,
    }


def sympy_features(rec):
    """Slower sympy-based symbolic depth — called only for survivors."""
    x, y, z = sp.symbols("x y z", integer=True)
    locs = {"x": x, "y": y, "z": z}
    G_mat = rec.get("G", [])
    degrees, heights, ops = [], [], []
    for row in G_mat:
        for e_str in row:
            try:
                e = sp.sympify(str(e_str), locals=locs)
                e_exp = sp.expand(e)
                ops.append(int(sp.count_ops(e_exp, visual=False)))
                n_e, d_e = sp.fraction(sp.together(e))
                for poly_e in (n_e, d_e):
                    free = sorted(poly_e.free_symbols, key=lambda s: s.name)
                    try:
                        p = sp.Poly(sp.expand(poly_e), *free) if free else sp.Poly(sp.expand(poly_e))
                        degrees.append(p.total_degree())
                        for c in p.coeffs():
                            try: heights.append(abs(int(c)))
                            except: pass
                    except Exception:
                        degrees.append(3); heights.append(3)
            except Exception:
                degrees.append(3); heights.append(3); ops.append(20)
    return {
        "max_total_degree":  int(max(degrees))        if degrees else 0,
        "max_den_degree":    int(max(degrees[1::2]))   if len(degrees) >= 2 else 0,
        "max_num_degree":    int(max(degrees[0::2]))   if degrees else 0,
        "coeff_height_max":  int(max(heights))         if heights else 0,
        "coeff_height_mean": float(np.mean(heights))   if heights else 0.0,
        "expr_ops_total":    int(sum(ops)),
        "expr_ops_mean":     float(np.mean(ops))       if ops else 0.0,
    }


def family_signature(rec, sf):
    """Soft hash grouping near-duplicates by structure."""
    k_signs = tuple(sf["kernel_sign_pattern"][:rec["dim"] * 3])
    raw = (f"{rec['agent']}|{rec['dim']}|{sf['nnz_G']}|{k_signs}"
           f"|{sf['ldu_signature'] or 'none'}")
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Scoring
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(rec, novelty=0.5):
    conv  = min(1.0, max(0.0, rec.get("best_delta", 0.0) / 60.0))
    stab  = min(1.0, max(0.0, rec.get("ray_stability", 0.0)))
    simp  = min(1.0, max(0.0, rec.get("simplicity",  0.5)))
    proof = min(1.0, max(0.0, rec.get("proofability", 0.3)))
    ident = min(1.0, max(0.0, rec.get("identifiability", 0.0)))
    nov   = min(1.0, max(0.0, novelty))

    discovery = 0.30*conv + 0.20*stab + 0.20*nov  + 0.10*ident + 0.10*simp + 0.10*proof
    proof_sc  = 0.30*proof + 0.25*simp + 0.20*stab + 0.15*conv  + 0.10*ident
    balance   = 0.25*conv  + 0.20*stab + 0.20*simp + 0.20*proof + 0.15*nov
    return {
        "discovery_score": round(discovery, 4),
        "proof_score":     round(proof_sc,  4),
        "balance_score":   round(balance,   4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Clustering
# ══════════════════════════════════════════════════════════════════════════════

def cluster_survivors(survivors):
    if len(survivors) < 4:
        return [1] * len(survivors)

    def fvec(c):
        ldu_d = 0.0
        if c.get("ldu_signature"):
            nums = re.findall(r'\d+', c["ldu_signature"])
            ldu_d = sum(int(n) for n in nums) / max(c["dim"]**2, 1)
        return [
            (c["dim"] - 3) / 2.0,
            0.0 if c["agent"] == "A" else 1.0,
            min(1.0, c.get("best_delta", 0) / 60.0),
            c.get("ray_stability", 0.0),
            c.get("simplicity", 0.5),
            c.get("proofability", 0.3),
            c.get("identifiability", 0.0),
            c.get("nnz_G", 0) / max(c["dim"]**2, 1),
            ldu_d,
            c.get("kernel_dep_count", 0) / max(c["dim"]*3, 1),
            c.get("dfinite_score", 0.0),
        ]

    X = np.nan_to_num(np.array([fvec(c) for c in survivors], float),
                      nan=0.0, posinf=1.0, neginf=0.0)
    X_s = StandardScaler().fit_transform(X)
    Z   = linkage(X_s, method="ward")
    n_clust = max(5, min(60, len(survivors) // 15))
    return fcluster(Z, t=n_clust, criterion="maxclust").tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_candidates():
    recs = []
    for sf in sorted(HERE.glob("store_*.jsonl")):
        for idx, line in enumerate(sf.read_text().splitlines()):
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                rec["candidate_id"]  = f"{sf.stem}_{idx:04d}"
                rec["source_file"]   = sf.name
                rec.setdefault("dfinite_score", 0.0)
                recs.append(rec)
            except Exception:
                pass
    return recs


def _slim(rec):
    """Drop heavy symbolic fields for JSONL output."""
    return {k: v for k, v in rec.items() if k not in ("X0","X1","X2")}


CSV_FIELDS = [
    "candidate_id","agent","dim","source_file","fingerprint","family_hash",
    "garbage_filter_pass","garbage_reasons",
    "path_independence_pass","path_independence_max_err",
    "invertibility_pass","pole_free_pass","bad_points_count","min_abs_det_on_test_box",
    "trivial_collapse_flag","variance_ratio",
    "kernel_dep_pass","kernel_dep_count","near_duplicate_family",
    "best_delta","conv_rate","ray_stability","identifiability",
    "simplicity","proofability","dfinite_score","score",
    "nnz_G","ldu_signature","expr_len_proxy",
    "max_total_degree","max_den_degree","max_num_degree",
    "coeff_height_max","coeff_height_mean","expr_ops_total","expr_ops_mean",
    "discovery_score","proof_score","balance_score",
    "cluster_id","novelty_score","timestamp",
]


def run_pipeline():
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading candidates …", flush=True)
    candidates = load_candidates()
    print(f"  {len(candidates)} candidates loaded from "
          f"{len(list(HERE.glob('store_*.jsonl')))} store files.", flush=True)

    # ── Step 1 : Garbage filter ───────────────────────────────────────────────
    print("\nStep 1 — Garbage filter …", flush=True)
    t0 = time.time()
    seen_fps     = set()
    seen_fam_sig = set()
    reason_tally: dict[str, int] = {}
    all_recs: list[dict] = []

    for idx, rec in enumerate(candidates):
        if idx % 200 == 0:
            print(f"  [{idx:>4}/{len(candidates)}]  {time.time()-t0:.0f}s elapsed", flush=True)

        reasons: list[str] = []

        # Fast structural features (no eval)
        sf = structural_features(rec)
        rec.update(sf)

        # Exact duplicate
        fp = rec.get("fingerprint", "")
        if fp and fp in seen_fps:
            reasons.append("exact_duplicate")

        # Eval-function reconstruction
        fns = build_eval_fns(rec)
        if fns is None:
            reasons.append("reconstruction_failed")

        # ── Checks that need fns ──────────────────────────────────────────────
        path_pass = False; path_err = float("inf")
        inv_pass  = False; bad_pts  = 999; min_det = 0.0
        collapsed = False; var_ratio = 0.0

        if fns is not None:
            # Path independence
            try:
                path_pass, path_err = check_path_independence(fns, rec["dim"])
                if not path_pass:
                    reasons.append("path_independence_fail")
            except Exception:
                path_pass, path_err = False, float("inf")
                reasons.append("path_independence_error")

            # Poles / invertibility
            try:
                inv_pass, bad_pts, min_det = check_poles_thorough(fns, rec["dim"])
                if not inv_pass:
                    reasons.append("pole_or_singularity_fail")
            except Exception:
                inv_pass, bad_pts, min_det = False, 999, 0.0
                reasons.append("pole_check_error")

            # Trivial collapse
            try:
                collapsed, var_ratio = check_trivial_collapse(fns, rec["dim"])
                if collapsed:
                    reasons.append("trivial_collapse")
            except Exception:
                collapsed, var_ratio = False, 0.0

        # Kernel coordinate-dependence (no fns needed)
        dep_ok, dep_count = check_kernel_dep(rec)
        if not dep_ok:
            reasons.append("insufficient_kernel_coord_dep")

        # Family signature
        fam_sig = family_signature(rec, sf)
        rec["family_hash"] = fam_sig
        near_dup = fam_sig in seen_fam_sig

        # Annotate
        rec["garbage_filter_pass"]       = len(reasons) == 0
        rec["garbage_reasons"]           = reasons
        rec["path_independence_pass"]    = path_pass
        rec["path_independence_max_err"] = round(path_err, 8) if math.isfinite(path_err) else 1e9
        rec["invertibility_pass"]        = inv_pass
        rec["pole_free_pass"]            = inv_pass
        rec["bad_points_count"]          = bad_pts
        rec["min_abs_det_on_test_box"]   = round(min_det, 12)
        rec["trivial_collapse_flag"]     = collapsed
        rec["variance_ratio"]            = round(var_ratio, 6)
        rec["kernel_dep_pass"]           = dep_ok
        rec["kernel_dep_count"]          = dep_count
        rec["near_duplicate_family"]     = near_dup
        rec["cluster_id"]                = -1
        rec["novelty_score"]             = 0.0

        if not reasons:
            seen_fps.add(fp)
            seen_fam_sig.add(fam_sig)

        for r in reasons:
            reason_tally[r] = reason_tally.get(r, 0) + 1

        all_recs.append(rec)

    survivors = [c for c in all_recs if c["garbage_filter_pass"]]
    rejected  = [c for c in all_recs if not c["garbage_filter_pass"]]
    print(f"\n  Total: {len(all_recs)}  |  Survived: {len(survivors)}  "
          f"|  Rejected: {len(rejected)}", flush=True)
    print(f"  Reasons: {reason_tally}", flush=True)

    # ── Step 2 : Symbolic features on survivors ───────────────────────────────
    print(f"\nStep 2 — Symbolic features on {len(survivors)} survivors …", flush=True)
    t1 = time.time()
    for idx, rec in enumerate(survivors):
        if idx % 50 == 0:
            print(f"  [{idx:>4}/{len(survivors)}]  {time.time()-t1:.0f}s", flush=True)
        rec.update(sympy_features(rec))
    gc.collect()

    # ── Step 3 : Scoring ──────────────────────────────────────────────────────
    print("\nStep 3 — Scoring …", flush=True)
    for rec in all_recs:
        rec.update(compute_scores(rec))

    # ── Step 4 : Clustering + novelty ────────────────────────────────────────
    print(f"\nStep 4 — Clustering {len(survivors)} survivors …", flush=True)
    if len(survivors) >= 4:
        labels     = cluster_survivors(survivors)
        clust_size = Counter(labels)
        for rec, lbl in zip(survivors, labels):
            rec["cluster_id"]   = int(lbl)
            rec["novelty_score"] = round(1.0 / (1.0 + math.log1p(clust_size[lbl])), 4)
        # Re-score with novelty factored in
        for rec in survivors:
            rec.update(compute_scores(rec, novelty=rec["novelty_score"]))
    else:
        for i, rec in enumerate(survivors):
            rec["cluster_id"] = 1; rec["novelty_score"] = 1.0

    n_clusters = len(set(c["cluster_id"] for c in survivors))
    print(f"  {n_clusters} clusters found.", flush=True)

    # ── Step 5 : Write outputs ────────────────────────────────────────────────
    print("\nStep 5 — Writing output files …", flush=True)

    # candidates_all.jsonl
    with open(OUT_DIR / "candidates_all.jsonl", "w") as f:
        for rec in all_recs:
            f.write(json.dumps(_slim(rec)) + "\n")

    # candidates_rejected.jsonl
    with open(OUT_DIR / "candidates_rejected.jsonl", "w") as f:
        for rec in rejected:
            f.write(json.dumps(_slim(rec)) + "\n")

    # candidates_survivors.jsonl
    with open(OUT_DIR / "candidates_survivors.jsonl", "w") as f:
        for rec in survivors:
            f.write(json.dumps(_slim(rec)) + "\n")

    # candidate_features.csv
    with open(OUT_DIR / "candidate_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in all_recs:
            row = {k: rec.get(k, "") for k in CSV_FIELDS}
            row["garbage_reasons"] = "|".join(rec.get("garbage_reasons", []))
            w.writerow(row)

    # summary_by_agent_dim.csv
    summary: dict[tuple, dict] = defaultdict(lambda: {"raw":0,"survived":0,"rejected":0})
    for rec in all_recs:
        key = (rec["agent"], rec["dim"])
        summary[key]["raw"] += 1
        if rec["garbage_filter_pass"]:
            summary[key]["survived"] += 1
        else:
            summary[key]["rejected"] += 1

    with open(OUT_DIR / "summary_by_agent_dim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Agent","Dim","Raw","Survived","Rejected","Survival%"])
        for (ag, dm), v in sorted(summary.items()):
            pct = 100.0 * v["survived"] / max(v["raw"], 1)
            w.writerow([ag, dm, v["raw"], v["survived"], v["rejected"], f"{pct:.1f}%"])

    # rejection_reasons.csv
    with open(OUT_DIR / "rejection_reasons.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reason","Count","Pct%"])
        tot = max(len(rejected), 1)
        for r, cnt in sorted(reason_tally.items(), key=lambda x: -x[1]):
            w.writerow([r, cnt, f"{100.0*cnt/tot:.1f}"])

    # cluster_summary.csv
    clust_groups: dict[int, list] = defaultdict(list)
    for rec in survivors:
        clust_groups[rec["cluster_id"]].append(rec)

    with open(OUT_DIR / "cluster_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ClusterID","Size","AgentMix","DimMix",
                    "MeanDelta","MeanBalance",
                    "BestBalance","BestProof","BestDiscovery"])
        for cid, mems in sorted(clust_groups.items()):
            amix = ",".join(sorted(set(m["agent"] for m in mems)))
            dmix = ",".join(str(d) for d in sorted(set(m["dim"] for m in mems)))
            mdelta = np.mean([m.get("best_delta", 0) for m in mems])
            mbal   = np.mean([m.get("balance_score", 0) for m in mems])
            bb  = max(mems, key=lambda m: m.get("balance_score",0))["candidate_id"]
            bp  = max(mems, key=lambda m: m.get("proof_score",0))["candidate_id"]
            bd  = max(mems, key=lambda m: m.get("discovery_score",0))["candidate_id"]
            w.writerow([cid, len(mems), amix, dmix,
                        f"{mdelta:.2f}", f"{mbal:.4f}", bb, bp, bd])

    # manual_review_queue.csv — top 3 per cluster by balance_score
    with open(OUT_DIR / "manual_review_queue.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rank","ClusterID","CandidateID","Agent","Dim",
                    "Delta","BalanceScore","ProofScore","DiscoveryScore",
                    "FamilyHash","Fingerprint"])
        rank = 1
        for cid, mems in sorted(clust_groups.items()):
            top = sorted(mems, key=lambda m: -m.get("balance_score", 0))[:3]
            for m in top:
                w.writerow([rank, cid, m["candidate_id"], m["agent"], m["dim"],
                            m.get("best_delta",""), m.get("balance_score",""),
                            m.get("proof_score",""), m.get("discovery_score",""),
                            m.get("family_hash",""), m.get("fingerprint","")])
                rank += 1

    # ── Print summary tables ─────────────────────────────────────────────────
    print("\n" + "="*66)
    print("  SURVIVAL TABLE")
    print("="*66)
    print(f"  {'Agent':>5}  {'Dim':>3}  {'Raw':>5}  {'Survived':>8}  {'Rejected':>8}  {'%':>6}")
    print("  " + "─"*52)
    tr = ts = 0
    for (ag, dm), v in sorted(summary.items()):
        pct = 100.0 * v["survived"] / max(v["raw"], 1)
        print(f"  {ag:>5}  {dm:>3}  {v['raw']:>5}  {v['survived']:>8}  "
              f"{v['rejected']:>8}  {pct:>5.1f}%")
        tr += v["raw"]; ts += v["survived"]
    print(f"  {'TOTAL':>5}  {'':>3}  {tr:>5}  {ts:>8}  {tr-ts:>8}  "
          f"{100.0*ts/max(tr,1):>5.1f}%")

    print("\n" + "="*66)
    print("  REJECTION REASONS")
    print("="*66)
    tot = max(len(rejected), 1)
    for r, cnt in sorted(reason_tally.items(), key=lambda x: -x[1]):
        print(f"  {r:<42} {cnt:>5}  ({100.0*cnt/tot:.1f}%)")

    if survivors:
        print("\n" + "="*66)
        print("  SURVIVOR PROFILE")
        print("="*66)
        def mn(k): return float(np.mean([c.get(k, 0) for c in survivors]))
        print(f"  mean best_delta:      {mn('best_delta'):>7.2f}")
        print(f"  mean ray_stability:   {mn('ray_stability'):>7.4f}")
        print(f"  mean proofability:    {mn('proofability'):>7.4f}")
        print(f"  mean simplicity:      {mn('simplicity'):>7.4f}")
        print(f"  mean balance_score:   {mn('balance_score'):>7.4f}")
        print(f"  clusters:             {n_clusters}")
        print(f"  mean cluster size:    {len(survivors)/n_clusters:.1f}")

    total_t = time.time() - t0
    print(f"\nPipeline complete in {total_t:.0f}s.")
    print(f"Outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    run_pipeline()
