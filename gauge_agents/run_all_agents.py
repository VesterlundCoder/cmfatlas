#!/usr/bin/env python3
"""
run_all_agents.py — Unified CMF Agent Runner (A through J)
===========================================================
Runs all 10 gauge-bootstrap agents.  Each agent searches until it has
found TARGET=100 NEW converging CMFs.  Every PUSH_EVERY=20 newly found
CMFs (across all agents) the results are ingested into atlas_2d.db and
pushed live to Railway.

NEW: Every candidate passes the 3-step asymptotic filter BEFORE the
     expensive T2/T3 walks are attempted.

Usage:
    python3 run_all_agents.py                        # all agents A-J, parallel
    python3 run_all_agents.py --agents A B E         # specific agents
    python3 run_all_agents.py --agents A --target 50 # custom target
    python3 run_all_agents.py --sequential            # run one at a time

Agents:
    A  3-5×3-5,  3-var,  classic LDU  (extended off-diag values)
    B  3-5×3-5,  3-var,  sparse LDU   (small fractions)
    C  6-8×6-8,  3-var,  large-dim climbing
    D  4-6×4-6,  4-var,  4-variable holonomic
    E  3-5×3-5,  3-var,  extended rationals (thirds / halves)
    F  3-5×3-5,  3-var,  symmetric G  (L_off = U_off^T)
    G  3-6×3-6,  3-var,  minimal structure (1 off-diagonal)
    H  4-6×4-6,  3-var,  dense off-diagonals
    I  3-4×3-4,  3-var,  integer-only parameters, wide range
    J  3-6×3-6,  5-var,  5-variable holonomic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import mpmath as mp
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
ROOT     = HERE.parent
DB_PATH  = ROOT / "data" / "atlas_2d.db"
INGEST   = ROOT / "ingest_gauge_agents.py"
PYTHON   = sys.executable

# ── Run-time config ────────────────────────────────────────────────────────────
TARGET     = 100      # new CMFs per agent
PUSH_EVERY = 20       # push DB + Railway after this many new CMFs (global)
SENTINEL   = HERE / "STOP_AGENTS"

# ── Filter thresholds ──────────────────────────────────────────────────────────
DELTA_FAST_MIN = 1.0
DELTA_FULL_MIN = 2.5
DPS_FULL       = 50

# ── Agent configuration table ─────────────────────────────────────────────────
#   dims      : matrix sizes to try  (chosen randomly each trial)
#   n_vars    : number of independent lattice variables
#   off_vals  : pool for L/U off-diagonal scalars
#   slope_vals: pool for D_params slope a_k
#   shift_vals: pool for D_params shift b_k
#   n_off_max : max off-diagonal entries per L or U
#   symmetric : if True, force U_off[(i,j)] = L_off[(j,i)]
#   sparse    : if True, exactly 1 off-diagonal per L and U
#   dense     : if True, n_off = dim*(dim-1)//2  (fully off-diagonal)

AGENT_CONFIGS: dict[str, dict] = {
    "A": dict(dims=[3,4,5], n_vars=3,
              off_vals=[-2,-1,-0.5,0.5,1,2],
              slope_vals=[-3,-2,-1,1,2,3],
              shift_vals=list(range(-3, 4))),
    "B": dict(dims=[3,4,5], n_vars=3,
              off_vals=[-0.5,-0.25,0.25,0.5],
              slope_vals=[-2,-1,1,2],
              shift_vals=list(range(-2, 3))),
    "C": dict(dims=[6,7,8], n_vars=3,
              off_vals=[-0.5,-0.25,0.25,0.5],
              slope_vals=[-2,-1,1,2],
              shift_vals=list(range(-2, 3))),
    "D": dict(dims=[4,5,6], n_vars=4,
              off_vals=[-2,-1,-0.5,-1/3,1/3,0.5,1,2],
              slope_vals=[-2,-1,1,2],
              shift_vals=list(range(-2, 3))),
    "E": dict(dims=[3,4,5], n_vars=3,
              off_vals=[-2/3,-1/3,1/3,2/3,-3/2,-1/2,1/2,3/2,-1,-2,1,2],
              slope_vals=[-3,-2,-1,1,2,3],
              shift_vals=list(range(-4, 5))),
    "F": dict(dims=[3,4,5], n_vars=3,
              off_vals=[-1,-0.5,0.5,1],
              slope_vals=[-1,1],
              shift_vals=[-1,0,1],
              symmetric=True),
    "G": dict(dims=[3,4,5,6], n_vars=3,
              off_vals=[-2,-1,1,2],
              slope_vals=[-2,-1,1,2],
              shift_vals=list(range(-2, 3)),
              sparse=True),
    "H": dict(dims=[4,5,6], n_vars=3,
              off_vals=[-1,-0.5,0.5,1],
              slope_vals=[-2,-1,1,2],
              shift_vals=list(range(-2, 3)),
              dense=True),
    "I": dict(dims=[3,4], n_vars=3,
              off_vals=[-3,-2,-1,1,2,3],
              slope_vals=[-3,-2,-1,1,2,3],
              shift_vals=list(range(-5, 6))),
    "J": dict(dims=[3,4,5,6], n_vars=5,
              off_vals=[-1,-0.5,0.5,1],
              slope_vals=[-2,-1,1,2],
              shift_vals=list(range(-2, 3))),
    # ── Irrational-targeted agents (K/L/M) ────────────────────────────────────
    # All-positive slopes   → no sign-oscillation in D_params
    # Half-integer shifts   → Γ(n+½), Γ(n+3/2) … → √π factors in products
    # n_vars == dim         → each dimension fully independent (max mixing)
    "K": dict(dims=[3], n_vars=3,
              off_vals=[-1, -0.5, -1/3, 1/3, 0.5, 1],
              slope_vals=[0.5, 1.0, 1.5, 2.0],
              shift_vals=[-0.5, 0.5, 1.5, 2.5, 3.5]),
    "L": dict(dims=[4], n_vars=4,
              off_vals=[-1, -0.5, -1/3, 1/3, 0.5, 1],
              slope_vals=[0.5, 1.0, 1.5, 2.0],
              shift_vals=[-0.5, 0.5, 1.5, 2.5, 3.5]),
    "M": dict(dims=[5], n_vars=5,
              off_vals=[-1, -0.5, -1/3, 1/3, 0.5, 1],
              slope_vals=[0.5, 1.0, 1.5, 2.0],
              shift_vals=[-0.5, 0.5, 1.5, 2.5, 3.5]),
}


# ══════════════════════════════════════════════════════════════════════════════
# LDU Gauge Bootstrap Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sample_params(dim: int, cfg: dict, rng: np.random.Generator) -> dict:
    """Sample LDU parameters according to agent config."""
    off_vals   = cfg["off_vals"]
    slope_vals = cfg["slope_vals"]
    shift_vals = cfg["shift_vals"]
    n_vars     = cfg["n_vars"]
    symmetric  = cfg.get("symmetric", False)
    sparse     = cfg.get("sparse", False)
    dense      = cfg.get("dense", False)

    # D_params: dim entries (slope, shift) — each entry uses coords[k % n_vars]
    D_params = [
        (float(rng.choice(slope_vals)), float(rng.choice(shift_vals)))
        for _ in range(dim)
    ]

    # Determine number of off-diagonal entries
    max_off = dim * (dim - 1) // 2
    if sparse:
        n_off = 1
    elif dense:
        n_off = max_off
    else:
        n_off = int(rng.integers(1, max(2, min(dim - 1, 4)) + 1))
    n_off = min(n_off, max_off)

    # L_off (lower triangular)
    lower_pairs = [(i, j) for i in range(dim) for j in range(i)]
    idx = rng.permutation(len(lower_pairs))[:n_off].tolist()
    L_off = {}
    for k in idx:
        i, j = lower_pairs[k]
        L_off[(i, j)] = float(rng.choice(off_vals))

    # U_off (upper triangular)
    upper_pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    if symmetric:
        # Mirror L_off into U: U[i,j] = L[j,i] if (j,i) in L_off
        U_off = {}
        for (i, j), v in L_off.items():
            U_off[(j, i)] = v          # L[i,j] ↔ U[j,i]
        # Fill remaining upper slots if n_off > len(L_off)
        remaining = [p for p in upper_pairs if p not in U_off]
        extra = n_off - len(L_off)
        if extra > 0 and remaining:
            rng.shuffle(remaining)
            for p in remaining[:extra]:
                U_off[p] = float(rng.choice(off_vals))
    else:
        idx2 = rng.permutation(len(upper_pairs))[:n_off].tolist()
        U_off = {}
        for k in idx2:
            i, j = upper_pairs[k]
            U_off[(i, j)] = float(rng.choice(off_vals))

    return {
        "dim":      dim,
        "n_vars":   n_vars,
        "D_params": D_params,
        "L_off":    L_off,
        "U_off":    U_off,
    }


def build_eval_fns(params: dict) -> list:
    """
    Build n_vars evaluation functions  fn_i(*coords) -> dim×dim ndarray.
    X_i(*coords) = G(coords + e_i) · D_i(coords[i]) · G(coords)^{-1}
    G(*coords)   = L · diag(a_k · (coords[k%n_vars] + b_k)) · U
    D_i(*coords) = diag(coords[i] + 0, coords[i] + 1, …, coords[i] + dim-1)
    """
    dim    = params["dim"]
    n_vars = params["n_vars"]
    D_p    = params["D_params"]
    L_off  = params["L_off"]
    U_off  = params["U_off"]

    def G(coords: list) -> np.ndarray:
        L = np.eye(dim)
        for (i, j), v in L_off.items():
            L[i, j] = v
        diag_v = np.array([
            D_p[k][0] * (coords[k % n_vars] + D_p[k][1])
            for k in range(dim)
        ])
        U = np.eye(dim)
        for (i, j), v in U_off.items():
            U[i, j] = v
        return L @ np.diag(diag_v) @ U

    fns = []
    for axis in range(n_vars):
        def make(ax=axis):
            def fn(*coords):
                c   = list(coords)
                csh = list(coords)
                csh[ax] += 1
                Gn  = G(c)
                det = np.linalg.det(Gn)
                if abs(det) < 1e-10:
                    raise ValueError("singular G")
                Di = np.diag(np.array([c[ax] + k for k in range(dim)], dtype=float))
                return G(csh) @ Di @ np.linalg.inv(Gn)
            return fn
        fns.append(make())
    return fns


# ══════════════════════════════════════════════════════════════════════════════
# Quality Checks  (T1 / T2 / T3 / T4)
# ══════════════════════════════════════════════════════════════════════════════

def t1_pole_check(fns: list, dim: int, n_vars: int,
                  rng: np.random.Generator) -> bool:
    """T1: 30 random points — all finite and invertible."""
    box = max(4, 15 - dim)
    for _ in range(30):
        coords = rng.integers(1, box + 1, size=n_vars).tolist()
        for fn in fns:
            try:
                M = np.asarray(fn(*coords), dtype=float)
                if not np.all(np.isfinite(M)):
                    return False
                if abs(np.linalg.det(M)) < 1e-14:
                    return False
            except Exception:
                return False
    return True


def _walk_np(fn, dim: int, n_vars: int, depth: int) -> Optional[float]:
    """Numpy walk along axis 0. Returns v[0]/v[-1] or None."""
    pos = [2] * n_vars
    v   = np.zeros(dim); v[0] = 1.0
    for _ in range(depth):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
            v = M @ v
        except Exception:
            return None
        n = np.max(np.abs(v))
        if n > 1e25:
            v /= n
        elif n < 1e-25:
            return None
    if abs(v[-1]) < 1e-18:
        return None
    return float(v[0] / v[-1])


def t2_fast(fns: list, dim: int, n_vars: int) -> tuple[bool, float]:
    """T2: fast numpy convergence. Returns (pass, best_delta)."""
    best = 0.0
    for fn in fns[:min(3, len(fns))]:
        r1 = _walk_np(fn, dim, n_vars, 150)
        r2 = _walk_np(fn, dim, n_vars, 600)
        if r1 is None or r2 is None:
            continue
        diff = abs(r2 - r1)
        d    = 50.0 if diff < 1e-50 else min(50.0, -math.log10(diff + 1e-55))
        best = max(best, d)
    return best >= DELTA_FAST_MIN, best


def _walk_mp(fn, dim: int, n_vars: int, depth: int, dps: int) -> Optional[mp.mpf]:
    """High-precision walk along axis 0."""
    mp.mp.dps = dps + 10
    pos = [2] * n_vars
    v   = mp.zeros(dim, 1); v[0] = mp.mpf(1)
    for _ in range(depth):
        pos[0] += 1
        try:
            raw = np.asarray(fn(*pos), dtype=float)
            M   = mp.matrix([[mp.mpf(str(raw[r][c]))
                              for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            return None
        scale = max(abs(v[i]) for i in range(dim))
        if scale > mp.power(10, 25):
            v /= scale
        elif scale < mp.power(10, -25):
            return None
    if abs(v[dim - 1]) < mp.power(10, -(dps - 5)):
        return None
    return v[0] / v[dim - 1]


def t3_thorough(fns: list, dim: int, n_vars: int) -> tuple[bool, list]:
    """T3: mpmath check on all n_vars axes."""
    depth1 = max(40, 300 // max(1, dim // 2))
    depth2 = max(150, 1200 // max(1, dim // 2))
    deltas = []
    for fn in fns:
        r1 = _walk_mp(fn, dim, n_vars, depth1, DPS_FULL)
        r2 = _walk_mp(fn, dim, n_vars, depth2, DPS_FULL)
        if r1 is None or r2 is None:
            deltas.append(0.0)
            continue
        diff = abs(r2 - r1)
        d    = float(DPS_FULL) if diff < mp.power(10, -DPS_FULL) \
               else float(-mp.log10(diff + mp.power(10, -(DPS_FULL + 5))))
        deltas.append(min(float(DPS_FULL), max(0.0, d)))

    best   = max(deltas) if deltas else 0.0
    n_good = sum(1 for d in deltas if d > 1.0)
    min_ok = max(1, len(fns) // 3)
    return best >= DELTA_FULL_MIN and n_good >= min_ok, deltas


def t4_flatness(fns: list, dim: int, n_vars: int,
                n_pts: int = 12) -> tuple[bool, float]:
    """T4: path-independence check on all C(n_vars,2) pairs."""
    rng   = np.random.default_rng(42)
    max_e = 0.0
    for _ in range(n_pts):
        pos = rng.integers(2, 8, size=n_vars).tolist()
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                pei, pej = list(pos), list(pos)
                pei[i] += 1; pej[j] += 1
                try:
                    Xi_n    = np.asarray(fns[i](*pos), dtype=float)
                    Xj_n    = np.asarray(fns[j](*pos), dtype=float)
                    Xj_pei  = np.asarray(fns[j](*pei), dtype=float)
                    Xi_pej  = np.asarray(fns[i](*pej), dtype=float)
                    err = np.max(np.abs(Xi_pej @ Xj_n - Xj_pei @ Xi_n))
                    if not np.isfinite(err):
                        return False, 1e30
                    max_e = max(max_e, err)
                    if err > 1e-5:
                        return False, max_e
                except Exception:
                    return False, 1e30
    return True, max_e


def bidir_ratio(fns: list, dim: int, n_vars: int,
                n_pts: int = 4) -> float:
    """Coupling measure: fraction of matrix pairs with bidirectional off-diag."""
    rng = np.random.default_rng(77)
    ij_pos: dict = {}; ij_neg: dict = {}
    for _ in range(n_pts):
        pos = rng.integers(3, 10, size=n_vars).tolist()
        for fn in fns:
            try:
                M = np.asarray(fn(*pos), dtype=float)
            except Exception:
                continue
            for i in range(dim):
                for j in range(i + 1, dim):
                    k = (i, j)
                    if abs(M[i, j]) > 1e-7: ij_pos[k] = True
                    if abs(M[j, i]) > 1e-7: ij_neg[k] = True
    all_k = set(ij_pos) | set(ij_neg)
    if not all_k:
        return 0.0
    bd = sum(1 for k in all_k if ij_pos.get(k) and ij_neg.get(k))
    return bd / len(all_k)


# ══════════════════════════════════════════════════════════════════════════════
# Fingerprint and Store
# ══════════════════════════════════════════════════════════════════════════════

def fingerprint(fns: list, dim: int, n_vars: int) -> str:
    rng = np.random.default_rng(999)
    vals = []
    for _ in range(8):
        coords = rng.integers(2, 8, size=n_vars).tolist()
        fn     = fns[int(rng.integers(0, len(fns)))]
        try:
            M = np.asarray(fn(*coords), dtype=float)
            vals.extend([round(float(v), 5) for v in M.ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:16]


def store_path(agent: str, dim: int) -> Path:
    return HERE / f"store_{agent}_{dim}x{dim}.jsonl"


def load_seen(agent: str) -> set:
    """Load all stored fingerprints for an agent (all dims)."""
    seen: set = set()
    for p in HERE.glob(f"store_{agent}_*.jsonl"):
        for line in p.read_text(errors="replace").splitlines():
            try:
                seen.add(json.loads(line)["fingerprint"])
            except Exception:
                pass
    return seen


def count_lines(agent: str, dim: int) -> int:
    p = store_path(agent, dim)
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text(errors="replace").splitlines() if ln.strip())


def serialise_params(p: dict) -> dict:
    return {
        "dim":      p["dim"],
        "n_vars":   p["n_vars"],
        "D_params": p["D_params"],
        "L_off":    {str(k): v for k, v in p["L_off"].items()},
        "U_off":    {str(k): v for k, v in p["U_off"].items()},
    }


def write_record(agent: str, dim: int, params: dict,
                 deltas: list, fp: str, pi_err: float,
                 br: float) -> None:
    rec = {
        "dim":                        dim,
        "n_matrices":                 len(deltas),
        "matrix_size":                dim,
        "agent":                      agent,
        "fingerprint":                fp,
        "best_delta":                 round(max(deltas) if deltas else 0.0, 3),
        "deltas":                     [round(d, 3) for d in deltas],
        "n_converging_axes":          sum(1 for d in deltas if d > 1.0),
        "path_independence_verified": True,
        "flatness_verified":          True,
        "max_flatness_error":         round(pi_err, 12),
        "n_pairs_checked":            params["n_vars"] * (params["n_vars"] - 1) // 2,
        "coupling_bucket":            4 if br >= 0.50 else 3 if br >= 0.10 else 2,
        "bidir_ratio":                round(br, 4),
        "params":                     serialise_params(params),
        "timestamp":                  time.time(),
    }
    with open(store_path(agent, dim), "a") as f:
        f.write(json.dumps(rec) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# DB Ingest + Railway Push
# ══════════════════════════════════════════════════════════════════════════════

def ingest_and_push() -> None:
    """Run ingest_gauge_agents.py then railway up --detach."""
    print("\n  ⟳  Ingesting to atlas_2d.db …", flush=True)
    try:
        subprocess.run(
            [PYTHON, str(INGEST), "--db", str(DB_PATH)],
            cwd=str(ROOT), timeout=120, check=True,
        )
        print("  ✓  Ingest done.", flush=True)
    except Exception as exc:
        print(f"  ✗  Ingest error: {exc}", flush=True)
        return

    print("  ⟳  Railway up …", flush=True)
    try:
        subprocess.run(
            ["railway", "up", "--detach"],
            cwd=str(ROOT), timeout=60, check=False,
        )
        print("  ✓  Railway pushed.", flush=True)
    except Exception as exc:
        print(f"  ✗  Railway push error: {exc}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Per-Agent Search Loop
# ══════════════════════════════════════════════════════════════════════════════

def _import_filter():
    sys.path.insert(0, str(HERE))
    from asymptotic_filter import asymptotic_filter
    return asymptotic_filter


def run_agent(agent: str, target: int, push_counter: "multiprocessing.Value") -> None:
    """
    Main search loop for one agent.  Runs until `target` new CMFs are found.
    `push_counter` is a shared multiprocessing.Value — when it crosses a
    PUSH_EVERY multiple the parent process handles the push.
    """
    asymptotic_filter = _import_filter()
    cfg    = AGENT_CONFIGS[agent]
    dims   = cfg["dims"]
    n_vars = cfg["n_vars"]
    rng    = np.random.default_rng(int(time.time() * 1000) % (2**31))

    seen   = load_seen(agent)
    found  = 0

    print(f"  [{agent}] starting | dims={dims} n_vars={n_vars} "
          f"already_seen={len(seen)}", flush=True)
    t0     = time.time()
    trials = 0
    af_rejected = 0
    t1_rejected = 0
    t2_rejected = 0
    t3_rejected = 0
    t4_rejected = 0

    while found < target:
        if SENTINEL.exists():
            print(f"  [{agent}] STOP_AGENTS sentinel detected — exiting.", flush=True)
            break

        trials += 1
        dim    = int(rng.choice(dims))

        # ── Sample params ──────────────────────────────────────────────────
        try:
            params = sample_params(dim, cfg, rng)
            fns    = build_eval_fns(params)
        except Exception:
            continue

        # ── T0: Asymptotic fast-fail filter (NEW) ─────────────────────────
        ok, reason = asymptotic_filter(fns, dim, n_vars)
        if not ok:
            af_rejected += 1
            continue

        # ── T1: Pole check ────────────────────────────────────────────────
        if not t1_pole_check(fns, dim, n_vars, rng):
            t1_rejected += 1
            continue

        # ── T2: Fast numpy convergence ────────────────────────────────────
        t2_ok, delta_fast = t2_fast(fns, dim, n_vars)
        if not t2_ok:
            t2_rejected += 1
            continue

        # ── T3: High-precision mpmath convergence ─────────────────────────
        t3_ok, deltas = t3_thorough(fns, dim, n_vars)
        if not t3_ok:
            t3_rejected += 1
            continue

        # ── T4: Path independence (flatness) ──────────────────────────────
        pi_ok, pi_err = t4_flatness(fns, dim, n_vars)
        if not pi_ok:
            t4_rejected += 1
            continue

        # ── Fingerprint + deduplication ───────────────────────────────────
        fp = fingerprint(fns, dim, n_vars)
        if fp in seen:
            continue
        seen.add(fp)

        # ── Measure coupling ──────────────────────────────────────────────
        br = bidir_ratio(fns, dim, n_vars)

        # ── Accept ────────────────────────────────────────────────────────
        write_record(agent, dim, params, deltas, fp, pi_err, br)
        found  += 1
        best_d  = max(deltas)
        elapsed = time.time() - t0

        print(
            f"  [{agent}] ✓ #{found:>3}/{target}  {dim}×{dim}  "
            f"Δ={best_d:.1f}  axes={sum(1 for d in deltas if d > 1.0)}/{n_vars}  "
            f"bidir={br:.2f}  fp={fp}  "
            f"trials={trials:,}  t={elapsed:.0f}s  "
            f"(AF-rej={af_rejected:,} T1={t1_rejected:,} "
            f"T2={t2_rejected:,} T3={t3_rejected:,})",
            flush=True,
        )

        # Signal push_counter (parent monitors this)
        with push_counter.get_lock():
            push_counter.value += 1

    total_t = time.time() - t0
    print(
        f"\n  [{agent}] Done.  found={found}/{target}  trials={trials:,}  "
        f"AF-rej={af_rejected:,}  time={total_t:.0f}s",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Monitor + Parallel Launcher
# ══════════════════════════════════════════════════════════════════════════════

def _agent_worker(agent: str, target: int,
                  push_counter: "multiprocessing.Value") -> None:
    """Wrapper for multiprocessing.Process."""
    try:
        run_agent(agent, target, push_counter)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"  [{agent}] CRASH: {exc}", flush=True)
        import traceback
        traceback.print_exc()


def run_parallel(agents: list[str], target: int) -> None:
    push_counter = multiprocessing.Value("i", 0)
    last_pushed  = 0

    processes: list[multiprocessing.Process] = []
    for agent in agents:
        p = multiprocessing.Process(
            target=_agent_worker,
            args=(agent, target, push_counter),
            name=f"Agent-{agent}",
            daemon=False,
        )
        p.start()
        processes.append(p)
        print(f"  Launched Agent {agent}  PID={p.pid}", flush=True)

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(10)
            with push_counter.get_lock():
                current = push_counter.value

            if current - last_pushed >= PUSH_EVERY:
                last_pushed = current
                ingest_and_push()

        # Final push after all agents finish
        with push_counter.get_lock():
            final = push_counter.value
        if final > last_pushed:
            ingest_and_push()

    except KeyboardInterrupt:
        print("\n  Interrupted — stopping agents …", flush=True)
        SENTINEL.touch()
        for p in processes:
            p.join(timeout=30)
        if SENTINEL.exists():
            SENTINEL.unlink()

    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join()

    print("\n  All agents finished.", flush=True)


def run_sequential(agents: list[str], target: int) -> None:
    push_counter = multiprocessing.Value("i", 0)
    last_pushed  = 0

    for agent in agents:
        run_agent(agent, target, push_counter)
        with push_counter.get_lock():
            current = push_counter.value
        if current - last_pushed >= PUSH_EVERY:
            last_pushed = current
            ingest_and_push()

    # Final push
    with push_counter.get_lock():
        if push_counter.value > last_pushed:
            ingest_and_push()


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Run CMF gauge agents A-J")
    ap.add_argument(
        "--agents", nargs="+",
        default=list(AGENT_CONFIGS.keys()),
        help="Agents to run (default: all A-J). Example: --agents A B E",
    )
    ap.add_argument(
        "--target", type=int, default=TARGET,
        help=f"New CMFs to find per agent (default: {TARGET})",
    )
    ap.add_argument(
        "--sequential", action="store_true",
        help="Run agents one at a time instead of in parallel",
    )
    args = ap.parse_args()

    agents = [a.upper() for a in args.agents]
    for a in agents:
        if a not in AGENT_CONFIGS:
            ap.error(f"Unknown agent '{a}'. Valid: {list(AGENT_CONFIGS.keys())}")

    print("=" * 68, flush=True)
    print(f"  CMF Agent Runner  |  agents={agents}  target={args.target}/each  "
          f"push_every={PUSH_EVERY}", flush=True)
    print("=" * 68, flush=True)

    if args.sequential:
        run_sequential(agents, args.target)
    else:
        run_parallel(agents, args.target)


if __name__ == "__main__":
    main()
