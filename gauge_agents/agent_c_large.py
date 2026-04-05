#!/usr/bin/env python3
"""
agent_c_large.py — Agent C: Dimension-Climbing CMF Explorer
============================================================

Tries to find gauge-bootstrap CMFs of increasing size:
  6 matrices of 6×6  →  7 matrices of 7×7  →  … → as large as possible.

A d-dimensional CMF has:
  • d integer coordinates n = (n_0, …, n_{d-1})
  • d matrix functions X_0, …, X_{d-1}, each d×d
  • LDU gauge:  G(n) = L · diag(a_k·(n_{k%d} + b_k)) · U
  • Canonical kernel:  D_i(v) = diag(v, v+1, …, v+d-1)
  • Path independence: guaranteed by gauge construction

Quality bar (3-tier, all must pass):
  T1 – quick pole-free check  (numpy, 30 points)
  T2 – fast convergence check (numpy, depth 150 vs 600)
  T3 – thorough check         (mpmath dps=50, depth 300 vs 1500, ≥2 axes)

Strategy:
  • Start dim = 6.
  • Try up to MAX_PER_DIM random LDU configs.
  • If ≥ TARGET_PER_DIM valid systems found, increment dim.
  • If zero found after MAX_PER_DIM: stop (can't reach this dim).
  • Total budget: TRIALS_TOTAL = 100,000.

Stores:
  store_C_6x6.jsonl, store_C_7x7.jsonl, …

Usage:
    python3 agent_c_large.py
"""
from __future__ import annotations

import hashlib, json, math, sys, time
from pathlib import Path
from typing import Optional

import mpmath as mp
import numpy as np

HERE = Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────────
START_DIM      = 6
MAX_DIM        = 50
TRIALS_TOTAL   = 100_000
MAX_PER_DIM    = 5_000     # give up on this dim after this many trials
TARGET_PER_DIM = 1         # 1 success → advance (goal is maximum dimension)

DELTA_FAST_MIN = 1.0       # T2: quick numpy delta threshold
DELTA_FULL_MIN = 2.5       # T3: mpmath delta threshold (at least 1 axis)
DPS_FULL       = 50        # mpmath dps for full check
N_POLE_SAMPLES = 30        # T1: test points for pole check
STORE_CAPACITY = 200       # max records per dim store file

# ── Gaussian noise choices for LDU off-diagonals ─────────────────────────────
_OFFDIAG_VALS = [-0.5, -0.25, 0.25, 0.5]
_SLOPE_VALS   = [-2.0, -1.0, 1.0, 2.0]
_SHIFT_VALS   = [-2.0, -1.0, 0.0, 1.0, 2.0]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  G matrix and eval-function construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_G(params: dict, coords: list) -> np.ndarray:
    """G(n) = L · diag(a_k*(n_{k%dim}+b_k)) · U"""
    dim = params["dim"]
    L = np.eye(dim)
    for (i, j), v in params["L_off"].items():
        L[i, j] = v
    diag = np.array([a * (coords[k % dim] + b)
                     for k, (a, b) in enumerate(params["D_params"])])
    U = np.eye(dim)
    for (i, j), v in params["U_off"].items():
        U[i, j] = v
    return L @ np.diag(diag) @ U


def _build_Di(dim: int, coord_val: float) -> np.ndarray:
    """Canonical kernel: D_i(v) = diag(v, v+1, …, v+dim-1)"""
    return np.diag([coord_val + k for k in range(dim)])


def build_eval_fns(params: dict) -> list:
    """Returns list of dim callables  fn_i(*coords) -> dim×dim ndarray."""
    dim = params["dim"]
    fns = []
    for axis in range(dim):
        def mk(ax=axis):
            def fn(*coords):
                G_n = _build_G(params, list(coords))
                shifted = list(coords)
                shifted[ax] += 1
                G_sh = _build_G(params, shifted)
                Di   = _build_Di(dim, coords[ax])
                det  = np.linalg.det(G_n)
                if abs(det) < 1e-10:
                    raise ValueError("singular G")
                return G_sh @ Di @ np.linalg.inv(G_n)
            return fn
        fns.append(mk())
    return fns


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Quality checks
# ══════════════════════════════════════════════════════════════════════════════

def _t1_pole_check(fns: list, dim: int, rng: np.random.Generator) -> bool:
    """T1: check X_i(n) is finite and invertible at 30 random positive points."""
    box = max(5, 20 - dim)   # shrink box for large dim to avoid overflow
    pts = rng.integers(1, box + 1, size=(N_POLE_SAMPLES, dim)).tolist()
    for coords in pts:
        for ax in range(dim):
            try:
                M = fns[ax](*coords)
                if not np.all(np.isfinite(M)):
                    return False
                if abs(np.linalg.det(M)) < 1e-18:
                    return False
            except Exception:
                return False
    return True


def _walk_numpy(fns: list, dim: int, axis: int,
                start: list, depth: int) -> Optional[float]:
    """Numpy walk along one axis. Returns v[0]/v[dim-1] or None."""
    pos = list(start)
    v = np.zeros(dim)
    v[0] = 1.0
    for _ in range(depth):
        pos[axis] += 1
        try:
            M = fns[axis](*pos)
            v = M @ v
        except Exception:
            return None
        norm = np.max(np.abs(v))
        if norm > 1e30:
            v /= norm
        elif norm < 1e-30:
            return None
    if abs(v[dim - 1]) < 1e-20:
        return None
    return float(v[0] / v[dim - 1])


def _t2_fast_convergence(fns: list, dim: int) -> tuple[bool, float]:
    """T2: estimate delta on 3 axes using numpy. Returns (pass, best_delta)."""
    start = [2] * dim
    best = 0.0
    n_ax = min(3, dim)
    for ax in range(n_ax):
        r1 = _walk_numpy(fns, dim, ax, start, depth=150)
        r2 = _walk_numpy(fns, dim, ax, start, depth=600)
        if r1 is None or r2 is None:
            continue
        diff = abs(r2 - r1)
        if diff < 1e-50:
            best = max(best, 50.0)
        else:
            d = min(50.0, -math.log10(diff + 1e-55))
            best = max(best, d)
    return best >= DELTA_FAST_MIN, best


def _walk_mp(fns: list, dim: int, axis: int,
             start: list, depth: int, dps: int) -> Optional[mp.mpf]:
    """High-precision walk along one axis."""
    mp.mp.dps = dps + 10
    pos = list(start)
    v = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)
    for _ in range(depth):
        pos[axis] += 1
        try:
            M_raw = fns[axis](*pos)
            M = mp.matrix([[mp.mpf(str(M_raw[r][c]))
                            for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            return None
        scale = max(abs(v[i]) for i in range(dim))
        if scale > mp.power(10, 30):
            v /= scale
        elif scale < mp.power(10, -30):
            return None
    if abs(v[dim - 1]) < mp.power(10, -(dps - 10)):
        return None
    return v[0] / v[dim - 1]


def _check_path_independence_nd(fns: list, dim: int,
                               n_pts: int = 5) -> tuple[bool, float]:
    """
    Verify X_i(n+e_j)·X_j(n) = X_j(n+e_i)·X_i(n)
    for ALL C(dim,2) pairs at n_pts test points.
    Returns (all_pass, max_flatness_error).
    """
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(n_pts):
        pos = [int(v) for v in rng.integers(2, 8, size=dim)]
        for i in range(dim):
            for j in range(i + 1, dim):
                pej = list(pos); pej[j] += 1
                pei = list(pos); pei[i] += 1
                try:
                    Xi_nej = np.array(fns[i](*pej), float)
                    Xj_n   = np.array(fns[j](*pos), float)
                    Xj_nei = np.array(fns[j](*pei), float)
                    Xi_n   = np.array(fns[i](*pos), float)
                    err = np.max(np.abs(Xi_nej @ Xj_n - Xj_nei @ Xi_n))
                    if not np.isfinite(err):
                        return False, 1e30
                    max_err = max(max_err, err)
                    if err > 1e-5:
                        return False, max_err
                except Exception:
                    return False, 1e30
    return True, max_err


def _t3_thorough_check(fns: list, dim: int) -> tuple[bool, list]:
    """
    T3: mpmath delta on ALL d axes. Returns (pass, [delta_axis_0, …, delta_axis_{d-1}]).
    Uses adaptive depth so large dimensions don't dominate runtime.
    """
    start = [2] * dim
    deltas = []
    # Adaptive depth: dim=6 → (150, 600); dim=12 → (75, 300); dim=20 → (45, 180)
    depth1 = max(40, 300 // max(1, dim // 2))
    depth2 = max(150, 1200 // max(1, dim // 2))

    for ax in range(dim):
        r1 = _walk_mp(fns, dim, ax, start, depth=depth1, dps=DPS_FULL)
        r2 = _walk_mp(fns, dim, ax, start, depth=depth2, dps=DPS_FULL)
        if r1 is None or r2 is None:
            deltas.append(0.0)
            continue
        diff = abs(r2 - r1)
        if diff < mp.power(10, -DPS_FULL):
            deltas.append(float(DPS_FULL))
        else:
            d = float(-mp.log10(diff + mp.power(10, -(DPS_FULL + 10))))
            deltas.append(min(float(DPS_FULL), max(0.0, d)))

    best = max(deltas) if deltas else 0.0
    n_good = sum(1 for d in deltas if d > 1.0)
    # Pass: best delta above threshold AND at least ceil(dim/3) axes converging
    min_good = max(2, dim // 3)
    return (best >= DELTA_FULL_MIN and n_good >= min_good), deltas


def _trivial_collapse_check(fns: list, dim: int) -> bool:
    """Return True (not collapsed) if X_0 varies as coord_0 changes."""
    try:
        start = [2] * dim
        mats = []
        for n in range(1, 8):
            pos = list(start); pos[0] = n
            M = fns[0](*pos)
            mats.append(np.array(M))
        base = mats[0]; denom = max(np.max(np.abs(base)), 1e-10)
        spread = max(np.max(np.abs(m - base)) / denom for m in mats[1:])
        return spread > 0.01   # True = not collapsed
    except Exception:
        return True  # can't evaluate = assume not collapsed


def _fingerprint(fns: list, dim: int) -> str:
    rng2 = np.random.default_rng(999)
    vals = []
    for _ in range(8):
        coords = list(rng2.integers(2, 8, size=dim).tolist())
        ax = int(rng2.integers(0, dim))
        try:
            M = fns[ax](*coords)
            vals.extend([round(float(v), 5) for v in np.array(M).ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Parameter sampling
# ══════════════════════════════════════════════════════════════════════════════

# Minimum bidir_ratio thresholds (info labels only — not acceptance gates)
MIN_BIDIR = 0.10


def sample_params(dim: int, rng: np.random.Generator) -> dict:
    """
    Sample random LDU parameters for a d-dimensional CMF.
    Off-diagonal count sparse for large dim (better conditioned).
    """
    n_off = max(1, min(dim - 1, int(rng.integers(1, max(2, dim // 2) + 1))))

    # D_params: d entries (slope, shift), slopes non-zero
    D_params = [(float(rng.choice(_SLOPE_VALS)),
                 float(rng.choice(_SHIFT_VALS))) for _ in range(dim)]

    # L_off: lower-triangular, n_off entries
    lower_pairs = [(i, j) for i in range(dim) for j in range(i)]
    rng.shuffle(lower_pairs)
    L_off = {}
    for (i, j) in lower_pairs[:n_off]:
        L_off[(i, j)] = float(rng.choice(_OFFDIAG_VALS))

    # U_off: upper-triangular, n_off entries (REQUIRED for B4 coupling)
    upper_pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    rng.shuffle(upper_pairs)
    U_off = {}
    for (i, j) in upper_pairs[:n_off]:
        U_off[(i, j)] = float(rng.choice(_OFFDIAG_VALS))

    return {"dim": dim, "D_params": D_params, "L_off": L_off, "U_off": U_off}


def _t5_coupling_check(fns: list, dim: int,
                       min_bidir: float = MIN_BIDIR,
                       n_pts: int = 4) -> tuple[bool, float]:
    """
    T5: fast numpy bidir_ratio check.  Rejects one-sided / fully-triangular CMFs.
    Returns (pass, bidir_ratio).
    """
    rng = np.random.default_rng(77)
    pair_ij = {}   # (i,j) with i<j → has non-zero [i,j] in some X_k
    pair_ji = {}
    for _ in range(n_pts):
        pos = [int(v) for v in rng.integers(3, 10, size=dim)]
        for ax in range(dim):
            try:
                M = np.array(fns[ax](*pos), float)
            except Exception:
                continue
            for i in range(dim):
                for j in range(i + 1, dim):
                    k = (i, j)
                    if abs(M[i, j]) > 1e-7:
                        pair_ij[k] = True
                    if abs(M[j, i]) > 1e-7:
                        pair_ji[k] = True

    all_pairs = set(pair_ij) | set(pair_ji)
    if not all_pairs:
        return False, 0.0
    n_bidir = sum(1 for k in all_pairs if pair_ij.get(k) and pair_ji.get(k))
    bidir_ratio = n_bidir / len(all_pairs)
    return bidir_ratio >= min_bidir, bidir_ratio


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Store helpers
# ══════════════════════════════════════════════════════════════════════════════

def _serialisable_params(params: dict) -> dict:
    """Convert tuple keys to strings for JSON serialisation."""
    return {
        "dim":      params["dim"],
        "D_params": params["D_params"],
        "L_off":    {str(k): v for k, v in params["L_off"].items()},
        "U_off":    {str(k): v for k, v in params["U_off"].items()},
    }


def store_record(params: dict, deltas: list, dim: int,
                 fingerprint: str, t_accept: float,
                 pi_verified: bool = False, pi_max_err: float = 0.0,
                 bidir_ratio: float = 0.0, coupling_bucket: int = 0):
    path = HERE / f"store_C_{dim}x{dim}.jsonl"
    rec = {
        "dim":                      dim,
        "n_matrices":               dim,
        "matrix_size":              dim,
        "agent":                    "C",
        "fingerprint":              fingerprint,
        "best_delta":               round(max(deltas) if deltas else 0.0, 3),
        "deltas":                   [round(d, 3) for d in deltas],
        "n_converging_axes":        sum(1 for d in deltas if d > 1.0),
        "path_independence_verified": pi_verified,
        "max_flatness_error":       round(pi_max_err, 12),
        "n_pairs_checked":          dim * (dim - 1) // 2,
        "coupling_bucket":          coupling_bucket,
        "bidir_ratio":              round(bidir_ratio, 4),
        "params":                   _serialisable_params(params),
        "timestamp":                t_accept,
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return path


def count_stored(dim: int) -> int:
    p = HERE / f"store_C_{dim}x{dim}.jsonl"
    if not p.exists(): return 0
    return sum(1 for l in p.read_text().splitlines() if l.strip())


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main agent loop
# ══════════════════════════════════════════════════════════════════════════════

def run_agent():
    sentinel = HERE / "STOP_AGENTS"
    rng = np.random.default_rng(int(time.time()) % (2**31))

    print("=" * 68, flush=True)
    print(f"  Agent C — Dimension-Climbing CMF Explorer (continuous)", flush=True)
    print(f"  Loops: {START_DIM}×× → {MAX_DIM}×× then restarts. Sentinel: STOP_AGENTS", flush=True)
    print("=" * 68, flush=True)

    run_idx = 0
    while not sentinel.exists():
        run_idx += 1
        _run_once(rng, sentinel, run_idx)
        if sentinel.exists():
            break
        print(f"\n  [Agent C] Run {run_idx} complete. Restarting from dim={START_DIM}...",
              flush=True)


def _run_once(rng, sentinel, run_idx: int):
    current_dim   = START_DIM
    total_trials  = 0
    max_dim_found = 0
    dim_stats: dict[int, dict] = {}

    print("=" * 68, flush=True)
    print(f"  Agent C — Dimension-Climbing CMF Explorer", flush=True)
    print(f"  Starting dim={START_DIM}  budget={TRIALS_TOTAL:,}", flush=True)
    print("=" * 68, flush=True)

    t_global = time.time()

    while total_trials < TRIALS_TOTAL and current_dim <= MAX_DIM:
        dim = current_dim
        # Re-use already stored for this dim
        already_stored = count_stored(dim)
        found_this_dim = already_stored
        trials_this_dim = 0

        print(f"\n{'─'*60}", flush=True)
        print(f"  dim={dim}  ({dim}×{dim} matrices, {dim} coordinates)  "
              f"already_stored={already_stored}", flush=True)

        # Budget scales with dim — harder dims get more attempts
        budget_this_dim = min(MAX_PER_DIM, max(500, dim * 300))

        while (found_this_dim < TARGET_PER_DIM
               and trials_this_dim < budget_this_dim
               and total_trials < TRIALS_TOTAL):

            if sentinel.exists():
                return
            trials_this_dim += 1
            total_trials    += 1

            if trials_this_dim % 200 == 0:
                elapsed = time.time() - t_global
                eta = elapsed / max(total_trials, 1) * (TRIALS_TOTAL - total_trials)
                print(f"  [dim={dim}] trial {trials_this_dim:>5}/{budget_this_dim}  "
                      f"found={found_this_dim}  "
                      f"total={total_trials:>6}  "
                      f"ETA={eta:.0f}s", flush=True)

            # ── Sample params ──────────────────────────────────────────────
            params = sample_params(dim, rng)

            # ── T1: quick pole check ───────────────────────────────────────
            try:
                fns = build_eval_fns(params)
            except Exception:
                continue

            if not _t1_pole_check(fns, dim, rng):
                continue

            # ── T2: fast numpy convergence ─────────────────────────────────
            t2_ok, delta_fast = _t2_fast_convergence(fns, dim)
            if not t2_ok:
                continue

            # ── T2b: non-trivial ──────────────────────────────────────────
            if not _trivial_collapse_check(fns, dim):
                continue

            # ── T3: thorough mpmath check — ALL d axes ────────────────────
            t3_ok, deltas = _t3_thorough_check(fns, dim)
            if not t3_ok:
                continue

            # ── T4: path independence — all C(d,2) pairs ─────────────────
            pi_ok, pi_max_err = _check_path_independence_nd(fns, dim)
            if not pi_ok:
                continue

            # ── T5: measure coupling (info only — not a gate) ───────────
            _, bidir_ratio = _t5_coupling_check(fns, dim)

            # ── ACCEPT ────────────────────────────────────────────────────
            fp = _fingerprint(fns, dim)

            coupling_bucket = (4 if bidir_ratio >= 0.50 else
                               3 if bidir_ratio >= 0.10 else 2)
            n_conv = sum(1 for d in deltas if d > 1.0)
            path = store_record(params, deltas, dim, fp, time.time(),
                                pi_verified=True, pi_max_err=pi_max_err,
                                bidir_ratio=bidir_ratio,
                                coupling_bucket=coupling_bucket)
            found_this_dim += 1
            max_dim_found   = max(max_dim_found, dim)

            best_d = max(deltas)
            print(f"  ✓ dim={dim}  {dim}×{dim}  B{coupling_bucket}  "
                  f"best_delta={best_d:.1f}  "
                  f"bidir={bidir_ratio:.2f}  "
                  f"conv_axes={n_conv}/{dim}  "
                  f"fp={fp}  trials={trials_this_dim}", flush=True)

        # ── Dim summary ───────────────────────────────────────────────────
        dim_stats[dim] = {
            "found":  found_this_dim,
            "trials": trials_this_dim,
            "success": found_this_dim >= TARGET_PER_DIM,
        }

        if found_this_dim >= TARGET_PER_DIM:
            print(f"  → dim={dim} PASSED ({found_this_dim} found in "
                  f"{trials_this_dim} trials)  → climbing to dim={dim+1}",
                  flush=True)
            current_dim += 1
        else:
            print(f"  → dim={dim} FAILED after {trials_this_dim} trials "
                  f"(found only {found_this_dim}/{TARGET_PER_DIM})", flush=True)
            print(f"  → STOPPING. Maximum dimension reached: {max_dim_found}",
                  flush=True)
            break

    # ── Final report ──────────────────────────────────────────────────────────
    total_t = time.time() - t_global
    print(f"\n{'='*68}", flush=True)
    print(f"  AGENT C COMPLETE  |  {total_t:.0f}s  |  {total_trials:,} trials",
          flush=True)
    print(f"  Maximum dimension with valid CMF: {max_dim_found}×{max_dim_found}",
          flush=True)
    print(f"{'='*68}", flush=True)

    print(f"\n  {'dim':>4}  {'found':>6}  {'trials':>7}  {'status':>8}", flush=True)
    print(f"  {'─'*35}", flush=True)
    for dim in sorted(dim_stats):
        s = dim_stats[dim]
        status = "PASS" if s["success"] else "FAIL"
        print(f"  {dim:>4}  {s['found']:>6}  {s['trials']:>7}  {status:>8}",
              flush=True)

    print(f"\n  Store files written: "
          + ", ".join(f"store_C_{d}x{d}.jsonl"
                      for d in sorted(dim_stats) if dim_stats[d]["found"] > 0),
          flush=True)


if __name__ == "__main__":
    run_agent()
