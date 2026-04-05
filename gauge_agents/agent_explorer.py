#!/usr/bin/env python3
"""
agent_explorer.py — Agent A: Free-Roaming Gauge-Bootstrap CMF Explorer
=======================================================================
Generates 3×3, 4×4, 5×5 CMFs via:
    X_i(n) = G(n + e_i) · D_i(n_i) · G(n)^{-1}

ARCHITECTURE: Numeric-first evaluation.
  G is parameterised as {diagonal params} + {sparse off-diagonal params}.
  Fast numpy eval is used for flatness/quality checks (99% of work).
  SymPy symbolic reconstruction is done ONLY on accepted hits.

D_i are diagonal with POLYNOMIAL entries in coordinate i (no constants),
forcing Pochhammer/factorial growth — prevents trivial gauge collapse.

Stores first 1000 verified non-trivial, non-duplicate systems per size
in gauge_agents/store_A_{dim}x{dim}.jsonl. Exits after MAX_EVALS.
"""
from __future__ import annotations
import gc, json, math, random, sys, time, hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, Rational

sys.path.insert(0, str(Path(__file__).parent))
from reward_engine import evaluate, check_poles

# ── Config ────────────────────────────────────────────────────────────────────
DIMS         = [3, 4, 5]
STORE_CAP    = 50_000
MAX_EVALS    = 10_000_000
MIN_DELTA    = 2.0
SCORE_THRESH = 0.08
B4_BONUS     = 1.15    # score multiplier for Bucket-4 hits
N_FLAT_CHECK = 15           # lattice points for flatness verification
N_POLE_CHECK = 50           # lattice points for pole check

HERE  = Path(__file__).parent
STORE = {d: HERE / f"store_A_{d}x{d}.jsonl" for d in DIMS}

x_s, y_s, z_s = symbols("x y z")
_SVARS = [x_s, y_s, z_s]

_seen: dict[int, set] = {d: set() for d in DIMS}
_family_reg: dict[int, dict[str, int]] = {d: {} for d in DIMS}  # coarse_fp -> count


def _coarse_fingerprint(eval_fns: list, dim: int) -> str:
    """Low-precision fingerprint: groups near-duplicate families."""
    rng_np = np.random.default_rng(888)
    vals = []
    for _ in range(8):
        pos = rng_np.integers(3, 12, 3).tolist()
        ax  = int(rng_np.integers(0, 3))
        try:
            M = np.array(eval_fns[ax](*pos), dtype=float)
            vals.extend([round(v, 1) for v in M.ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:10]


def _novelty_factor(coarse_fp: str, dim: int) -> float:
    """
    Diminishing return factor based on family population.
    0 already known  -> 1.0 (full score)
    1 known          -> 0.70
    5 known          -> 0.45
    20+ known        -> 0.15
    """
    n = _family_reg[dim].get(coarse_fp, 0)
    if n == 0:
        return 1.0
    return max(0.10, 1.0 / (1.0 + math.sqrt(float(n))))


def _load_seen():
    for d in DIMS:
        if STORE[d].exists():
            for line in STORE[d].read_text().splitlines():
                try:
                    rec = json.loads(line)
                    _seen[d].add(rec.get("fingerprint", ""))
                    cfp = rec.get("coarse_fp", "")
                    if cfp:
                        _family_reg[d][cfp] = _family_reg[d].get(cfp, 0) + 1
                except Exception:
                    pass

def _count_stored(dim: int) -> int:
    p = STORE[dim]
    return sum(1 for ln in p.read_text().splitlines() if ln.strip()) if p.exists() else 0


# ── Parametric G representation ───────────────────────────────────────────────
# G(x,y,z) = diag(a0*(x+b0), a1*(y+b1), a2*(z+b2), ...)
#           + sparse off-diagonal: entry [i,j] = c * var_k  (c rational, var_k one of x,y,z)
# All parameters stored as plain floats for fast numpy eval.

def _sample_params(dim: int, rng: random.Random) -> dict:
    """Sample G parameters. Returns dict with 'diag' and 'offdiag' lists."""
    # diagonal: each entry is (a_i, b_i) for a_i*(var_i + b_i), var cycles x,y,z
    diag = []
    for i in range(dim):
        a = rng.choice([-3,-2,-1,1,2,3]) + rng.choice([0, 0.5, -0.5])
        b = rng.choice(range(-3, 4)) + rng.choice([0, 0.5])
        diag.append((float(a), float(b)))

    # off-diagonal: sparse list of (i, j, c, var_idx) with i != j
    n_off = rng.randint(0, min(dim, 4))    # 0..4 off-diagonal entries
    offdiag = []
    used = set()
    for _ in range(n_off * 5):             # try to pick unique (i,j) pairs
        if len(offdiag) >= n_off: break
        i = rng.randint(0, dim-1)
        j = rng.randint(0, dim-1)
        if i == j or (i,j) in used: continue
        used.add((i,j))
        c = rng.choice([-2,-1,1,2]) * rng.choice([1, 0.5])
        v = rng.randint(0, 2)
        offdiag.append((i, j, float(c), v))

    return {"diag": diag, "offdiag": offdiag, "dim": dim}


def _G_numpy(params: dict, xv: float, yv: float, zv: float) -> np.ndarray:
    """Evaluate G numerically at (xv, yv, zv)."""
    dim   = params["dim"]
    coords = [xv, yv, zv]
    G = np.zeros((dim, dim), dtype=float)
    for i, (a, b) in enumerate(params["diag"]):
        v = coords[i % 3]
        G[i, i] = a * (v + b)
    for (i, j, c, vi) in params["offdiag"]:
        G[i, j] = c * coords[vi]
    return G


def _Di_numpy(dim: int, axis: int, coord_val: float,
              d_params: list) -> np.ndarray:
    """
    D_i = diag(a_k*(v + b_k + k) : k=0..dim-1)  where v = coord_val.
    d_params[axis] = list of (a_k, b_k).
    NEVER constant — forces factorial growth.
    """
    D = np.zeros((dim, dim), dtype=float)
    for k, (a, b) in enumerate(d_params[axis]):
        D[k, k] = a * (coord_val + b + k)
    return D


def _sample_d_params(dim: int, rng: random.Random) -> list:
    """Returns list of 3 lists, each with dim (a_k, b_k) pairs."""
    result = []
    for _ in range(3):
        axis_params = []
        for k in range(dim):
            a = float(rng.choice([-2,-1,1,2]))
            b = float(rng.choice(range(-2, 3)))
            axis_params.append((a, b))
        result.append(axis_params)
    return result


# ── Build numeric CMF eval functions ─────────────────────────────────────────

def _build_numeric_fns(params: dict, d_params: list):
    """
    Returns list of 3 functions: eval_fns[i](xv,yv,zv) -> dim×dim ndarray.
    X_i(n) = G(n+e_i) * D_i(n_i) * G(n)^{-1}
    """
    dim = params["dim"]
    shifts = [(1,0,0), (0,1,0), (0,0,1)]
    eval_fns = []
    for i, (sx, sy, sz) in enumerate(shifts):
        def make_fn(si=(sx,sy,sz), axis=i):
            def fn(xv, yv, zv):
                G_n    = _G_numpy(params, xv, yv, zv)
                G_sh   = _G_numpy(params, xv+si[0], yv+si[1], zv+si[2])
                Di     = _Di_numpy(dim, axis, [xv,yv,zv][axis], d_params)
                det_G  = np.linalg.det(G_n)
                if abs(det_G) < 1e-10:
                    raise ValueError("singular G")
                G_inv  = np.linalg.inv(G_n)
                return (G_sh @ Di @ G_inv).tolist()
            return fn
        eval_fns.append(make_fn())
    return eval_fns


# ── Flatness verification (numeric) ──────────────────────────────────────────

def _check_flatness(eval_fns: list, n_checks: int = N_FLAT_CHECK) -> bool:
    """X_i(n+e_j)*X_j(n) = X_j(n+e_i)*X_i(n)  — gauge bootstrap convention."""
    rng_np = np.random.default_rng(17)
    shifts = [(1,0,0), (0,1,0), (0,0,1)]
    for _ in range(n_checks):
        pos = rng_np.integers(2, 10, 3).tolist()
        for i in range(3):
            for j in range(i+1, 3):
                try:
                    Xi_n   = np.array(eval_fns[i](*pos), dtype=float)
                    Xj_n   = np.array(eval_fns[j](*pos), dtype=float)
                    pei    = [pos[0]+shifts[i][0], pos[1]+shifts[i][1], pos[2]+shifts[i][2]]
                    pej    = [pos[0]+shifts[j][0], pos[1]+shifts[j][1], pos[2]+shifts[j][2]]
                    Xj_nei = np.array(eval_fns[j](*pei), dtype=float)
                    Xi_nej = np.array(eval_fns[i](*pej), dtype=float)
                    # Gauge bootstrap: X_i(n+e_j)*X_j(n) = X_j(n+e_i)*X_i(n)
                    err = np.max(np.abs(Xi_nej @ Xj_n - Xj_nei @ Xi_n))
                    if err > 1e-5:
                        return False
                except Exception:
                    return False
    return True


# ── Pole check (numeric) ──────────────────────────────────────────────────────

def _fast_pole_check(eval_fns: list, n_samples: int = N_POLE_CHECK) -> bool:
    rng_np = np.random.default_rng(0)
    for _ in range(n_samples):
        pos = rng_np.integers(1, 20, 3).tolist()
        for ax in range(3):
            try:
                M = np.array(eval_fns[ax](*pos), dtype=float)
                if not np.all(np.isfinite(M)):
                    return True
                if abs(np.linalg.det(M)) < 1e-9:
                    return True
            except Exception:
                return True
    return False


# ── Fingerprint ────────────────────────────────────────────────────────────────

def _fingerprint(eval_fns: list, dim: int) -> str:
    rng_np = np.random.default_rng(888)
    vals = []
    for _ in range(12):
        pos = rng_np.integers(3, 15, 3).tolist()
        ax  = int(rng_np.integers(0, 3))
        try:
            M = np.array(eval_fns[ax](*pos), dtype=float)
            vals.extend([round(v, 5) for v in M.ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:16]


# ── Delta estimation (numeric) ────────────────────────────────────────────────

def _estimate_deltas(eval_fns: list, dim: int) -> list:
    """Quick multi-ray delta estimation using short vs long product walks."""
    import mpmath as mp
    mp.mp.dps = 40
    deltas = []
    rays   = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
    for ray in rays:
        ratios = []
        for depth in [100, 600]:
            P = mp.eye(dim)
            pos = [0, 0, 0]
            for step in range(1, depth+1):
                ax = step % 3
                pos[ax] += ray[ax]
                try:
                    M = mp.matrix(eval_fns[ax](*pos))
                    P = M * P
                    sc = max(abs(P[i,j]) for i in range(dim) for j in range(dim))
                    if sc > mp.mpf(10)**30: P /= sc
                except Exception:
                    break
            if abs(P[dim-1, 0]) > mp.mpf(10)**-35:
                ratios.append(float(P[0,0]/P[dim-1,0]))
        if len(ratios) == 2 and math.isfinite(ratios[0]) and math.isfinite(ratios[1]):
            diff = abs(ratios[1] - ratios[0])
            deltas.append(min(40.0, -math.log10(diff + 1e-45)))
        else:
            deltas.append(0.0)
    return deltas


# ── Symbolic reconstruction (only on hits) ────────────────────────────────────

def _reconstruct_symbolic(params: dict, d_params: list) -> tuple:
    """Build G_sym and X_sym only when we have a confirmed hit."""
    dim    = params["dim"]
    coords = [x_s, y_s, z_s]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]

    # G symbolic
    G_entries = [[sp.Integer(0)]*dim for _ in range(dim)]
    for i, (a, b) in enumerate(params["diag"]):
        v = coords[i % 3]
        G_entries[i][i] = sp.Rational(a).limit_denominator(8) * (v + sp.Rational(b).limit_denominator(8))
    for (i, j, c, vi) in params["offdiag"]:
        G_entries[i][j] = sp.Rational(c).limit_denominator(8) * coords[vi]
    G_sym = Matrix(G_entries)

    # D_i symbolic
    D_syms = []
    for axis in range(3):
        v = coords[axis]
        diag_entries = []
        for k, (a, b) in enumerate(d_params[axis]):
            diag_entries.append(
                sp.Rational(a).limit_denominator(8) * (v + sp.Rational(b).limit_denominator(8) + k)
            )
        D_syms.append(sp.diag(*diag_entries))

    # X_i symbolic
    try:
        G_inv_sym = G_sym.inv()
    except Exception:
        return G_sym, D_syms, []

    Xs = []
    for i, (sx, sy, sz) in enumerate(shifts):
        G_sh = G_sym.subs([(x_s, x_s+sx),(y_s, y_s+sy),(z_s, z_s+sz)])
        Xi   = sp.simplify(G_sh * D_syms[i] * G_inv_sym)
        Xs.append(Xi)
    return G_sym, D_syms, Xs


def _ser_matrix(M: Matrix, dim: int) -> list:
    return [[str(M[i,j]) for j in range(dim)] for i in range(dim)]


def _measure_bidir(eval_fns: list, dim: int, n_pts: int = 4) -> tuple:
    """Fast numpy bidir_ratio for (x,y,z)-signature eval_fns. Returns (bidir_ratio, bucket)."""
    rng_np = np.random.default_rng(77)
    pair_ij: dict = {}
    pair_ji: dict = {}
    for _ in range(n_pts):
        x, y, z = [int(v) for v in rng_np.integers(3, 12, 3)]
        for ax in range(min(len(eval_fns), 3)):
            try:
                M = np.array(eval_fns[ax](x, y, z), float)
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
        return 0.0, 2
    n_bidir = sum(1 for k in all_pairs if pair_ij.get(k) and pair_ji.get(k))
    bidir_ratio = n_bidir / len(all_pairs)
    bucket = 4 if bidir_ratio >= 0.50 else 3 if bidir_ratio >= 0.10 else 2
    return bidir_ratio, bucket


# ── One evaluation attempt ────────────────────────────────────────────────────

def _try_one(dim: int, rng: random.Random) -> Optional[dict]:
    params   = _sample_params(dim, rng)
    d_params = _sample_d_params(dim, rng)

    # Build numeric eval functions (cheap)
    eval_fns = _build_numeric_fns(params, d_params)

    # Fast pole check
    if _fast_pole_check(eval_fns):
        return None

    # Flatness spot-check (always holds by construction — quick sanity)
    if not _check_flatness(eval_fns):
        return None

    # Fingerprint dedup
    fp = _fingerprint(eval_fns, dim)
    if fp in _seen[dim]:
        return None

    # Novelty check: coarse family fingerprint
    cfp = _coarse_fingerprint(eval_fns, dim)
    nfactor = _novelty_factor(cfp, dim)

    # Coupling measurement (B4 bonus)
    bidir_ratio, coupling_bucket = _measure_bidir(eval_fns, dim)
    b4_bonus = B4_BONUS if coupling_bucket == 4 else 1.0

    # Full reward evaluation (handles delta + ISC scoring)
    def mfn(pos, axis):
        return eval_fns[axis % 3](*pos[:3])
    result = evaluate(mfn, dim, sympy_matrices=None, n_rays=6, dps=40)
    # Apply novelty penalty + B4 bonus to score
    novelty_score = result["score"] * nfactor * b4_bonus
    if novelty_score < SCORE_THRESH or result["best_delta"] < MIN_DELTA:
        return None

    # Update family registry in-memory
    _family_reg[dim][cfp] = _family_reg[dim].get(cfp, 0) + 1

    # Symbolic reconstruction — done only for accepted hits
    G_sym, D_syms, Xs_sym = _reconstruct_symbolic(params, d_params)

    def _ser(M):
        return [[str(M[i,j]) for j in range(dim)] for i in range(dim)]

    return {
        "dim":            dim,
        "agent":          "A",
        "fingerprint":    fp,
        "coarse_fp":      cfp,
        "score":          result["score"],
        "novelty_score":  round(novelty_score, 4),
        "novelty_factor": round(nfactor, 4),
        "family_count":   _family_reg[dim][cfp],
        "coupling_bucket": coupling_bucket,
        "bidir_ratio":    round(bidir_ratio, 4),
        "best_delta":    result["best_delta"],
        "deltas":        result["deltas"],
        "ratios":        result["ratios"],
        "conv_rate":     result["conv_rate"],
        "ray_stability": result["ray_stability"],
        "identifiability": result["identifiability"],
        "simplicity":    result["simplicity"],
        "proofability":  result["proofability"],
        "X0": _ser(Xs_sym[0]) if Xs_sym else [],
        "X1": _ser(Xs_sym[1]) if len(Xs_sym) > 1 else [],
        "X2": _ser(Xs_sym[2]) if len(Xs_sym) > 2 else [],
        "G":  _ser(G_sym),
        "D0": [str(D_syms[0][i,i]) for i in range(dim)],
        "D1": [str(D_syms[1][i,i]) for i in range(dim)],
        "D2": [str(D_syms[2][i,i]) for i in range(dim)],
        "params":   {"diag": params["diag"], "offdiag": params["offdiag"]},
        "d_params": d_params,
        "timestamp": time.time(),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    _load_seen()
    stored   = {d: _count_stored(d) for d in DIMS}
    evals    = 0
    t0       = time.time()
    rng_py   = random.Random()
    sentinel = HERE / "STOP_AGENTS"

    print("=" * 60)
    print("  Agent A — Free-Roaming Gauge-Bootstrap CMF Explorer")
    print(f"  Running until {STORE_CAP:,}/dim or STOP_AGENTS sentinel  |  min_delta={MIN_DELTA}")
    print("=" * 60)
    for d in DIMS:
        print(f"  {d}×{d}: {stored[d]:,} already stored")

    fh = {d: open(STORE[d], "a") for d in DIMS}

    try:
        while evals < MAX_EVALS:
            if sentinel.exists():
                print("  STOP_AGENTS sentinel detected — exiting cleanly."); break
            evals += 1

            candidates = [d for d in DIMS if stored[d] < STORE_CAP]
            if not candidates:
                print(f"All stores full ({STORE_CAP:,}/dim) — exiting."); break
            dim = rng_py.choice(candidates)

            result = _try_one(dim, rng_py)

            if result is not None:
                stored[dim] += 1
                _seen[dim].add(result["fingerprint"])
                fh[dim].write(json.dumps(result) + "\n")
                fh[dim].flush()
                elapsed = time.time() - t0
                b = result.get("coupling_bucket", "?")
                print(f"  [A] {dim}×{dim} #{stored[dim]:05d}  "
                      f"B{b}  score={result['score']:.4f}  "
                      f"delta={result['best_delta']:.2f}  "
                      f"bidir={result.get('bidir_ratio',0):.2f}  "
                      f"({elapsed:.0f}s  eval#{evals})")

            if evals % 500 == 0:
                gc.collect()
                elapsed = time.time() - t0
                total_stored = sum(stored.values())
                print(f"  --- eval {evals}  stored={total_stored:,}  {elapsed:.0f}s ---")

    finally:
        for fh_ in fh.values():
            fh_.close()

    print(f"\nAgent A done: {evals} evaluations.")


if __name__ == "__main__":
    main()
