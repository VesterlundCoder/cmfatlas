#!/usr/bin/env python3
"""
agent_holonomic.py — Agent B: LDU Holonomic Proof-Oriented CMF Generator
=========================================================================
Generates 3×3, 4×4, 5×5 CMFs via gauge bootstrap:
    X_i(n) = G(n + e_i) · D_i(n_i) · G(n)^{-1}

ARCHITECTURE: Numeric-first evaluation (same as Agent A).
  G = L * D_diag * U  (LDU factorization), all components with rational parameters.
  All flatness/quality checks are done numerically (fast numpy ops).
  SymPy symbolic reconstruction is done ONLY on accepted hits.

G structure:
  L = unit lower triangular  (constant rational off-diag entries)
  D_diag = diagonal with entries a_i*(coord_i + b_i)  (one coord per slot)
  U = unit upper triangular  (constant rational off-diag entries)

Base matrices D_i = diag(v, v+1, ..., v+dim-1) where v=x/y/z for i=0/1/2.
This is the MINIMAL canonical choice — forces Pochhammer growth, prevents
gauge collapse.

Stores first 1000 verified systems per size in store_B_{dim}x{dim}.jsonl.
Exits after MAX_EVALS for PM2 restart.
"""
from __future__ import annotations
import gc, heapq, json, math, random, sys, time, hashlib
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
B4_BONUS     = 1.15    # score multiplier for Bucket-4 (bidirectionally coupled) hits
N_FLAT_CHECK = 15
N_POLE_CHECK = 50

HERE  = Path(__file__).parent
STORE = {d: HERE / f"store_B_{d}x{d}.jsonl" for d in DIMS}

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


# ── LDU parameter sampling ────────────────────────────────────────────────────
# G = L * D_diag * U
# L[i,j] = l_{ij}  (constant float, unit lower triangular)
# D_diag[i,i] = a_i * (coord_{i%3} + b_i)  (linear, coordinate-dependent)
# U[i,j] = u_{ij}  (constant float, unit upper triangular)

# Richer off-diagonal values: includes ±1/3 from make_ldu_gauge in cmf_core.py
_OFF_VALS = [-2.0, -1.0, -0.5, -1/3, 1/3, 0.5, 1.0, 2.0]


def _sample_ldu_params(dim: int, rng: random.Random) -> dict:
    # L: unit lower triangular — off-diag entries are small rationals (incl. ±1/3)
    L_off = {}   # (i,j) -> float  for i > j
    for i in range(dim):
        for j in range(i):
            if rng.random() < 0.5:
                L_off[(i,j)] = rng.choice(_OFF_VALS)

    # D_diag: diagonal — entry i = a_i * (coord_{i%3} + b_i)
    D_params = []
    for i in range(dim):
        a = float(rng.choice([-2,-1,1,2]))
        b = float(rng.choice(range(-2, 3)))
        D_params.append((a, b))   # D[i,i] = a*(coord_{i%3} + b)

    # U: unit upper triangular — off-diag entries are small rationals (incl. ±1/3)
    U_off = {}   # (i,j) -> float  for i < j
    for i in range(dim):
        for j in range(i+1, dim):
            if rng.random() < 0.5:
                U_off[(i,j)] = rng.choice(_OFF_VALS)

    return {"dim": dim, "L_off": L_off, "D_params": D_params, "U_off": U_off}


def _G_numpy_ldu(params: dict, xv: float, yv: float, zv: float) -> np.ndarray:
    """Evaluate G = L * diag(D) * U numerically."""
    dim    = params["dim"]
    coords = [xv, yv, zv]

    # Build L
    L = np.eye(dim, dtype=float)
    for (i, j), v in params["L_off"].items():
        L[i, j] = v

    # Build D diagonal
    D_diag = np.zeros(dim, dtype=float)
    for i, (a, b) in enumerate(params["D_params"]):
        D_diag[i] = a * (coords[i % 3] + b)

    # Build U
    U = np.eye(dim, dtype=float)
    for (i, j), v in params["U_off"].items():
        U[i, j] = v

    return L @ np.diag(D_diag) @ U


# ── Canonical D_i base matrices (numeric) ─────────────────────────────────────

def _Di_numpy_canonical(dim: int, axis: int, coord_val: float) -> np.ndarray:
    """
    D_i = diag(v, v+1, v+2, ..., v+dim-1)  where v = coord_val.
    Minimal canonical: forces Pochhammer/(v+k)! growth.
    """
    return np.diag([coord_val + k for k in range(dim)])


# ── Build numeric CMF eval functions ─────────────────────────────────────────

def _build_numeric_fns_ldu(params: dict):
    """
    Returns 3 functions eval_fns[i](xv,yv,zv) -> ndarray.
    X_i = G(n+e_i) * D_i * G(n)^{-1}
    """
    dim = params["dim"]
    shifts = [(1,0,0), (0,1,0), (0,0,1)]
    eval_fns = []
    for i, (sx, sy, sz) in enumerate(shifts):
        def make_fn(si=(sx,sy,sz), axis=i):
            def fn(xv, yv, zv):
                G_n  = _G_numpy_ldu(params, xv, yv, zv)
                G_sh = _G_numpy_ldu(params, xv+si[0], yv+si[1], zv+si[2])
                Di   = _Di_numpy_canonical(dim, axis, [xv,yv,zv][axis])
                det_G = np.linalg.det(G_n)
                if abs(det_G) < 1e-10:
                    raise ValueError("singular G")
                return (G_sh @ Di @ np.linalg.inv(G_n)).tolist()
            return fn
        eval_fns.append(make_fn())
    return eval_fns


# ── Flatness, pole check, fingerprint (shared with Agent A style) ─────────────

def _check_flatness(eval_fns: list, n_checks: int = N_FLAT_CHECK) -> bool:
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
                    err = np.max(np.abs(Xi_nej @ Xj_n - Xj_nei @ Xi_n))
                    if err > 1e-5:
                        return False
                except Exception:
                    return False
    return True


def _fast_pole_check(eval_fns: list, n_samples: int = N_POLE_CHECK) -> bool:
    rng_np = np.random.default_rng(0)
    for _ in range(n_samples):
        pos = rng_np.integers(1, 20, 3).tolist()
        for ax in range(3):
            try:
                M = np.array(eval_fns[ax](*pos), dtype=float)
                if not np.all(np.isfinite(M)): return True
                if abs(np.linalg.det(M)) < 1e-9: return True
            except Exception:
                return True
    return False


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


def _estimate_deltas(eval_fns: list, dim: int) -> list:
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


def _make_mfn(eval_fns: list):
    def fn(pos, axis): return eval_fns[axis % 3](*pos[:3])
    return fn


# ── Beam search (ported from cmf_gauge_agents/agent_holonomic.py) ─────────────
# Maintains a max-heap (via negated score) of top-BEAM_WIDTH best param dicts
# per dim. _try_one exploits the beam 40% of the time by mutating a top entry.

_BEAM_WIDTH = 20
_beams: dict[int, list] = {d: [] for d in DIMS}  # heap entries: (-score, uid, params)
_beam_uid = [0]  # mutable counter for unique heap IDs


def _beam_push(dim: int, score: float, params: dict) -> None:
    _beam_uid[0] += 1
    entry = (-score, _beam_uid[0], params)
    heap = _beams[dim]
    if len(heap) < _BEAM_WIDTH:
        heapq.heappush(heap, entry)
    elif score > -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _beam_mutate(dim: int, rng: random.Random) -> Optional[dict]:
    """Return a mutated copy of a randomly chosen top-beam entry, or None."""
    if not _beams[dim]:
        return None
    _, _, src = rng.choice(_beams[dim])
    p = {
        "dim":      dim,
        "L_off":    dict(src["L_off"]),
        "D_params": list(src["D_params"]),
        "U_off":    dict(src["U_off"]),
    }
    # mutate one randomly chosen component
    target = rng.choice(["L", "D", "U", "D", "D"])  # weight D higher
    if target == "D":
        i = rng.randint(0, dim - 1)
        a, b = p["D_params"][i]
        if rng.random() < 0.5:
            a = float(rng.choice([-2, -1, 1, 2]))
        else:
            b = float(rng.choice(range(-2, 3)))
        p["D_params"][i] = (a, b)
    elif target == "L":
        i = rng.randint(1, dim - 1)
        j = rng.randint(0, i - 1)
        k = (i, j)
        if k in p["L_off"] and rng.random() < 0.25:
            del p["L_off"][k]
        else:
            p["L_off"][k] = rng.choice(_OFF_VALS)
    else:  # U
        i = rng.randint(0, dim - 2)
        j = rng.randint(i + 1, dim - 1)
        k = (i, j)
        if k in p["U_off"] and rng.random() < 0.25:
            del p["U_off"][k]
        else:
            p["U_off"][k] = rng.choice(_OFF_VALS)
    return p


# ── Symbolic reconstruction (only on hits) ────────────────────────────────────

def _dfinite_score(Xs_sym: list) -> float:
    """Reward entries that are products of linear factors (Pochhammer-type)."""
    score, total = 0.0, 0
    for M in Xs_sym:
        for e in M:
            total += 1
            try:
                f = sp.factor(e)
                num, den = sp.fraction(f)
                def _lf(expr):
                    if isinstance(expr, sp.Mul):
                        return sum(_lf(a) for a in expr.args)
                    if isinstance(expr, sp.Pow):
                        b, ex = expr.args
                        if isinstance(ex, sp.Integer) and ex > 0:
                            return int(ex) * _lf(b)
                    if isinstance(expr, sp.Add):
                        try:
                            p = sp.Poly(expr, *_SVARS)
                            if p.total_degree() == 1: return 1
                        except Exception: pass
                    return 0
                score += min(1.0, (_lf(num) + _lf(den)) / 4.0)
            except Exception:
                pass
    return score / max(total, 1)


def _reconstruct_symbolic(params: dict) -> tuple:
    """Build G_sym and X_sym only for accepted hits."""
    dim    = params["dim"]
    coords = [x_s, y_s, z_s]
    shifts = [(1,0,0),(0,1,0),(0,0,1)]

    # L
    L_sym = sp.eye(dim)
    for (i, j), v in params["L_off"].items():
        L_sym[i, j] = sp.Rational(v).limit_denominator(8)

    # D_diag
    D_entries = []
    for i, (a, b) in enumerate(params["D_params"]):
        v = coords[i % 3]
        D_entries.append(sp.Rational(a).limit_denominator(8) * (v + sp.Rational(b).limit_denominator(8)))
    D_sym = sp.diag(*D_entries)

    # U
    U_sym = sp.eye(dim)
    for (i, j), v in params["U_off"].items():
        U_sym[i, j] = sp.Rational(v).limit_denominator(8)

    G_sym = L_sym * D_sym * U_sym

    # D_i canonical
    D_i_syms = [sp.diag(*[coords[ax] + k for k in range(dim)]) for ax in range(3)]

    # X_i
    try:
        G_inv_sym = G_sym.inv()
    except Exception:
        return G_sym, D_i_syms, []

    Xs = []
    for i, (sx, sy, sz) in enumerate(shifts):
        G_sh = G_sym.subs([(x_s, x_s+sx),(y_s, y_s+sy),(z_s, z_s+sz)])
        Xi   = sp.simplify(G_sh * D_i_syms[i] * G_inv_sym)
        Xs.append(Xi)
    return G_sym, D_i_syms, Xs


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
    # 40% exploit beam (hill-climb), 60% random exploration
    params = None
    if rng.random() < 0.40:
        params = _beam_mutate(dim, rng)
    if params is None:
        params = _sample_ldu_params(dim, rng)
    eval_fns = _build_numeric_fns_ldu(params)

    if _fast_pole_check(eval_fns):
        return None
    if not _check_flatness(eval_fns):
        return None

    fp = _fingerprint(eval_fns, dim)
    if fp in _seen[dim]:
        return None

    # Novelty check: coarse family fingerprint
    cfp = _coarse_fingerprint(eval_fns, dim)
    nfactor = _novelty_factor(cfp, dim)

    # Coupling measurement (B4 bonus)
    bidir_ratio, coupling_bucket = _measure_bidir(eval_fns, dim)
    b4_bonus = B4_BONUS if coupling_bucket == 4 else 1.0

    mfn    = _make_mfn(eval_fns)
    result = evaluate(mfn, dim, sympy_matrices=None, n_rays=6, dps=40)
    # Apply novelty penalty + B4 bonus to score
    novelty_score = result["score"] * nfactor * b4_bonus
    if novelty_score < SCORE_THRESH or result["best_delta"] < MIN_DELTA:
        return None

    # Feed into beam so future iterations can hill-climb from this point
    _beam_push(dim, result["best_delta"], params)

    # Symbolic reconstruction — only for accepted hits
    G_sym, D_i_syms, Xs_sym = _reconstruct_symbolic(params)
    df_score = _dfinite_score(Xs_sym) if Xs_sym else 0.0

    def _ser(M):
        return [[str(M[i,j]) for j in range(dim)] for i in range(dim)]

    # Update family registry in-memory
    _family_reg[dim][cfp] = _family_reg[dim].get(cfp, 0) + 1

    return {
        "dim":            dim,
        "agent":          "B",
        "fingerprint":    fp,
        "coarse_fp":      cfp,
        "score":          result["score"],
        "novelty_score":  round(novelty_score, 4),
        "novelty_factor": round(nfactor, 4),
        "family_count":   _family_reg[dim][cfp],
        "coupling_bucket": coupling_bucket,
        "bidir_ratio":    round(bidir_ratio, 4),
        "dfinite_score":  round(df_score, 4),
        "best_delta":     result["best_delta"],
        "deltas":         result["deltas"],
        "ratios":         result["ratios"],
        "conv_rate":      result["conv_rate"],
        "ray_stability":  result["ray_stability"],
        "identifiability": result["identifiability"],
        "simplicity":     result["simplicity"],
        "proofability":   result["proofability"],
        "X0": _ser(Xs_sym[0]) if Xs_sym else [],
        "X1": _ser(Xs_sym[1]) if len(Xs_sym) > 1 else [],
        "X2": _ser(Xs_sym[2]) if len(Xs_sym) > 2 else [],
        "G":  _ser(G_sym),
        "D0": [str(D_i_syms[0][k,k]) for k in range(dim)],
        "D1": [str(D_i_syms[1][k,k]) for k in range(dim)],
        "D2": [str(D_i_syms[2][k,k]) for k in range(dim)],
        "params":  {"L_off": {str(k): v for k,v in params["L_off"].items()},
                    "D_params": params["D_params"],
                    "U_off":  {str(k): v for k,v in params["U_off"].items()}},
        "timestamp": time.time(),
    }

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    _load_seen()
    stored = {d: _count_stored(d) for d in DIMS}
    evals  = 0
    t0     = time.time()
    rng_py = random.Random()
    sentinel = HERE / "STOP_AGENTS"

    print("=" * 60)
    print("  Agent B — Holonomic LDU Proof-Oriented CMF Generator")
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
                print(f"  [B] {dim}×{dim} #{stored[dim]:05d}  "
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

    print(f"\nAgent B done: {evals} evaluations.")


if __name__ == "__main__":
    main()
