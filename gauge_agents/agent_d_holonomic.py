#!/usr/bin/env python3
"""
agent_d_holonomic.py — Agent D: GENUINE 4-Variable Holonomic CMF Generator
============================================================================
Generates NxN CMFs (N >= 4) using a TRUE 4-variable gauge structure:

    X_i(n) = G(n + e_i) · D_i(n_i) · G(n)^{-1}

where  n = (n1, n2, n3, n4) ∈ Z^4  (four independent lattice dimensions).

KEY DIFFERENCE from Agents A/B/C:
  - 4 independent variables: w, x, y, z
  - D_diag[i] = a_i * (coords[i % 4] + b_i)
    → for dim=4: all 4 coords appear exactly once (Jacobian rank 4 possible)
    → for dim=5: w appears twice, still 4 independent coordinates
  - 4 stepping matrices instead of 3
  - Flatness: C(4,2)=6 commutation pairs instead of 3
  - Walk: 4 pure-axis rays + (1,1,1,1) mixed ray

G structure (same LDU form as Agents A/B/C):
  L = unit lower triangular  (constant rational off-diag entries)
  D_diag = diagonal, entry i = a_i * (coords[i%4] + b_i)
  U = unit upper triangular  (constant rational off-diag entries)

Base matrices D_i = diag(v, v+1, ..., v+dim-1)  where v = n_i, i=0..3.
Flatness is AUTOMATIC by construction (D_i and D_j commute: different variables).

Goal: 1 000 verified 4-variable CMFs per matrix size, then proceed to 5-var.

STOP sentinel: create gauge_agents/STOP_AGENTS to stop cleanly.
"""
from __future__ import annotations
import gc, heapq, json, math, random, sys, time, hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, Rational

sys.path.insert(0, str(Path(__file__).parent))

# ── Config ────────────────────────────────────────────────────────────────────
NVARS        = 4                   # number of independent variables
DIMS         = [4, 5, 6]           # matrix sizes to explore
STORE_CAP    = 1_000               # target: 1 000 per size
MAX_EVALS    = 5_000_000
MIN_DELTA    = 1.5                 # slightly relaxed (4D walks converge slower)
SCORE_THRESH = 0.05
N_FLAT_CHECK = 12
N_POLE_CHECK = 40
BEAM_WIDTH   = 20

HERE  = Path(__file__).parent
STORE = {d: HERE / f"store_D_{d}x{d}.jsonl" for d in DIMS}

# 4 symbolic variables: w, x, y, z
w_s, x_s, y_s, z_s = symbols("w x y z")
_SVARS = [w_s, x_s, y_s, z_s]

# 4 unit shifts in Z^4
SHIFTS_4D = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

_OFF_VALS = [-2.0, -1.0, -0.5, -1/3, 1/3, 0.5, 1.0, 2.0]

_seen: dict[int, set] = {d: set() for d in DIMS}
_family_reg: dict[int, dict[str, int]] = {d: {} for d in DIMS}
_beams: dict[int, list] = {d: [] for d in DIMS}
_beam_uid = [0]


# ── Load previously stored fingerprints ───────────────────────────────────────

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


# ── LDU parameter sampling ─────────────────────────────────────────────────────

def _sample_ldu_params(dim: int, rng: random.Random) -> dict:
    """Sample a random LDU gauge parameter set for NVARS=4 variables."""
    L_off = {}
    for i in range(dim):
        for j in range(i):
            if rng.random() < 0.5:
                L_off[(i, j)] = rng.choice(_OFF_VALS)

    # D_diag[i] = a_i * (coords[i % NVARS] + b_i)
    D_params = []
    for i in range(dim):
        a = float(rng.choice([-2, -1, 1, 2]))
        b = float(rng.choice(range(-2, 3)))
        D_params.append((a, b))

    U_off = {}
    for i in range(dim):
        for j in range(i + 1, dim):
            if rng.random() < 0.5:
                U_off[(i, j)] = rng.choice(_OFF_VALS)

    return {"dim": dim, "nvars": NVARS, "L_off": L_off, "D_params": D_params, "U_off": U_off}


# ── Numeric G and X_i evaluation ──────────────────────────────────────────────

def _G_numpy(params: dict, coords: list) -> np.ndarray:
    """Evaluate G = L * diag(D) * U numerically. coords has length >= NVARS."""
    dim = params["dim"]
    L = np.eye(dim, dtype=float)
    for (i, j), v in params["L_off"].items():
        L[i, j] = v

    D_diag = np.zeros(dim, dtype=float)
    for i, (a, b) in enumerate(params["D_params"]):
        D_diag[i] = a * (coords[i % NVARS] + b)

    U = np.eye(dim, dtype=float)
    for (i, j), v in params["U_off"].items():
        U[i, j] = v

    return L @ np.diag(D_diag) @ U


def _build_numeric_fns(params: dict):
    """
    Returns 4 functions eval_fns[i](w,x,y,z) -> ndarray.
    X_i = G(n+e_i) * D_i(n_i) * G(n)^{-1}
    """
    dim = params["dim"]
    eval_fns = []
    for axis, shift in enumerate(SHIFTS_4D):
        def make_fn(ax=axis, sh=shift):
            def fn(wv, xv, yv, zv):
                c = [wv, xv, yv, zv]
                G_n  = _G_numpy(params, c)
                G_sh = _G_numpy(params, [c[k] + sh[k] for k in range(4)])
                Di   = np.diag([c[ax] + k for k in range(dim)])
                det  = np.linalg.det(G_n)
                if abs(det) < 1e-10:
                    raise ValueError("singular G")
                return (G_sh @ Di @ np.linalg.inv(G_n)).tolist()
            return fn
        eval_fns.append(make_fn())
    return eval_fns


# ── Flatness, pole checks ──────────────────────────────────────────────────────

def _check_flatness(eval_fns: list, n_checks: int = N_FLAT_CHECK) -> bool:
    """
    Flatness: X_i(n+e_j) * X_j(n) = X_j(n+e_i) * X_i(n)  for all i<j.
    6 pairs for 4 variables.  Guaranteed by gauge construction but checked
    numerically to catch singular/degenerate configurations.
    """
    rng_np = np.random.default_rng(17)
    for _ in range(n_checks):
        pos = rng_np.integers(2, 10, 4).tolist()
        for i in range(4):
            for j in range(i + 1, 4):
                try:
                    Xi_n   = np.array(eval_fns[i](*pos), dtype=float)
                    Xj_n   = np.array(eval_fns[j](*pos), dtype=float)
                    pei    = pos.copy(); pei[i] += 1
                    pej    = pos.copy(); pej[j] += 1
                    Xj_nei = np.array(eval_fns[j](*pei), dtype=float)
                    Xi_nej = np.array(eval_fns[i](*pej), dtype=float)
                    # flatness: Xi(n+ej)*Xj(n) == Xj(n+ei)*Xi(n)
                    err = np.max(np.abs(Xi_nej @ Xj_n - Xj_nei @ Xi_n))
                    if err > 1e-5:
                        return False
                except Exception:
                    return False
    return True


def _fast_pole_check(eval_fns: list, n_samples: int = N_POLE_CHECK) -> bool:
    rng_np = np.random.default_rng(0)
    for _ in range(n_samples):
        pos = rng_np.integers(1, 20, 4).tolist()
        for ax in range(4):
            try:
                M = np.array(eval_fns[ax](*pos), dtype=float)
                if not np.all(np.isfinite(M)):
                    return True
                if abs(np.linalg.det(M)) < 1e-9:
                    return True
            except Exception:
                return True
    return False


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def _fingerprint(eval_fns: list, dim: int) -> str:
    rng_np = np.random.default_rng(888)
    vals = []
    for _ in range(12):
        pos = rng_np.integers(3, 15, 4).tolist()
        ax  = int(rng_np.integers(0, 4))
        try:
            M = np.array(eval_fns[ax](*pos), dtype=float)
            vals.extend([round(v, 5) for v in M.ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:16]


def _coarse_fingerprint(eval_fns: list, dim: int) -> str:
    rng_np = np.random.default_rng(888)
    vals = []
    for _ in range(8):
        pos = rng_np.integers(3, 12, 4).tolist()
        ax  = int(rng_np.integers(0, 4))
        try:
            M = np.array(eval_fns[ax](*pos), dtype=float)
            vals.extend([round(v, 1) for v in M.ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:10]


def _novelty_factor(coarse_fp: str, dim: int) -> float:
    n = _family_reg[dim].get(coarse_fp, 0)
    if n == 0:
        return 1.0
    return max(0.10, 1.0 / (1.0 + math.sqrt(float(n))))


# ── Convergence / delta estimation ────────────────────────────────────────────

def _estimate_deltas(eval_fns: list, dim: int) -> dict:
    """
    Walk along multiple 4D rays and compute convergence delta.
    Lower-triangular readout: ratio = P[N-1,0] / P[0,0].

    Returns {"best_delta": float, "deltas": [float, ...], "ratios": [float, ...]}.
    """
    import mpmath as mp
    mp.mp.dps = 40

    # Pure-axis rays + main diagonal
    rays_4d = [
        (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
        (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1),
        (1, 1, 1, 1),
    ]

    deltas, ratios = [], []

    for ray in rays_4d:
        res_by_depth = []
        for depth in [80, 500]:
            P = mp.eye(dim)
            pos = [0, 0, 0, 0]
            for step in range(1, depth + 1):
                ax = step % NVARS
                pos[ax] += ray[ax]
                try:
                    M = mp.matrix(eval_fns[ax](*pos))
                    P = M * P
                    sc = max(abs(P[i, j]) for i in range(dim) for j in range(dim))
                    if sc > mp.mpf(10) ** 30:
                        P /= sc
                except Exception:
                    break
            # Lower-triangular readout: P[N-1,0] / P[0,0]
            if abs(P[0, 0]) > mp.mpf(10) ** -35:
                r = float(mp.re(P[dim - 1, 0] / P[0, 0]))
                res_by_depth.append(r)

        if len(res_by_depth) == 2 and all(math.isfinite(r) for r in res_by_depth):
            diff = abs(res_by_depth[1] - res_by_depth[0])
            delta = min(40.0, -math.log10(diff + 1e-45))
            deltas.append(delta)
            ratios.append(res_by_depth[-1])
        else:
            deltas.append(0.0)
            ratios.append(0.0)

    best_delta = max(deltas) if deltas else 0.0
    return {"best_delta": best_delta, "deltas": deltas, "ratios": ratios}


def _score_from_deltas(delta_result: dict) -> float:
    """Simple score: fraction of rays with delta > MIN_DELTA."""
    d = delta_result["deltas"]
    if not d:
        return 0.0
    good = sum(1 for x in d if x >= MIN_DELTA)
    return good / len(d)


# ── Coupling measurement (off-diagonal structure) ─────────────────────────────

def _measure_bidir(eval_fns: list, dim: int, n_pts: int = 4) -> tuple:
    rng_np = np.random.default_rng(77)
    pair_ij: dict = {}
    pair_ji: dict = {}
    for _ in range(n_pts):
        pos = [int(v) for v in rng_np.integers(3, 12, 4)]
        for ax in range(4):
            try:
                M = np.array(eval_fns[ax](*pos), float)
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


# ── Effective-dimension check (Jacobian rank) ─────────────────────────────────

def _jacobian_rank(eval_fns: list, h: float = 0.01) -> int:
    """
    Quick check: rank of ∂(X0,X1,X2,X3)/∂(w,x,y,z) at a generic point.
    Should be 4 for genuinely 4-variable CMFs.
    """
    pos0 = [5.3, 7.1, 11.7, 4.9]
    try:
        def F(p):
            vecs = []
            for ax in range(4):
                M = np.array(eval_fns[ax](*p), dtype=float)
                vecs.append(M.ravel())
            return np.concatenate(vecs)

        F0 = F(pos0)
        cols = []
        for vi in range(4):
            p1 = pos0.copy(); p1[vi] += h
            cols.append((F(p1) - F0) / h)
        J = np.column_stack(cols)           # (4*N^2, 4)
        _, sv, _ = np.linalg.svd(J, full_matrices=False)
        thresh = 1e-4 * sv[0] if sv[0] > 1e-12 else 1e-8
        return int(np.sum(sv > thresh))
    except Exception:
        return 0


# ── Symbolic reconstruction ────────────────────────────────────────────────────

def _reconstruct_symbolic(params: dict) -> tuple:
    """Build G_sym and X_sym for accepted hits. Skips for dim > 5 (too slow)."""
    dim  = params["dim"]
    if dim > 5:
        return None, None, []

    coords = _SVARS  # [w, x, y, z]

    L_sym = sp.eye(dim)
    for (i, j), v in params["L_off"].items():
        L_sym[i, j] = sp.Rational(v).limit_denominator(8)

    D_entries = []
    for i, (a, b) in enumerate(params["D_params"]):
        v = coords[i % NVARS]
        D_entries.append(sp.Rational(a).limit_denominator(8) * (v + sp.Rational(b).limit_denominator(8)))
    D_sym = sp.diag(*D_entries)

    U_sym = sp.eye(dim)
    for (i, j), v in params["U_off"].items():
        U_sym[i, j] = sp.Rational(v).limit_denominator(8)

    G_sym = L_sym * D_sym * U_sym

    # Canonical base matrices D_i (one per variable)
    D_i_syms = [sp.diag(*[coords[ax] + k for k in range(dim)]) for ax in range(NVARS)]

    try:
        G_inv_sym = G_sym.inv()
    except Exception:
        return G_sym, D_i_syms, []

    Xs = []
    for i, shift in enumerate(SHIFTS_4D):
        subs = list(zip(_SVARS, [_SVARS[k] + shift[k] for k in range(NVARS)]))
        G_sh = G_sym.subs(subs)
        Xi   = sp.simplify(G_sh * D_i_syms[i] * G_inv_sym)
        Xs.append(Xi)

    return G_sym, D_i_syms, Xs


# ── Beam search ────────────────────────────────────────────────────────────────

def _beam_push(dim: int, score: float, params: dict) -> None:
    _beam_uid[0] += 1
    entry = (-score, _beam_uid[0], params)
    heap  = _beams[dim]
    if len(heap) < BEAM_WIDTH:
        heapq.heappush(heap, entry)
    elif score > -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _beam_mutate(dim: int, rng: random.Random) -> Optional[dict]:
    if not _beams[dim]:
        return None
    _, _, src = rng.choice(_beams[dim])
    p = {
        "dim":      dim,
        "nvars":    NVARS,
        "L_off":    dict(src["L_off"]),
        "D_params": list(src["D_params"]),
        "U_off":    dict(src["U_off"]),
    }
    target = rng.choice(["L", "D", "D", "D", "U"])
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
    else:
        i = rng.randint(0, dim - 2)
        j = rng.randint(i + 1, dim - 1)
        k = (i, j)
        if k in p["U_off"] and rng.random() < 0.25:
            del p["U_off"][k]
        else:
            p["U_off"][k] = rng.choice(_OFF_VALS)
    return p


# ── One evaluation attempt ─────────────────────────────────────────────────────

def _try_one(dim: int, rng: random.Random) -> Optional[dict]:
    params = None
    if rng.random() < 0.40:
        params = _beam_mutate(dim, rng)
    if params is None:
        params = _sample_ldu_params(dim, rng)

    eval_fns = _build_numeric_fns(params)

    if _fast_pole_check(eval_fns):
        return None
    if not _check_flatness(eval_fns):
        return None

    fp = _fingerprint(eval_fns, dim)
    if fp in _seen[dim]:
        return None

    cfp     = _coarse_fingerprint(eval_fns, dim)
    nfactor = _novelty_factor(cfp, dim)

    bidir_ratio, coupling_bucket = _measure_bidir(eval_fns, dim)

    delta_result = _estimate_deltas(eval_fns, dim)
    score        = _score_from_deltas(delta_result)
    nov_score    = score * nfactor

    if nov_score < SCORE_THRESH or delta_result["best_delta"] < MIN_DELTA:
        return None

    # Quick effective-dimension check
    jac_rank = _jacobian_rank(eval_fns)

    _beam_push(dim, delta_result["best_delta"], params)

    # Symbolic reconstruction (dim <= 5 only)
    G_sym, D_i_syms, Xs_sym = _reconstruct_symbolic(params)

    def _ser(M):
        return [[str(M[i, j]) for j in range(dim)] for i in range(dim)]

    _family_reg[dim][cfp] = _family_reg[dim].get(cfp, 0) + 1

    rec = {
        "dim":             dim,
        "nvars":           NVARS,
        "n_matrices":      NVARS,
        "matrix_size":     dim,
        "agent":           "D",
        "fingerprint":     fp,
        "coarse_fp":       cfp,
        "effective_vars":  jac_rank,
        "score":           round(score, 4),
        "novelty_score":   round(nov_score, 4),
        "novelty_factor":  round(nfactor, 4),
        "family_count":    _family_reg[dim][cfp],
        "coupling_bucket": coupling_bucket,
        "bidir_ratio":     round(bidir_ratio, 4),
        "best_delta":      delta_result["best_delta"],
        "deltas":          delta_result["deltas"],
        "ratios":          [round(r, 8) for r in delta_result["ratios"]],
        "params": {
            "dim":      dim,
            "nvars":    NVARS,
            "D_params": params["D_params"],
            "L_off":    {str(k): v for k, v in params["L_off"].items()},
            "U_off":    {str(k): v for k, v in params["U_off"].items()},
        },
        "timestamp": time.time(),
    }

    if Xs_sym:
        rec["X0"] = _ser(Xs_sym[0])
        rec["X1"] = _ser(Xs_sym[1]) if len(Xs_sym) > 1 else []
        rec["X2"] = _ser(Xs_sym[2]) if len(Xs_sym) > 2 else []
        rec["X3"] = _ser(Xs_sym[3]) if len(Xs_sym) > 3 else []
        rec["G"]  = _ser(G_sym)
        rec["D0"] = [str(D_i_syms[0][k, k]) for k in range(dim)]
        rec["D1"] = [str(D_i_syms[1][k, k]) for k in range(dim)]
        rec["D2"] = [str(D_i_syms[2][k, k]) for k in range(dim)]
        rec["D3"] = [str(D_i_syms[3][k, k]) for k in range(dim)]

    return rec


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    _load_seen()
    stored  = {d: _count_stored(d) for d in DIMS}
    evals   = 0
    t0      = time.time()
    rng_py  = random.Random()
    sentinel = HERE / "STOP_AGENTS"

    print("=" * 68)
    print("  Agent D — 4-Variable Holonomic CMF Generator")
    print(f"  NVARS=4 (w,x,y,z)  |  dims={DIMS}  |  target={STORE_CAP:,}/dim")
    print(f"  min_delta={MIN_DELTA}  |  score_thresh={SCORE_THRESH}")
    print("=" * 68)
    for d in DIMS:
        print(f"  {d}×{d}: {stored[d]:,} already stored")
    print()

    fh = {d: open(STORE[d], "a") for d in DIMS}

    try:
        while evals < MAX_EVALS:
            if sentinel.exists():
                print("  STOP_AGENTS sentinel — exiting."); break
            evals += 1

            candidates = [d for d in DIMS if stored[d] < STORE_CAP]
            if not candidates:
                print(f"All stores at {STORE_CAP:,}/dim — done."); break
            dim = rng_py.choice(candidates)

            result = _try_one(dim, rng_py)

            if result is not None:
                stored[dim] += 1
                _seen[dim].add(result["fingerprint"])
                fh[dim].write(json.dumps(result) + "\n")
                fh[dim].flush()
                elapsed = time.time() - t0
                ev = result.get("effective_vars", "?")
                print(
                    f"  [D] {dim}×{dim} #{stored[dim]:04d}  "
                    f"vars={ev}/4  "
                    f"B{result.get('coupling_bucket','?')}  "
                    f"delta={result['best_delta']:.2f}  "
                    f"bidir={result.get('bidir_ratio',0):.2f}  "
                    f"({elapsed:.0f}s  eval#{evals})"
                )

            if evals % 500 == 0:
                gc.collect()
                elapsed = time.time() - t0
                print(f"  --- eval {evals}  "
                      f"stored={sum(stored.values()):,}  {elapsed:.0f}s ---")

    finally:
        for fh_ in fh.values():
            fh_.close()

    print(f"\nAgent D done: {evals} evaluations, "
          f"{sum(stored.values())} CMFs stored.")


if __name__ == "__main__":
    main()
