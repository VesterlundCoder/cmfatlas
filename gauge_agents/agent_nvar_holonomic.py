#!/usr/bin/env python3
"""
agent_nvar_holonomic.py — Generalized N-Variable Holonomic CMF Generator
=========================================================================
Generates NxN CMFs with a TRUE K-variable gauge structure (K=5,6,...,10):

    X_i(n) = G(n + e_i) · D_i(n_i) · G(n)^{-1}
    n = (n_0, ..., n_{K-1}) ∈ Z^K

G structure (LDU form):
  L = unit lower triangular  (constant rational off-diag)
  D_diag[i] = a_i * (coords[i % K] + b_i)
              → uses coords cycling through K distinct variables
  U = unit upper triangular  (constant rational off-diag)

Base D_i = diag(n_i, n_i+1, ..., n_i+dim-1)  (Pochhammer growth in each var)
Flatness: K*(K-1)/2 commutation pairs — AUTOMATIC by construction.

Effective-variable guarantee:
  For dim >= K: each coordinate appears at least once in D_diag → rank K.
  For dim < K:  only dim variables active (wrap avoided by DIMS constraint).

Usage:
  python3 agent_nvar_holonomic.py --nvars 5 [--dims 5,6,7] [--cap 1000]
  python3 agent_nvar_holonomic.py --nvars 6 [--dims 6,7,8]
  ...
  python3 agent_nvar_holonomic.py --nvars 10 [--dims 10,11,12]

Agent letter mapping:  nvars 4=D, 5=E, 6=F, 7=G, 8=H, 9=I, 10=J
STOP sentinel: create gauge_agents/STOP_AGENTS
"""
from __future__ import annotations
import argparse, gc, heapq, json, math, random, sys, time, hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import sympy as sp
from sympy import symbols

HERE = Path(__file__).parent

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nvars", type=int, default=5,
                   help="Number of independent variables (4=D,5=E,...,10=J)")
    p.add_argument("--dims", type=str, default="",
                   help="Comma-sep matrix sizes, default = nvars,nvars+1,nvars+2")
    p.add_argument("--cap", type=int, default=1000, help="Store capacity per dim")
    p.add_argument("--min-delta", type=float, default=1.2)
    p.add_argument("--score-thresh", type=float, default=0.04)
    p.add_argument("--max-evals", type=int, default=5_000_000)
    return p.parse_args()


_AGENT_LETTER = {4:"D", 5:"E", 6:"F", 7:"G", 8:"H", 9:"I", 10:"J"}
_OFF_VALS = [-2.0, -1.0, -0.5, -1/3, 1/3, 0.5, 1.0, 2.0]

# ── Variable symbols (up to 10) ───────────────────────────────────────────────
_ALL_SYMS = list(symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9"))
# Human-readable names for small K
_VAR_NAMES = {
    4: list(symbols("w x y z")),
    5: list(symbols("v w x y z")),
    6: list(symbols("u v w x y z")),
}


def _get_syms(nvars: int):
    return _VAR_NAMES.get(nvars, _ALL_SYMS[:nvars])


# ── LDU param sampling ────────────────────────────────────────────────────────

def _sample_params(dim: int, nvars: int, rng: random.Random) -> dict:
    L_off = {}
    for i in range(dim):
        for j in range(i):
            if rng.random() < 0.5:
                L_off[(i, j)] = rng.choice(_OFF_VALS)

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

    return {"dim": dim, "nvars": nvars, "L_off": L_off, "D_params": D_params, "U_off": U_off}


# ── G evaluation ──────────────────────────────────────────────────────────────

def _G_numpy(params: dict, coords: list) -> np.ndarray:
    dim, nvars = params["dim"], params["nvars"]
    L = np.eye(dim, dtype=float)
    for (i, j), v in params["L_off"].items():
        L[i, j] = v
    D_diag = np.array([a * (coords[i % nvars] + b)
                        for i, (a, b) in enumerate(params["D_params"])], dtype=float)
    U = np.eye(dim, dtype=float)
    for (i, j), v in params["U_off"].items():
        U[i, j] = v
    return L @ np.diag(D_diag) @ U


def _build_eval_fns(params: dict):
    dim, nvars = params["dim"], params["nvars"]
    shifts = [tuple(1 if k == ax else 0 for k in range(nvars)) for ax in range(nvars)]
    fns = []
    for axis, shift in enumerate(shifts):
        def make_fn(ax=axis, sh=shift):
            def fn(*coords):
                c = list(coords)
                G_n  = _G_numpy(params, c)
                G_sh = _G_numpy(params, [c[k] + sh[k] for k in range(nvars)])
                Di   = np.diag([c[ax] + k for k in range(dim)])
                det  = np.linalg.det(G_n)
                if abs(det) < 1e-10:
                    raise ValueError("singular")
                return (G_sh @ Di @ np.linalg.inv(G_n)).tolist()
            return fn
        fns.append(make_fn())
    return fns


# ── Quality checks ────────────────────────────────────────────────────────────

def _check_flatness(fns: list, nvars: int, n_checks: int = 10) -> bool:
    rng_np = np.random.default_rng(17)
    for _ in range(n_checks):
        pos = rng_np.integers(2, 10, nvars).tolist()
        for i in range(nvars):
            for j in range(i + 1, nvars):
                try:
                    Xi = np.array(fns[i](*pos), dtype=float)
                    Xj = np.array(fns[j](*pos), dtype=float)
                    pei = pos.copy(); pei[i] += 1
                    pej = pos.copy(); pej[j] += 1
                    Xj_ei = np.array(fns[j](*pei), dtype=float)
                    Xi_ej = np.array(fns[i](*pej), dtype=float)
                    if np.max(np.abs(Xi_ej @ Xj - Xj_ei @ Xi)) > 1e-5:
                        return False
                except Exception:
                    return False
    return True


def _pole_check(fns: list, nvars: int, n: int = 30) -> bool:
    rng_np = np.random.default_rng(0)
    for _ in range(n):
        pos = rng_np.integers(1, 20, nvars).tolist()
        for ax in range(nvars):
            try:
                M = np.array(fns[ax](*pos), dtype=float)
                if not np.all(np.isfinite(M)) or abs(np.linalg.det(M)) < 1e-9:
                    return True
            except Exception:
                return True
    return False


# ── Fingerprinting ────────────────────────────────────────────────────────────

def _fp(fns: list, dim: int, nvars: int, prec: int = 5) -> str:
    rng_np = np.random.default_rng(888)
    vals = []
    for _ in range(12):
        pos = rng_np.integers(3, 15, nvars).tolist()
        ax  = int(rng_np.integers(0, nvars))
        try:
            M = np.array(fns[ax](*pos), dtype=float)
            vals.extend([round(v, prec) for v in M.ravel()])
        except Exception:
            vals.extend([0.0] * dim * dim)
    return hashlib.md5(json.dumps(vals).encode()).hexdigest()[:16]


def _coarse_fp(fns: list, dim: int, nvars: int) -> str:
    return _fp(fns, dim, nvars, prec=1)[:10]


# ── Delta estimation ──────────────────────────────────────────────────────────

def _estimate_deltas(fns: list, dim: int, nvars: int) -> dict:
    import mpmath as mp
    mp.mp.dps = 40

    # Pure-axis rays + all-ones diagonal
    rays = [tuple(1 if k == ax else 0 for k in range(nvars)) for ax in range(nvars)]
    rays.append(tuple(1 for _ in range(nvars)))  # (1,1,1,...,1) diagonal

    deltas, ratios = [], []
    for ray in rays:
        res_pair = []
        for depth in [60, 400]:
            P = mp.eye(dim)
            pos = [0] * nvars
            for step in range(1, depth + 1):
                ax = step % nvars
                pos[ax] += ray[ax]
                try:
                    M = mp.matrix(fns[ax](*pos))
                    P = M * P
                    sc = max(abs(P[i, j]) for i in range(dim) for j in range(dim))
                    if sc > mp.mpf(10) ** 30:
                        P /= sc
                except Exception:
                    break
            if abs(P[0, 0]) > mp.mpf(10) ** -35:
                r = float(mp.re(P[dim - 1, 0] / P[0, 0]))
                res_pair.append(r)
        if len(res_pair) == 2 and all(math.isfinite(r) for r in res_pair):
            diff = abs(res_pair[1] - res_pair[0])
            deltas.append(min(40.0, -math.log10(diff + 1e-45)))
            ratios.append(res_pair[-1])
        else:
            deltas.append(0.0)
            ratios.append(0.0)

    return {"best_delta": max(deltas) if deltas else 0.0,
            "deltas": deltas, "ratios": ratios}


def _score(dr: dict, min_delta: float) -> float:
    d = dr["deltas"]
    return sum(1 for x in d if x >= min_delta) / len(d) if d else 0.0


# ── Bidir coupling ────────────────────────────────────────────────────────────

def _bidir(fns: list, dim: int, nvars: int) -> tuple:
    rng_np = np.random.default_rng(77)
    pij, pji = {}, {}
    for _ in range(4):
        pos = rng_np.integers(3, 12, nvars).tolist()
        for ax in range(nvars):
            try:
                M = np.array(fns[ax](*pos), float)
            except Exception:
                continue
            for i in range(dim):
                for j in range(i + 1, dim):
                    if abs(M[i, j]) > 1e-7: pij[(i, j)] = True
                    if abs(M[j, i]) > 1e-7: pji[(i, j)] = True
    all_p = set(pij) | set(pji)
    if not all_p:
        return 0.0, 2
    n_bi = sum(1 for k in all_p if pij.get(k) and pji.get(k))
    r = n_bi / len(all_p)
    return r, 4 if r >= 0.5 else 3 if r >= 0.1 else 2


# ── Jacobian rank ──────────────────────────────────────────────────────────────

def _jac_rank(fns: list, nvars: int, h: float = 0.01) -> int:
    pos0 = [5.3 + k * 1.7 for k in range(nvars)]
    try:
        def F(p):
            vecs = [np.array(fns[ax](*p), dtype=float).ravel() for ax in range(nvars)]
            return np.concatenate(vecs)
        F0 = F(pos0)
        cols = []
        for vi in range(nvars):
            p1 = pos0.copy(); p1[vi] += h
            cols.append((F(p1) - F0) / h)
        J = np.column_stack(cols)
        _, sv, _ = np.linalg.svd(J, full_matrices=False)
        thresh = 1e-4 * sv[0] if sv[0] > 1e-12 else 1e-8
        return int(np.sum(sv > thresh))
    except Exception:
        return 0


# ── Symbolic reconstruction (small dims only) ─────────────────────────────────

def _symbolic(params: dict, syms: list) -> tuple:
    dim, nvars = params["dim"], params["nvars"]
    if dim > 5:
        return None, None, []

    L_sym = sp.eye(dim)
    for (i, j), v in params["L_off"].items():
        L_sym[i, j] = sp.Rational(v).limit_denominator(8)

    D_entries = [sp.Rational(a).limit_denominator(8) *
                 (syms[i % nvars] + sp.Rational(b).limit_denominator(8))
                 for i, (a, b) in enumerate(params["D_params"])]
    D_sym = sp.diag(*D_entries)

    U_sym = sp.eye(dim)
    for (i, j), v in params["U_off"].items():
        U_sym[i, j] = sp.Rational(v).limit_denominator(8)

    G_sym = L_sym * D_sym * U_sym
    Di_syms = [sp.diag(*[syms[ax] + k for k in range(dim)]) for ax in range(nvars)]

    try:
        G_inv = G_sym.inv()
    except Exception:
        return G_sym, Di_syms, []

    Xs = []
    for ax in range(nvars):
        shift = [1 if k == ax else 0 for k in range(nvars)]
        subs  = list(zip(syms, [syms[k] + shift[k] for k in range(nvars)]))
        G_sh  = G_sym.subs(subs)
        Xs.append(sp.simplify(G_sh * Di_syms[ax] * G_inv))
    return G_sym, Di_syms, Xs


# ── Beam ──────────────────────────────────────────────────────────────────────

class Beam:
    def __init__(self, width=20):
        self._heap = []
        self._uid  = 0
        self._w    = width

    def push(self, score: float, params: dict):
        self._uid += 1
        e = (-score, self._uid, params)
        if len(self._heap) < self._w:
            heapq.heappush(self._heap, e)
        elif score > -self._heap[0][0]:
            heapq.heapreplace(self._heap, e)

    def mutate(self, rng: random.Random) -> Optional[dict]:
        if not self._heap:
            return None
        _, _, src = rng.choice(self._heap)
        nvars = src["nvars"]
        dim   = src["dim"]
        p = {"dim": dim, "nvars": nvars,
             "L_off": dict(src["L_off"]),
             "D_params": list(src["D_params"]),
             "U_off": dict(src["U_off"])}
        target = rng.choice(["L", "D", "D", "D", "U"])
        if target == "D":
            i = rng.randint(0, dim - 1)
            a, b = p["D_params"][i]
            p["D_params"][i] = (float(rng.choice([-2,-1,1,2])) if rng.random() < .5 else a,
                                float(rng.choice(range(-2,3)))  if rng.random() >= .5 else b)
        elif target == "L" and dim > 1:
            i = rng.randint(1, dim - 1); j = rng.randint(0, i - 1)
            k = (i, j)
            if k in p["L_off"] and rng.random() < .25: del p["L_off"][k]
            else: p["L_off"][k] = rng.choice(_OFF_VALS)
        else:
            if dim > 1:
                i = rng.randint(0, dim - 2); j = rng.randint(i + 1, dim - 1)
                k = (i, j)
                if k in p["U_off"] and rng.random() < .25: del p["U_off"][k]
                else: p["U_off"][k] = rng.choice(_OFF_VALS)
        return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    nvars = args.nvars
    if nvars < 4 or nvars > 10:
        print(f"nvars must be 4..10, got {nvars}"); sys.exit(1)

    dims = [int(x) for x in args.dims.split(",") if x.strip()] if args.dims else \
           [nvars, nvars + 1, nvars + 2]
    # Enforce min dim >= nvars (need all K variables to appear in D_diag)
    dims = [d for d in dims if d >= nvars]
    if not dims:
        print(f"All dims < nvars={nvars}, aborting"); sys.exit(1)

    letter   = _AGENT_LETTER.get(nvars, f"N{nvars}")
    store    = {d: HERE / f"store_{letter}_{d}x{d}.jsonl" for d in dims}
    syms     = _get_syms(nvars)
    sentinel = HERE / "STOP_AGENTS"

    # Load seen fingerprints
    seen:   dict[int, set] = {d: set() for d in dims}
    famreg: dict[int, dict[str, int]] = {d: {} for d in dims}
    for d in dims:
        if store[d].exists():
            for ln in store[d].read_text().splitlines():
                try:
                    r = json.loads(ln)
                    seen[d].add(r.get("fingerprint",""))
                    cfp = r.get("coarse_fp","")
                    if cfp: famreg[d][cfp] = famreg[d].get(cfp,0)+1
                except Exception: pass

    stored = {d: sum(1 for ln in store[d].read_text().splitlines() if ln.strip())
              if store[d].exists() else 0 for d in dims}

    beams = {d: Beam() for d in dims}

    print("=" * 68)
    print(f"  Agent {letter} — {nvars}-Variable Holonomic CMF Generator")
    print(f"  Variables: {', '.join(str(s) for s in syms)}")
    print(f"  dims={dims}  |  target={args.cap:,}/dim  |  min_delta={args.min_delta}")
    print("=" * 68)
    for d in dims:
        print(f"  {d}×{d}: {stored[d]:,} already stored")
    print()

    fh = {d: open(store[d], "a") for d in dims}
    t0 = time.time()
    rng = random.Random()
    evals = 0

    try:
        while evals < args.max_evals:
            if sentinel.exists():
                print("  STOP_AGENTS sentinel — exiting."); break

            candidates = [d for d in dims if stored[d] < args.cap]
            if not candidates:
                print(f"All stores at {args.cap:,}/dim — done."); break

            evals += 1
            dim = rng.choice(candidates)

            # Exploit or explore
            params = beams[dim].mutate(rng) if rng.random() < 0.4 else None
            if params is None:
                params = _sample_params(dim, nvars, rng)

            fns = _build_eval_fns(params)

            if _pole_check(fns, nvars): continue
            if not _check_flatness(fns, nvars): continue

            fingerprint = _fp(fns, dim, nvars)
            if fingerprint in seen[dim]: continue

            cfp     = _coarse_fp(fns, dim, nvars)
            nfact   = max(0.1, 1.0 / (1.0 + math.sqrt(float(famreg[dim].get(cfp, 0)))))
            bidir_r, bucket = _bidir(fns, dim, nvars)
            dr      = _estimate_deltas(fns, dim, nvars)
            score   = _score(dr, args.min_delta)
            nov_sc  = score * nfact

            if nov_sc < args.score_thresh or dr["best_delta"] < args.min_delta:
                continue

            jrank = _jac_rank(fns, nvars)
            beams[dim].push(dr["best_delta"], params)

            G_sym, Di_syms, Xs_sym = _symbolic(params, syms)
            def _ser(M):
                return [[str(M[i, j]) for j in range(dim)] for i in range(dim)]

            famreg[dim][cfp] = famreg[dim].get(cfp, 0) + 1
            stored[dim] += 1
            seen[dim].add(fingerprint)

            rec = {
                "dim": dim, "nvars": nvars,
                "n_matrices": nvars, "matrix_size": dim,
                "agent": letter,
                "fingerprint": fingerprint, "coarse_fp": cfp,
                "effective_vars": jrank,
                "score": round(score, 4), "novelty_score": round(nov_sc, 4),
                "family_count": famreg[dim][cfp],
                "coupling_bucket": bucket, "bidir_ratio": round(bidir_r, 4),
                "best_delta": dr["best_delta"],
                "deltas": dr["deltas"],
                "ratios": [round(r, 8) for r in dr["ratios"]],
                "params": {
                    "dim": dim, "nvars": nvars,
                    "D_params": params["D_params"],
                    "L_off": {str(k): v for k, v in params["L_off"].items()},
                    "U_off": {str(k): v for k, v in params["U_off"].items()},
                },
                "timestamp": time.time(),
            }
            if Xs_sym:
                for ax, X in enumerate(Xs_sym):
                    rec[f"X{ax}"] = _ser(X)
                rec["G"]  = _ser(G_sym)
                for ax in range(nvars):
                    rec[f"D{ax}"] = [str(Di_syms[ax][k, k]) for k in range(dim)]

            fh[dim].write(json.dumps(rec) + "\n")
            fh[dim].flush()

            elapsed = time.time() - t0
            print(f"  [{letter}] {dim}×{dim} #{stored[dim]:04d}  "
                  f"vars={jrank}/{nvars}  B{bucket}  "
                  f"delta={dr['best_delta']:.2f}  bidir={bidir_r:.2f}  "
                  f"({elapsed:.0f}s  eval#{evals})")

            if evals % 500 == 0:
                gc.collect()
                print(f"  --- eval {evals}  stored={sum(stored.values()):,}  "
                      f"{time.time()-t0:.0f}s ---")

    finally:
        for f in fh.values():
            f.close()

    print(f"\nAgent {letter} done: {evals} evaluations, "
          f"{sum(stored.values())} CMFs stored.")


if __name__ == "__main__":
    main()
