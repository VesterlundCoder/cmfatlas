#!/usr/bin/env python3
"""
jacobian_rank_analysis.py — Effective-Dimension Audit of All Gauge CMFs
========================================================================
Tests the effective variable-dimension of all A/B/C gauge CMFs via the
Jacobian rank criterion:

  F(x,y,z) = flatten[X0(x,y,z), X1(x,y,z), X2(x,y,z)]
  J = ∂F/∂(x,y,z)   — shape (3*N^2, 3)
  rank(J) = number of non-negligible singular values

  rank 3 → genuinely 3-variable family  (keep)
  rank 2 → only 2 independent combinations of (x,y,z)  (flag)
  rank 1 → effectively 1-variable  (delete candidate)

Also reports per-matrix separability:
  sep_i = True  if X_i depends only on coord_i (∂X_i/∂other ≈ 0)

Run from the cmf_atlas root:
  python3 gauge_agents/jacobian_rank_analysis.py [--sample N]
"""
from __future__ import annotations
import argparse, ast, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

# ── Param normalisation (handles string keys from C-series) ──────────────────

def _parse_params(raw: dict, dim: int) -> dict:
    L_off, U_off = {}, {}
    for k, v in raw.get("L_off", {}).items():
        key = ast.literal_eval(k) if isinstance(k, str) else k
        L_off[tuple(key)] = float(v)
    for k, v in raw.get("U_off", {}).items():
        key = ast.literal_eval(k) if isinstance(k, str) else k
        U_off[tuple(key)] = float(v)
    dp = [(float(a), float(b)) for a, b in raw.get("D_params", [])]
    return {"dim": dim, "L_off": L_off, "D_params": dp, "U_off": U_off}


def _G(params: dict, coords: list) -> np.ndarray:
    dim = params["dim"]
    L = np.eye(dim)
    for (i, j), v in params["L_off"].items():
        L[i, j] = v
    D = np.array([a * (coords[i % 3] + b) for i, (a, b) in enumerate(params["D_params"])])
    U = np.eye(dim)
    for (i, j), v in params["U_off"].items():
        U[i, j] = v
    return L @ np.diag(D) @ U


_SHIFTS_3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def _Xi(params: dict, axis: int, x: float, y: float, z: float) -> np.ndarray:
    dim = params["dim"]
    c = [x, y, z]
    sh = _SHIFTS_3[axis]
    G_n  = _G(params, c)
    G_sh = _G(params, [c[k] + sh[k] for k in range(3)])
    Di   = np.diag([c[axis] + k for k in range(dim)])
    det  = np.linalg.det(G_n)
    if abs(det) < 1e-12:
        raise ValueError("singular G")
    return G_sh @ Di @ np.linalg.inv(G_n)


# ── Jacobian rank via finite differences ─────────────────────────────────────

def _eval_all(params: dict, x: float, y: float, z: float) -> np.ndarray:
    """Return flattened [X0, X1, X2] at (x,y,z)."""
    vecs = []
    for ax in range(3):
        vecs.append(_Xi(params, ax, x, y, z).ravel())
    return np.concatenate(vecs)


def jacobian_rank(params: dict,
                  pts: list | None = None,
                  h: float = 0.001) -> tuple[int, list[float]]:
    """
    Compute rank of J = ∂F/∂(x,y,z) at one or more generic float points.
    Returns (rank, singular_values_at_first_point).
    """
    if pts is None:
        pts = [(5.3, 7.1, 11.7), (8.9, 4.2, 6.5), (3.7, 12.3, 9.1)]

    ranks = []
    svs_out = None
    for (x0, y0, z0) in pts:
        try:
            F0 = _eval_all(params, x0, y0, z0)
            Fx = (_eval_all(params, x0 + h, y0, z0) - F0) / h
            Fy = (_eval_all(params, x0, y0 + h, z0) - F0) / h
            Fz = (_eval_all(params, x0, y0, z0 + h) - F0) / h
            J = np.column_stack([Fx, Fy, Fz])           # (3*N^2, 3)
            _, sv, _ = np.linalg.svd(J, full_matrices=False)
            thresh = 1e-4 * sv[0] if sv[0] > 1e-12 else 1e-8
            r = int(np.sum(sv > thresh))
            ranks.append(r)
            if svs_out is None:
                svs_out = sv.tolist()
        except Exception:
            pass

    return (max(ranks) if ranks else 0), (svs_out or [])


def separability_check(params: dict, x0=5.3, y0=7.1, z0=11.7, h=0.001) -> dict:
    """
    For each X_i, check which variables it truly depends on.
    Returns {axis: {var: bool}}.
    """
    result = {}
    for ax in range(3):
        dep = {}
        try:
            F0 = _Xi(params, ax, x0, y0, z0).ravel()
            for var_idx, var_name in enumerate(["x", "y", "z"]):
                c = [x0, y0, z0]
                c[var_idx] += h
                Fv = _Xi(params, ax, *c).ravel()
                grad_norm = np.linalg.norm((Fv - F0) / h)
                dep[var_name] = bool(grad_norm > 1e-3)
        except Exception:
            dep = {"x": None, "y": None, "z": None}
        result[ax] = dep
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Jacobian rank audit of gauge CMF store files")
    parser.add_argument("--sample", type=int, default=200,
                        help="Max records to sample per store file (0=all)")
    parser.add_argument("--out", type=str, default="jacobian_rank_results.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()

    store_files = sorted(HERE.glob("store_[ABC]_*.jsonl"))
    if not store_files:
        print("No store files found in", HERE); sys.exit(1)

    out_path = HERE / args.out
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_total = 0
    n_errors = 0

    low_rank_fps = []   # fingerprints with rank < 3

    print(f"{'File':<30} {'N_sampled':>10} {'rank0':>7} {'rank1':>7} {'rank2':>7} {'rank3':>7}")
    print("-" * 75)

    with open(out_path, "w") as fout:
        for sf in store_files:
            lines = [l.strip() for l in sf.read_text(errors="replace").splitlines() if l.strip()]
            if args.sample > 0 and len(lines) > args.sample:
                # Random sample
                rng = np.random.default_rng(42)
                idx = rng.choice(len(lines), args.sample, replace=False)
                lines = [lines[i] for i in sorted(idx)]

            file_ranks = {0: 0, 1: 0, 2: 0, 3: 0}

            for ln in lines:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue

                dim = int(rec.get("dim") or rec.get("matrix_size", 0))
                if dim < 2:
                    continue

                fp = rec.get("fingerprint", "")
                raw_params = rec.get("params")

                # Build params dict (handle both inline and nested)
                if raw_params is None:
                    # A/B series have params as top-level keys
                    raw_params = {
                        "D_params": rec.get("D_params") or [],
                        "L_off":    rec.get("L_off") or {},
                        "U_off":    rec.get("U_off") or {},
                    }
                    if not raw_params["D_params"]:
                        n_errors += 1
                        continue

                try:
                    params = _parse_params(raw_params, dim)
                    if len(params["D_params"]) != dim:
                        n_errors += 1
                        continue

                    rank, svs = jacobian_rank(params)
                    sep = separability_check(params)
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1
                    file_ranks[rank] = file_ranks.get(rank, 0) + 1
                    n_total += 1

                    row = {
                        "fingerprint": fp,
                        "dim": dim,
                        "store_file": sf.name,
                        "jacobian_rank": rank,
                        "singular_values": [round(s, 6) for s in svs[:3]],
                        "separability": {str(k): v for k, v in sep.items()},
                    }
                    fout.write(json.dumps(row) + "\n")

                    if rank < 3:
                        low_rank_fps.append(fp)

                except Exception as e:
                    n_errors += 1

            print(f"{sf.name:<30} {sum(file_ranks.values()):>10} "
                  f"{file_ranks.get(0,0):>7} {file_ranks.get(1,0):>7} "
                  f"{file_ranks.get(2,0):>7} {file_ranks.get(3,0):>7}")

    print("\n" + "=" * 75)
    print(f"TOTAL ANALYSED : {n_total}")
    print(f"Errors/skipped : {n_errors}")
    print(f"Rank 0 (const) : {rank_counts.get(0,0)}")
    print(f"Rank 1 (1-var) : {rank_counts.get(1,0)}")
    print(f"Rank 2 (2-var) : {rank_counts.get(2,0)}")
    print(f"Rank 3 (3-var) : {rank_counts.get(3,0)}")
    print("=" * 75)

    if low_rank_fps:
        print(f"\n⚠  {len(low_rank_fps)} CMF(s) with Jacobian rank < 3:")
        for fp in low_rank_fps[:20]:
            print(f"   {fp}")
        if len(low_rank_fps) > 20:
            print(f"   ... and {len(low_rank_fps)-20} more")
        low_path = HERE / "low_rank_fingerprints.txt"
        low_path.write_text("\n".join(low_rank_fps))
        print(f"   (saved to {low_path.name})")
    else:
        print("\n✓  All sampled CMFs have Jacobian rank = 3 — all are genuinely 3-variable.")
        print("   NOTE: Current agents A/B/C always produce rank-3 CMFs by construction")
        print("   (D_diag cycles through x/y/z with non-zero coefficients).")
        print("   To get genuinely 4-variable CMFs, use Agent D (agent_d_holonomic.py).")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
