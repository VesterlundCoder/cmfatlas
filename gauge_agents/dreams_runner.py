#!/usr/bin/env python3
"""
dreams_runner.py — Ramanujan Dreams: PSLQ constant identification on B3/B4 CMFs
=================================================================================

For each B3/B4 CMF in the classified stores:
  1. Rebuild eval functions from stored params
  2. Walk the CMF along 6 rays to depth 1200 (mpmath, dps=60)
  3. Run PSLQ against a basis of 25 known constants
  4. Report any identified limits

Target constants:
  π, π², π³, π⁴, π⁵
  log(2), log(3), log(5)
  ζ(2)=π²/6, ζ(3), ζ(4)=π⁴/90, ζ(5), ζ(7)
  Catalan G, e, γ (Euler-Mascheroni)
  √2, √3, √5
  1 (identity)

Outputs:
  dreams_hits.jsonl   — identified constants with CMF fingerprint + formula
  dreams_report.md    — human-readable summary
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Optional

import mpmath as mp
import numpy as np

HERE    = Path(__file__).parent
OUT_DIR = HERE / "pipeline_out"
OUT_DIR.mkdir(exist_ok=True)

# ── Walk config ────────────────────────────────────────────────────────────────
WALK_DEPTH = 1200
DPS        = 60
PSLQ_TOL   = 1e-12
RAYS_3     = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
RAYS_D     = [(1,) + (0,) * (d-1) for d in range(3, 15)]   # unit vectors for C hits


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Constant basis
# ══════════════════════════════════════════════════════════════════════════════

def _make_focused_basis(dps: int) -> tuple[list, list]:
    """Small 8-element basis — minimal PSLQ overfitting risk."""
    mp.mp.dps = dps + 20
    vals = [
        mp.mpf(1), mp.pi, mp.pi**2,
        mp.log(2), mp.zeta(3), mp.zeta(5),
        mp.catalan, mp.sqrt(2),
    ]
    names = ["1", "π", "π²", "log(2)", "ζ(3)", "ζ(5)", "G", "√2"]
    return vals, names


def _make_basis(dps: int) -> tuple[list, list]:
    """Return (values_mp, names) for the PSLQ basis."""
    mp.mp.dps = dps + 20
    pi   = mp.pi
    log2 = mp.log(2)
    log3 = mp.log(3)
    log5 = mp.log(5)
    zeta = mp.zeta
    G    = mp.catalan
    e    = mp.e
    gam  = mp.euler   # Euler-Mascheroni

    vals = [
        mp.mpf(1),
        pi, pi**2, pi**3, pi**4, pi**5,
        log2, log3, log5,
        zeta(3), zeta(5), zeta(7),
        G,
        e, gam,
        mp.sqrt(2), mp.sqrt(3), mp.sqrt(5),
        pi * log2,
        pi**2 * log2,
        zeta(3) / pi**2,
        zeta(5) / pi**4,
        log2**2,
        pi * G,
    ]
    names = [
        "1",
        "π", "π²", "π³", "π⁴", "π⁵",
        "log(2)", "log(3)", "log(5)",
        "ζ(3)", "ζ(5)", "ζ(7)",
        "G",
        "e", "γ",
        "√2", "√3", "√5",
        "π·log(2)",
        "π²·log(2)",
        "ζ(3)/π²",
        "ζ(5)/π⁴",
        "log(2)²",
        "π·G",
    ]
    return vals, names


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Eval-function rebuilders (same as coupling_classifier)
# ══════════════════════════════════════════════════════════════════════════════

def _pk(s):
    return tuple(int(x.strip()) for x in str(s).strip("()").split(","))


def _build_fns_ldu(rec: dict):
    dim    = rec["dim"]
    params = rec["params"]
    L_off  = {_pk(k): float(v) for k, v in params.get("L_off", {}).items()}
    U_off  = {_pk(k): float(v) for k, v in params.get("U_off", {}).items()}
    D_pars = [(float(a), float(b)) for a, b in params["D_params"]]

    def G_fn(pos):
        L = np.eye(dim, dtype=float)
        for (i, j), v in L_off.items(): L[i, j] = v
        d_diag = np.array([a * (pos[k % dim] + b) for k, (a, b) in enumerate(D_pars)])
        U = np.eye(dim, dtype=float)
        for (i, j), v in U_off.items(): U[i, j] = v
        return L @ np.diag(d_diag) @ U

    fns = []
    for ax in range(dim):
        def make_fn(axis=ax):
            def fn(*pos):
                pos = list(pos)
                Gn  = G_fn(pos)
                Gs  = G_fn([p + (1 if k == axis else 0) for k, p in enumerate(pos)])
                Di  = np.diag([pos[axis] + k for k in range(dim)])
                det = np.linalg.det(Gn)
                if abs(det) < 1e-9: raise ValueError("singular")
                return Gs @ Di @ np.linalg.inv(Gn)
            return fn
        fns.append(make_fn())
    return fns, dim


def _build_fns_ab(rec: dict):
    """Rebuild for Agent A or B (3-coord format). Returns (fns, n_ax=3)."""
    agent  = rec.get("agent", "B")
    dim    = rec["dim"]
    params = rec["params"]

    if agent == "A":
        diag    = params["diag"]
        offdiag = params["offdiag"]
        d_params= rec.get("d_params", [])

        def G_fn(x, y, z):
            coords = [x, y, z]
            G = np.zeros((dim, dim), float)
            for i, (a, b) in enumerate(diag):
                G[i, i] = float(a) * (coords[i % 3] + float(b))
            for e in offdiag:
                i2, j2, c, vi = int(e[0]), int(e[1]), float(e[2]), int(e[3])
                G[i2, j2] = c * coords[vi % 3]
            return G

        def Di_fn(axis, v):
            if d_params and axis < len(d_params):
                dp = d_params[axis]
                return np.diag([float(a)*(v + float(b) + k) for k,(a,b) in enumerate(dp)])
            return np.diag([v + k for k in range(dim)])

        shifts = [(1,0,0),(0,1,0),(0,0,1)]
        fns = []
        for ax in range(3):
            def mk(axis=ax, sh=shifts[ax]):
                def fn(x, y, z):
                    Gn = G_fn(x, y, z)
                    Gs = G_fn(x+sh[0], y+sh[1], z+sh[2])
                    Di = Di_fn(axis, [x,y,z][axis])
                    det = np.linalg.det(Gn)
                    if abs(det) < 1e-9: raise ValueError("singular")
                    return Gs @ Di @ np.linalg.inv(Gn)
                return fn
            fns.append(mk())
        return fns, 3

    else:  # Agent B — LDU 3-coord
        L_off  = {_pk(k): float(v) for k, v in params.get("L_off", {}).items()}
        U_off  = {_pk(k): float(v) for k, v in params.get("U_off", {}).items()}
        D_pars = [(float(a), float(b)) for a, b in params["D_params"]]

        def G_fn(x, y, z):
            coords = [x, y, z]
            L = np.eye(dim, float)
            for (i, j), v in L_off.items(): L[i, j] = v
            D = np.array([a*(coords[k % 3]+b) for k,(a,b) in enumerate(D_pars)])
            U = np.eye(dim, float)
            for (i, j), v in U_off.items(): U[i, j] = v
            return L @ np.diag(D) @ U

        shifts = [(1,0,0),(0,1,0),(0,0,1)]
        fns = []
        for ax in range(3):
            def mk(axis=ax, sh=shifts[ax]):
                def fn(x, y, z):
                    Gn = G_fn(x, y, z)
                    Gs = G_fn(x+sh[0], y+sh[1], z+sh[2])
                    Di = np.diag([([x,y,z][axis]) + k for k in range(dim)])
                    det = np.linalg.det(Gn)
                    if abs(det) < 1e-9: raise ValueError("singular")
                    return Gs @ Di @ np.linalg.inv(Gn)
                return fn
            fns.append(mk())
        return fns, 3


def build_fns(rec: dict):
    agent = rec.get("agent", "B")
    if agent == "C":
        return _build_fns_ldu(rec)
    return _build_fns_ab(rec)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  mpmath walk
# ══════════════════════════════════════════════════════════════════════════════

def _walk_mp(fns, dim: int, n_ax: int, ray, start: list, depth: int, dps: int
             ) -> Optional[mp.mpf]:
    mp.mp.dps = dps + 10
    pos = list(start)
    v = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)

    for step in range(1, depth + 1):
        ax = step % n_ax
        pos[ax] += ray[ax] if ax < len(ray) else 0
        try:
            Mr = fns[ax](*pos)
            M  = mp.matrix([[mp.mpf(str(float(Mr[r][c] if hasattr(Mr[r], '__len__') else Mr[r,c])))
                             for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            return None
        sc = max(abs(v[i]) for i in range(dim))
        if sc > mp.power(10, 30): v /= sc
        elif sc < mp.power(10, -30): return None

    if abs(v[dim-1]) < mp.power(10, -(dps - 8)):
        return None
    return v[0] / v[dim-1]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PSLQ identification
# ══════════════════════════════════════════════════════════════════════════════

def _pslq_identify(val: mp.mpf, basis_vals: list, basis_names: list,
                   dps: int, tol: float) -> Optional[dict]:
    """Try PSLQ: find integer relation c[0]*val + c[1]*b[1] + ... + c[n]*b[n] = 0."""
    mp.mp.dps = dps + 10
    vec = [val] + list(basis_vals)
    try:
        rel = mp.pslq(vec, tol=tol, maxcoeff=200)
    except Exception:
        return None
    if rel is None:
        return None
    c0 = int(rel[0])
    if c0 == 0:
        return None
    # Normalise sign so denominator is always positive
    sign = 1 if c0 > 0 else -1
    c0 = sign * c0
    rel = [sign * int(r) for r in rel]
    # val = - sum(rel[k]*b[k-1] for k=1..) / rel[0]
    numer_parts = []
    for k, (c, name) in enumerate(zip(rel[1:], basis_names)):
        if int(c) != 0:
            numer_parts.append(f"{-int(c)}·{name}" if name != "1" else f"{-int(c)}")
    if not numer_parts:
        return None
    formula = " + ".join(numer_parts) + f" / {c0}"
    # Verify
    reconstructed = sum(-mp.mpf(int(rel[k+1])) * basis_vals[k]
                        for k in range(len(basis_vals))) / mp.mpf(c0)
    err = abs(float(val - reconstructed))
    if err > 1e-10:
        return None
    return {"formula": formula, "rel": [int(r) for r in rel], "residual": err}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Process one CMF record
# ══════════════════════════════════════════════════════════════════════════════

def process_record(rec: dict, basis_vals: list, basis_names: list) -> list:
    """Returns list of identified limit dicts (may be empty)."""
    agent = rec.get("agent", "?")
    dim   = rec.get("dim", 3)
    fp    = rec.get("fingerprint", "?")[:16]

    if dim > 5:
        return []   # PSLQ walk is unfeasibly slow for large-dim records

    try:
        fns, n_ax = build_fns(rec)
    except Exception as e:
        return []

    starts = [[3, 3, 3] + [3] * (n_ax - 3),
              [4, 2, 5] + [3] * (n_ax - 3),
              [2, 5, 3] + [3] * (n_ax - 3)]
    starts = [s[:n_ax] for s in starts]

    if n_ax == 3:
        rays = RAYS_3
        ray_labels = ["ex","ey","ez","exy","exz","eyz"]
    else:
        rays = [tuple(1 if k == ax else 0 for k in range(n_ax)) for ax in range(n_ax)]
        ray_labels = [f"e{ax}" for ax in range(n_ax)]

    identified = []
    for ri, (ray, rlabel) in enumerate(zip(rays, ray_labels)):
        for si, start in enumerate(starts[:2]):
            try:
                val = _walk_mp(fns, dim, n_ax, ray, start, WALK_DEPTH, DPS)
            except Exception:
                continue
            if val is None:
                continue
            match = _pslq_identify(val, basis_vals, basis_names, DPS, PSLQ_TOL)
            if match:
                identified.append({
                    "fingerprint": fp,
                    "agent": agent,
                    "dim": dim,
                    "coupling_bucket": rec.get("coupling_bucket", "?"),
                    "ray": rlabel,
                    "start": start,
                    "limit_approx": float(mp.re(val)),
                    "formula": match["formula"],
                    "residual": match["residual"],
                    "pslq_rel": match["rel"],
                    "source": rec.get("source_store", ""),
                })
    return identified


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def _load_top_n_records(n: int) -> list:
    """Load top-N CMFs by total_score from global_summary.csv."""
    csv_path = HERE.parent / "cmf_analysis_results" / "global_summary.csv"
    if not csv_path.exists():
        return []
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("total_score", 0) or 0), reverse=True)

    # Extract fingerprint from cmf_id (format: gauge_A_<fingerprint>)
    fps = set()
    for row in rows[:n]:
        cid = row.get("cmf_id", "")
        parts = cid.split("_", 2)
        if len(parts) == 3:
            fps.add(parts[2][:16])
    result = []
    seen: set = set()
    for store in sorted(HERE.glob("store_*.jsonl")):
        for line in store.read_text().splitlines():
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                fp16 = rec.get("fingerprint", "")[:16]
                if fp16 in fps and fp16 not in seen:
                    seen.add(fp16)
                    result.append(rec)
            except Exception:
                pass
    return result


def run_dreams(max_per_store: int = 200, only_b4: bool = False,
               focused_basis: bool = False, top_n: int = 0):
    mp.mp.dps = DPS + 20
    basis_vals, basis_names = (_make_focused_basis(DPS) if focused_basis
                               else _make_basis(DPS))

    # Load records
    if top_n > 0:
        sources = _load_top_n_records(top_n)
    else:
        sources = []
        for b in ([4] if only_b4 else [3, 4]):
            p = HERE / f"classified_bucket{b}.jsonl"
            if p.exists():
                lines = [l for l in p.read_text().splitlines() if l.strip()]
                for line in lines[:max_per_store]:
                    try:
                        sources.append(json.loads(line))
                    except Exception:
                        pass

    print(f"\n{'═'*68}")
    print(f"  Dreams Runner — PSLQ identification on {len(sources)} CMFs")
    print(f"  Walk depth={WALK_DEPTH}  dps={DPS}  tol={PSLQ_TOL}")
    print(f"{'═'*68}\n")

    # Build set of fingerprints that already have a confirmed limit (skip them)
    already_identified: set = set()
    results_dir = HERE.parent / "cmf_analysis_results"
    if top_n > 0 and results_dir.exists():
        for p in results_dir.glob("cmf_*/summary.json"):
            try:
                s = json.loads(p.read_text())
                if s.get("identified_limit"):
                    already_identified.add(s.get("fingerprint", "")[:16])
            except Exception:
                pass
        if already_identified:
            print(f"  Skipping {len(already_identified)} already-identified CMFs")

    all_hits = []
    hits_fh  = open(OUT_DIR / "dreams_hits.jsonl", "a")

    try:
        for idx, rec in enumerate(sources):
            fp  = rec.get("fingerprint", "?")[:12]
            fp16 = rec.get("fingerprint", "")[:16]
            dim = rec.get("dim", 3)
            b   = rec.get("coupling_bucket", "?")
            if fp16 in already_identified:
                print(f"  [{idx+1:>5}/{len(sources)}] fp={fp}  dim={dim}  B{b} → already identified, skip")
                continue
            print(f"  [{idx+1:>5}/{len(sources)}] fp={fp}  dim={dim}  B{b} ...",
                  end=" ", flush=True)
            t0 = time.time()
            hits = process_record(rec, basis_vals, basis_names)
            dt = time.time() - t0
            if hits:
                print(f"✓ {len(hits)} hit(s)  ({dt:.1f}s)")
                for h in hits:
                    print(f"           → {h['formula']}  [ray={h['ray']}  "
                          f"err={h['residual']:.2e}]")
                    hits_fh.write(json.dumps(h) + "\n")
                    hits_fh.flush()
                all_hits.extend(hits)
            else:
                print(f"no match  ({dt:.1f}s)")
    finally:
        hits_fh.close()

    # Write report
    _write_report(all_hits)
    return all_hits


def _write_report(hits: list):
    lines = ["# Ramanujan Dreams — Identified Limits\n",
             f"Total identifications: **{len(hits)}**\n"]

    if not hits:
        lines.append("_No limits identified in current run._\n")
    else:
        lines += ["## Hits\n",
                  "| fp | agent | dim | B | ray | limit | formula | err |",
                  "|----|----|---|---|---|---|---|---|"]
        for h in sorted(hits, key=lambda x: x.get("residual", 1)):
            lines.append(
                f"| `{h['fingerprint'][:10]}` | {h['agent']} | {h['dim']} "
                f"| B{h['coupling_bucket']} | {h['ray']} "
                f"| {h['limit_approx']:.8f} | `{h['formula']}` "
                f"| {h['residual']:.2e} |"
            )

    (OUT_DIR / "dreams_report.md").write_text("\n".join(lines))
    print(f"\n  Dreams report: {OUT_DIR / 'dreams_report.md'}")
    print(f"  Dreams hits:   {OUT_DIR / 'dreams_hits.jsonl'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200,
                    help="Max records per bucket to process")
    ap.add_argument("--b4-only", action="store_true",
                    help="Only process Bucket-4 CMFs")
    ap.add_argument("--depth", type=int, default=WALK_DEPTH,
                    help="Walk depth for mpmath")
    ap.add_argument("--dps", type=int, default=DPS,
                    help="mpmath decimal places")
    ap.add_argument("--focused-basis", action="store_true",
                    help="Use small 8-constant basis to reduce PSLQ overfitting")
    ap.add_argument("--top-n", type=int, default=0,
                    help="Select top-N CMFs by total_score from leaderboard (ignores --b4-only)")
    args = ap.parse_args()
    WALK_DEPTH = args.depth
    DPS = args.dps
    run_dreams(max_per_store=args.max, only_b4=args.b4_only,
               focused_basis=args.focused_basis, top_n=args.top_n)
