#!/usr/bin/env python3
"""
backfill_step_matrices.py
=========================
Finds all CMFs in atlas_2d.db whose canonical_payload has no step matrices
(empty `matrices` list), computes X0…Xk from LDU params, and updates:

  1. representation.canonical_payload  — adds "matrices" list
  2. cmf.cmf_payload                   — adds "params" + "limit_value"

Only processes dim <= 5 (symbolic inversion feasible).
For dim > 5 we add params to cmf_payload only (no symbolic matrices).

Usage:
  python3 backfill_step_matrices.py
  python3 backfill_step_matrices.py --limit 500       # first N records only
  python3 backfill_step_matrices.py --dim 3           # single dim
  python3 backfill_step_matrices.py --dry-run         # no DB writes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import sympy as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "gauge_agents"))

_VARNAMES = ["x", "y", "z", "w", "v"]

# ── Symbolic matrix builder (faster: sp.cancel instead of sp.simplify) ────────

def _as_rational(v: float) -> sp.Rational:
    f = Fraction(v).limit_denominator(1000)
    return sp.Rational(f.numerator, f.denominator)


def build_step_matrices(params: dict) -> dict[str, list]:
    """
    Compute X_i(coords) = G(coords+e_i) · D_i · G(coords)^{-1}.
    Returns {"X0": [[str,...], ...], "X1": ..., ...}.
    Returns {} if dim > 5.
    """
    dim    = params["dim"]
    n_vars = params["n_vars"]
    D_p    = params["D_params"]
    L_off  = params["L_off"]
    U_off  = params["U_off"]

    if dim > 5:
        return {}

    coords = sp.symbols(_VARNAMES[:n_vars])
    if n_vars == 1:
        coords = (coords,)

    def make_G(cvars):
        L = sp.eye(dim)
        for (i, j), v in L_off.items():
            L[i, j] = _as_rational(v)
        diag_v = [_as_rational(D_p[k][0]) * (cvars[k % n_vars] + _as_rational(D_p[k][1]))
                  for k in range(dim)]
        U = sp.eye(dim)
        for (i, j), v in U_off.items():
            U[i, j] = _as_rational(v)
        return L * sp.diag(*diag_v) * U

    G     = make_G(coords)
    G_inv = G.inv()          # exact rational — fast for dim <= 5

    result = {}
    for axis in range(n_vars):
        shifted = list(coords)
        shifted[axis] = shifted[axis] + 1
        G_sh = make_G(shifted)
        D_i  = sp.diag(*[coords[axis] + k for k in range(dim)])
        X    = G_sh * D_i * G_inv
        X    = sp.cancel(X)  # sp.cancel is ~3× faster than sp.simplify
        rows = [[str(X[r, c]) for c in range(dim)] for r in range(dim)]
        result[f"X{axis}"] = rows

    return result


# ── Reconstruct params from serialised form (string tuple keys) ───────────────

import ast

def reconstruct_params(raw: dict) -> dict:
    def parse_key(k: str) -> tuple:
        return ast.literal_eval(k)
    dim     = int(raw["dim"])
    d_p     = [tuple(dp) for dp in raw["D_params"]]
    # n_vars was added in a later scout version; old records lack it.
    # For old records: n_vars == number of D_params entries (one per axis).
    n_vars  = int(raw.get("n_vars", len(d_p)))
    return {
        "dim":      dim,
        "n_vars":   n_vars,
        "D_params": d_p,
        "L_off":    {parse_key(k): v for k, v in raw.get("L_off", {}).items()},
        "U_off":    {parse_key(k): v for k, v in raw.get("U_off", {}).items()},
    }


# ── Load all store records into a fingerprint → record map ────────────────────

def load_store_index(gauge_dir: Path, target_dims: Optional[list] = None) -> dict[str, dict]:
    """
    Returns fp → record for every store_*.jsonl record that has params.
    """
    import glob
    index = {}
    for sf in sorted(glob.glob(str(gauge_dir / "store_*.jsonl"))):
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not rec.get("params"):
                    continue
                fp  = rec.get("fingerprint", "")
                dim = rec.get("dim", 0)
                if target_dims and dim not in target_dims:
                    continue
                if fp and fp not in index:
                    index[fp] = rec
    print(f"  Store index: {len(index)} unique fingerprints loaded")
    return index


# ── Build updated canonical_payload ──────────────────────────────────────────

_AXIS_LABELS = ["x", "y", "z", "w", "v"]
_K_LABELS    = ["Kx", "Ky", "Kz", "Kw", "Kv"]


def enrich_canon_payload(existing: dict, x_matrices: dict, params: dict) -> dict:
    """
    Add step matrices (Kx/Ky/Kz) to canonical_payload alongside existing LDU entries.
    LDU matrices are preserved; step matrices appended.
    If no x_matrices computed (dim>5), still adds params.
    """
    updated = dict(existing)
    # Keep existing LDU matrices, append step matrices
    existing_mats = list(existing.get("matrices", []))
    step_mats = []
    for i in range(params["n_vars"]):
        key = f"X{i}"
        rows = x_matrices.get(key)
        if rows is None:
            break
        step_mats.append({
            "label":  _K_LABELS[i],
            "axis":   _AXIS_LABELS[i],
            "index":  i,
            "source": "explicit",
            "rows":   rows,
        })
    updated["matrices"] = existing_mats + step_mats
    updated["params"]   = {
        "dim":      params["dim"],
        "n_vars":   params["n_vars"],
        "D_params": params["D_params"],
        "L_off":    {str(k): v for k, v in params["L_off"].items()},
        "U_off":    {str(k): v for k, v in params["U_off"].items()},
    }
    return updated


def enrich_cmf_payload(existing: dict, params: dict, rec: dict) -> dict:
    updated = dict(existing)
    updated["params"] = {
        "dim":      params["dim"],
        "n_vars":   params["n_vars"],
        "D_params": params["D_params"],
        "L_off":    {str(k): v for k, v in params["L_off"].items()},
        "U_off":    {str(k): v for k, v in params["U_off"].items()},
    }
    if rec.get("limit_value"):
        updated["limit_value"]  = rec["limit_value"]
    if rec.get("limit_label"):
        updated["limit_label"]  = rec["limit_label"]
    if rec.get("looks_irrational") is not None:
        updated["looks_irrational"] = rec["looks_irrational"]
    if rec.get("gate5_ray_limits"):
        updated["gate5_ray_limits"] = rec["gate5_ray_limits"]
    return updated


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",      type=Path, default=HERE / "data" / "atlas_2d.db")
    ap.add_argument("--limit",   type=int,  default=None, help="Max records to process")
    ap.add_argument("--dim",     type=int,  default=None, help="Process only this dim")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--params-only", action="store_true",
                    help="Only backfill params into cmf_payload (skip symbolic matrices)")
    args = ap.parse_args()

    import sqlite3
    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row

    target_dims = [args.dim] if args.dim else None

    gauge_dir = HERE / "gauge_agents"
    print(f"\nBackfill Step Matrices → {args.db}")
    print(f"  dry-run: {args.dry_run}  limit: {args.limit}  dim filter: {target_dims}")
    print(f"  params-only: {args.params_only}\n")

    # Load store index
    print("Loading store index …")
    store_index = load_store_index(gauge_dir, target_dims)

    # Find all representations whose canonical_payload has empty matrices
    print("Scanning DB for representations missing step matrices …")
    query = """
        SELECT r.id AS rep_id,
               r.canonical_fingerprint AS fp,
               r.canonical_payload,
               c.id AS cmf_id,
               c.cmf_payload
        FROM representation r
        JOIN cmf c ON c.representation_id = r.id
        ORDER BY r.id
    """
    rows = con.execute(query).fetchall()
    print(f"  Total representation rows: {len(rows)}")

    # Filter: records missing step matrices (source="explicit" / label Kx/Ky/Kz)
    # All records have LDU matrices already — we need to ADD step matrices
    def _has_step_matrices(canon: dict) -> bool:
        for m in canon.get("matrices", []):
            if m.get("source") == "explicit" or m.get("label", "").startswith("K"):
                return True
        return False

    to_update = []
    for row in rows:
        fp = row["fp"]
        if fp not in store_index:
            continue
        try:
            canon = json.loads(row["canonical_payload"])
        except Exception:
            continue
        if _has_step_matrices(canon):
            continue
        to_update.append(row)
        if args.limit and len(to_update) >= args.limit:
            break

    print(f"  Need updating: {len(to_update)}")

    # Sort by dim (smallest first for quick wins)
    def get_dim(row):
        return store_index.get(row["fp"], {}).get("dim", 99)
    to_update.sort(key=get_dim)

    stats = {"done": 0, "symbolic": 0, "params_only": 0, "skipped": 0, "errors": 0}
    t0 = time.time()
    batch = []
    BATCH_SIZE = 50

    for i, row in enumerate(to_update):
        fp  = row["fp"]
        rec = store_index[fp]
        dim = rec.get("dim", 0)

        try:
            raw_params = rec["params"]
            params     = reconstruct_params(raw_params)
        except Exception as e:
            print(f"  [{i+1}] fp={fp} — param reconstruct error: {e}")
            stats["errors"] += 1
            continue

        # Compute symbolic matrices
        x_matrices = {}
        if not args.params_only and dim <= 5:
            try:
                t_sym = time.time()
                x_matrices = build_step_matrices(params)
                elapsed_sym = round(time.time() - t_sym, 2)
                stats["symbolic"] += 1
                if (i + 1) % 25 == 0 or elapsed_sym > 2:
                    print(f"  [{i+1}/{len(to_update)}] dim={dim}  fp={fp[:12]}  "
                          f"symbolic {elapsed_sym}s  "
                          f"({stats['symbolic']} done, {stats['errors']} errors, "
                          f"{round(time.time()-t0,0)}s elapsed)",
                          flush=True)
            except Exception as e:
                print(f"  [{i+1}] fp={fp} dim={dim} symbolic error: {e}")
                stats["errors"] += 1
                x_matrices = {}
        else:
            stats["params_only"] += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(to_update)}] dim={dim} params-only  "
                      f"({round(time.time()-t0,0)}s)", flush=True)

        # Build updated payloads
        try:
            canon  = json.loads(row["canonical_payload"])
            cmf_pl = json.loads(row["cmf_payload"])
        except Exception:
            stats["errors"] += 1
            continue

        new_canon  = enrich_canon_payload(canon, x_matrices, params)
        new_cmf_pl = enrich_cmf_payload(cmf_pl, params, rec)

        batch.append((
            json.dumps(new_canon),  row["rep_id"],
            json.dumps(new_cmf_pl), row["cmf_id"],
        ))
        stats["done"] += 1

        # Commit batch
        if not args.dry_run and len(batch) >= BATCH_SIZE:
            cur = con.cursor()
            for canon_s, rep_id, cmf_s, cmf_id in batch:
                cur.execute("UPDATE representation SET canonical_payload=? WHERE id=?",
                            (canon_s, rep_id))
                cur.execute("UPDATE cmf SET cmf_payload=? WHERE id=?",
                            (cmf_s, cmf_id))
            con.commit()
            batch.clear()
            print(f"    ✓ Committed batch  total_done={stats['done']}", flush=True)

    # Final commit
    if not args.dry_run and batch:
        cur = con.cursor()
        for canon_s, rep_id, cmf_s, cmf_id in batch:
            cur.execute("UPDATE representation SET canonical_payload=? WHERE id=?",
                        (canon_s, rep_id))
            cur.execute("UPDATE cmf SET cmf_payload=? WHERE id=?",
                        (cmf_s, cmf_id))
        con.commit()
        batch.clear()

    con.close()
    elapsed = round(time.time() - t0, 1)

    print(f"\n{'═'*55}")
    print(f"  Backfill complete — {elapsed}s")
    print(f"  Records processed : {stats['done']}")
    print(f"  Symbolic matrices : {stats['symbolic']}")
    print(f"  Params-only       : {stats['params_only']}")
    print(f"  Errors            : {stats['errors']}")
    print(f"  Dry-run           : {args.dry_run}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
