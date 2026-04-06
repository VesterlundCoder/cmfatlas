#!/usr/bin/env python3
"""
backfill_matrices.py
====================
Patches canonical_payload.matrices for gauge_agent CMFs that have empty matrices.

Strategy:
  Pass 1 — Scan ALL store_*.jsonl files. Build fp16 → record_with_X0 index
            (prefer records that have X0/X1/X2 already computed).
  Pass 2 — Query DB for representations with empty matrices.
            For each, if we have a store record with X0: use those.
  Pass 3 — For remaining (no X0 in store): rebuild symbolically from
            LDU params using SymPy.

Usage:
    python3 backfill_matrices.py            # all CMFs with empty matrices
    python3 backfill_matrices.py --dry-run  # count only, no DB writes
    python3 backfill_matrices.py --max 500  # cap at 500 updates
"""
from __future__ import annotations
import argparse, json, sqlite3, time
from pathlib import Path
from typing import Optional

import sympy as sp
from sympy import symbols, Matrix, Rational, cancel, expand

HERE  = Path(__file__).parent
GAUGE = HERE / "gauge_agents"
DB    = HERE / "data" / "atlas_2d.db"

_AXIS_LABELS = ["x","y","z","w","v"]
_K_LABELS    = ["Kx","Ky","Kz","Kw","Kv"]


# ── Symbolic matrix rebuilder (new LDU params format) ─────────────────────

def _build_X_sym(params: dict, dim: int, axis: int):
    """Return X_{axis}(n) as list-of-lists of str, or None on failure."""
    n_vars = symbols(" ".join(f"n{i}" for i in range(dim)))
    if isinstance(n_vars, sp.Symbol):
        n_vars = (n_vars,)

    L_off  = {eval(str(k)) if isinstance(k, str) else k: float(v)
              for k, v in params.get("L_off", {}).items()}
    U_off  = {eval(str(k)) if isinstance(k, str) else k: float(v)
              for k, v in params.get("U_off", {}).items()}
    D_pars = [(float(a), float(b)) for a, b in params.get("D_params", [])]

    if not D_pars:
        return None

    def _rat(v):
        return sp.Rational(v).limit_denominator(10000) if abs(v - round(v)) > 1e-9 else int(round(v))

    def G_sym(n_shift=None):
        ns = list(n_vars)
        if n_shift is not None:
            ns = [ns[k] + (1 if k == n_shift else 0) for k in range(dim)]
        L = sp.eye(dim)
        for (i, j), v in L_off.items():
            L[i, j] = _rat(v)
        D_diag = [_rat(a) * (ns[k % dim] + _rat(b)) for k, (a, b) in enumerate(D_pars)]
        D = sp.diag(*D_diag)
        U = sp.eye(dim)
        for (i, j), v in U_off.items():
            U[i, j] = _rat(v)
        return L * D * U

    try:
        Gn  = G_sym()
        Gsh = G_sym(n_shift=axis)
        Di  = sp.diag(*[n_vars[axis] + k for k in range(dim)])
        Xi  = Gsh * Di * Gn.inv()
        # Simplify each entry and convert to string
        rows = []
        for i in range(dim):
            row = []
            for j in range(dim):
                entry = cancel(expand(Xi[i, j]))
                row.append(str(entry))
            rows.append(row)
        return rows
    except Exception:
        return None


# ── Store-file index builder ──────────────────────────────────────────────

def build_store_index() -> dict[str, dict]:
    """Scan all store_*.jsonl, return fp16 → best_record (prefers records with X0)."""
    index: dict[str, dict] = {}
    for sf in sorted(GAUGE.glob("store_*.jsonl")):
        for line in sf.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fp = (rec.get("fingerprint") or "")[:16]
            if not fp:
                continue
            # Prefer records that already have X0
            existing = index.get(fp)
            if existing is None:
                index[fp] = rec
            elif "X0" not in existing and "X0" in rec:
                index[fp] = rec  # upgrade to one with symbolic matrices
    return index


# ── Matrix entry builder from store record ───────────────────────────────

def matrices_from_record(rec: dict, n_vars: int) -> Optional[list[dict]]:
    """Build canonical matrices list from a store record."""
    dim = int(rec.get("dim", 3))
    matrices = []
    for i, key in enumerate(_AXIS_LABELS[:n_vars]):
        mat_key = f"X{i}"
        rows = rec.get(mat_key)
        if rows is None:
            break
        matrices.append({
            "label":  _K_LABELS[i],
            "axis":   key,
            "index":  i,
            "source": "symbolic",
            "rows":   rows,
        })
    return matrices if matrices else None


def matrices_from_params(rec: dict) -> Optional[list[dict]]:
    """Rebuild symbolic matrices from LDU params (new format only)."""
    params = rec.get("params", {})
    if "D_params" not in params:
        return None  # old format — skip
    dim    = int(rec.get("dim", 3))
    n_vars = len(params.get("D_params", []))
    if n_vars < 2:
        n_vars = dim
    n_axes = min(n_vars, dim, len(_K_LABELS))
    matrices = []
    for i in range(n_axes):
        rows = _build_X_sym(params, dim, i)
        if rows is None:
            return None
        matrices.append({
            "label":  _K_LABELS[i],
            "axis":   _AXIS_LABELS[i],
            "index":  i,
            "source": "rebuilt_sympy",
            "rows":   rows,
        })
    return matrices if matrices else None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    print("Building store index …")
    t0 = time.time()
    idx = build_store_index()
    print(f"  {len(idx)} fingerprints indexed  ({time.time()-t0:.1f}s)")
    with_X0 = sum(1 for r in idx.values() if "X0" in r)
    print(f"  {with_X0} have X0/X1/X2 in store")

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    rows = con.execute("""
        SELECT r.id AS rep_id,
               r.canonical_fingerprint AS fp,
               r.canonical_payload
        FROM representation r
        WHERE r.primary_group = 'gauge_agent'
          AND (json_array_length(json_extract(r.canonical_payload,'$.matrices'))=0
               OR json_extract(r.canonical_payload,'$.matrices') IS NULL)
    """).fetchall()

    print(f"\nRepresentations with empty matrices: {len(rows)}")
    if args.max:
        rows = rows[:args.max]
        print(f"  (capped at {args.max})")

    updated_store = 0
    updated_sympy = 0
    skipped       = 0
    t1 = time.time()

    cur = con.cursor()
    for i, row in enumerate(rows):
        fp = (row["fp"] or "")[:16]
        rec = idx.get(fp)
        matrices = None

        if rec:
            dim    = int(rec.get("dim", 3))
            n_vars = int(rec.get("n_matrices") or
                         len(rec.get("params", {}).get("D_params", [])) or
                         dim)
            n_vars = min(n_vars, dim, len(_K_LABELS))
            # Pass 1: use pre-computed X0/X1/X2 from store
            matrices = matrices_from_record(rec, n_vars)
            if matrices:
                updated_store += 1
            else:
                # Pass 2: rebuild from LDU params
                matrices = matrices_from_params(rec)
                if matrices:
                    updated_sympy += 1

        if matrices is None:
            skipped += 1
            continue

        try:
            cp = json.loads(row["canonical_payload"])
        except Exception:
            skipped += 1
            continue

        cp["matrices"] = matrices

        if not args.dry_run:
            cur.execute("UPDATE representation SET canonical_payload=? WHERE id=?",
                        (json.dumps(cp), row["rep_id"]))

        if (i + 1) % 1000 == 0:
            if not args.dry_run:
                con.commit()
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            remaining = (len(rows) - i - 1) / rate / 60
            print(f"  {i+1}/{len(rows)}  store={updated_store}  sympy={updated_sympy}"
                  f"  skip={skipped}  {rate:.0f}/s  ~{remaining:.1f}min left")

    if not args.dry_run:
        con.commit()
    con.close()

    elapsed = time.time() - t1
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Updated from store:  {updated_store}")
    print(f"  Rebuilt via SymPy:   {updated_sympy}")
    print(f"  Skipped (no data):   {skipped}")
    if args.dry_run:
        print("  (dry-run, no changes written)")


if __name__ == "__main__":
    main()
