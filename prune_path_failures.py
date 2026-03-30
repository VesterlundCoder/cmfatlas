"""
prune_path_failures.py
======================
Run the Kx/Ky path-independence test on every CMF in the database that has
f_poly + fbar_poly and remove those whose max Frobenius residual exceeds
THRESHOLD (default 1e-10).

Usage:
    python prune_path_failures.py [--db PATH] [--threshold 1e-10] [--dry-run]

Options:
    --db PATH         Path to atlas.db  (default: data/atlas_2d.db or data/atlas.db)
    --threshold F     Max residual before removal  (default: 1e-10)
    --dry-run         Report failures but do NOT delete from database
    --k-max N         Grid size k = 1..N  (default: 5)
    --m-max N         Grid size m = 1..N  (default: 5)
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import mpmath
import sympy as sp
from sympy import symbols, lambdify, expand, sympify

# ---------------------------------------------------------------------------
# Helpers (mirrors api.py logic)
# ---------------------------------------------------------------------------

def _safe_json(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _poly_has_z(f_poly_str):
    return 'z' in str(sympify(f_poly_str).free_symbols)


def _build_walk_fns(f_poly_str, fbar_poly_str):
    k_s, m_s, x_s, y_s = symbols("k m x y")
    f_expr    = sympify(f_poly_str)
    fbar_expr = sympify(fbar_poly_str)
    is_3d     = _poly_has_z(f_poly_str)

    if is_3d:
        _free  = f_expr.free_symbols
        x_sym  = next((s for s in _free if s.name == 'x'), x_s)
        y_sym  = next((s for s in _free if s.name == 'y'), y_s)
        z_sym  = next((s for s in _free if s.name == 'z'), symbols("z"))
        g_kmz    = f_expr.subs([(x_sym, k_s), (y_sym, m_s)])
        gbar_kmz = fbar_expr.subs([(x_sym, k_s), (y_sym, m_s)])
        b_kz_expr = expand(g_kmz.subs(m_s, 0) * gbar_kmz.subs(m_s, 0))
        a_expr    = expand(g_kmz - gbar_kmz.subs(k_s, k_s + 1))

        def _ev(expr, k_v, m_v, z_v):
            return complex(expr.subs([(k_s, k_v), (m_s, m_v), (z_sym, z_v)]))
        def _evb(expr, k_v, z_v):
            return complex(expr.subs([(k_s, k_v), (z_sym, z_v)]))

        def Kx(k, m, n):
            return mpmath.matrix([[0, 1], [_evb(b_kz_expr, k+1, n), _ev(a_expr, k, m, n)]])
        def Ky(k, m, n):
            return mpmath.matrix([[_ev(gbar_kmz,k,m,n),1],[_evb(b_kz_expr,k,n),_ev(g_kmz,k,m,n)]])
        def Kz(k, m, n):
            return mpmath.matrix([[_ev(gbar_kmz,k,m,n),1],[_evb(b_kz_expr,k,n),_ev(g_kmz,k,m,n)]])
        return Kx, Ky, Kz, True

    g_km    = f_expr.subs([(x_s, k_s), (y_s, m_s)])
    gbar_km = fbar_expr.subs([(x_s, k_s), (y_s, m_s)])
    b_expr  = expand(g_km.subs(m_s, 0) * gbar_km.subs(m_s, 0))
    a_expr  = expand(g_km - gbar_km.subs(k_s, k_s + 1))

    g_fn    = lambdify([k_s, m_s], g_km,    modules="mpmath")
    gbar_fn = lambdify([k_s, m_s], gbar_km, modules="mpmath")
    b_fn    = lambdify([k_s],       b_expr,  modules="mpmath")
    a_fn    = lambdify([k_s, m_s], a_expr,  modules="mpmath")

    def Kx2(k, m, n=0):
        return mpmath.matrix([[0, 1], [b_fn(k+1), a_fn(k, m)]])
    def Ky2(k, m, n=0):
        return mpmath.matrix([[gbar_fn(k,m), 1], [b_fn(k), g_fn(k,m)]])
    return Kx2, Ky2, None, False


def _frob(M):
    return float(mpmath.sqrt(sum(
        mpmath.fabs(M[i, j])**2
        for i in range(M.rows) for j in range(M.cols)
    )))


def run_path_test(f_poly, fbar_poly, k_max=5, m_max=5, n_fixed=1, dps=20):
    """Returns (max_residual, n_tested, verdict)."""
    mpmath.mp.dps = dps
    try:
        Kx, Ky, Kz, is_3d = _build_walk_fns(f_poly, fbar_poly)
    except Exception as e:
        return None, 0, f"parse_error: {e}"

    max_res = 0.0
    n_tested = 0

    for k in range(1, k_max + 1):
        for m in range(1, m_max + 1):
            try:
                lhs = Kx(k, m, n_fixed) * Ky(k+1, m, n_fixed)
                rhs = Ky(k, m, n_fixed) * Kx(k, m+1, n_fixed)
                res = _frob(lhs - rhs)
                max_res = max(max_res, res)
                n_tested += 1
            except Exception:
                pass

    if n_tested == 0:
        return None, 0, "no_points"
    if max_res < 1e-10:
        verdict = "pass"
    elif max_res < 1e-3:
        verdict = "approx"
    else:
        verdict = "fail"
    return max_res, n_tested, verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prune path-independence failures from atlas DB")
    parser.add_argument("--db",        default=None,   help="Path to atlas.db")
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--k-max",     type=int, default=5)
    parser.add_argument("--m-max",     type=int, default=5)
    parser.add_argument("--skip-b-verified", action="store_true", default=True,
                        help="Keep B_verified_numeric failures (they have Sage flatness proof via different method)")
    args = parser.parse_args()

    # Locate DB
    if args.db:
        db_path = Path(args.db)
    else:
        base = Path(__file__).parent / "data"
        db_path = base / "atlas_2d.db"
        if not db_path.exists():
            db_path = base / "atlas.db"
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Database : {db_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Dry run  : {args.dry_run}")
    print(f"Grid     : k_max={args.k_max}, m_max={args.m_max}")
    print()

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT id, cmf_payload FROM cmf ORDER BY id"
    ).fetchall()

    print(f"Total CMFs: {len(rows)}")

    failures = []
    passes   = []
    skipped  = 0

    for i, row in enumerate(rows):
        cmf_id  = row["id"]
        payload = _safe_json(row["cmf_payload"]) or {}
        f_poly    = payload.get("f_poly", "")
        fbar_poly = payload.get("fbar_poly", "")

        if not f_poly or not fbar_poly:
            skipped += 1
            continue

        max_res, n_tested, verdict = run_path_test(
            f_poly, fbar_poly, args.k_max, args.m_max
        )

        cert_level = payload.get("certification_level", "?")
        log10 = f"{math.log10(max_res):.1f}" if max_res and max_res > 0 else "—"

        status = "PASS" if verdict == "pass" else ("APPROX" if verdict == "approx" else "FAIL")
        print(f"  [{i+1:4d}/{len(rows)}] CMF #{cmf_id:5d} [{cert_level:20s}] "
              f"max_res=10^{log10:>6s}  n={n_tested:3d}  {status}")

        skip = (args.skip_b_verified and cert_level == "B_verified_numeric")
        if max_res is not None and max_res > args.threshold and not skip:
            failures.append({"id": cmf_id, "max_res": max_res,
                             "log10": log10, "cert": cert_level,
                             "f_poly": f_poly[:60]})
        else:
            passes.append(cmf_id)

    print()
    print(f"Results: {len(passes)} PASS, {len(failures)} FAIL, {skipped} skipped (no poly)")
    print()

    if not failures:
        print("No failures — database unchanged.")
        con.close()
        return

    print(f"{'CMF ID':>8}  {'cert_level':20s}  {'log10_residual':>14}  f_poly")
    print("-" * 80)
    for f in sorted(failures, key=lambda x: x["max_res"], reverse=True):
        print(f"  #{f['id']:5d}  {f['cert']:20s}  10^{f['log10']:>6s}  {f['f_poly']}")
    print()

    if args.dry_run:
        print("DRY RUN — no deletions performed.")
    else:
        ids = [f["id"] for f in failures]
        placeholders = ",".join("?" * len(ids))
        con.execute(f"DELETE FROM cmf WHERE id IN ({placeholders})", ids)
        con.commit()
        print(f"Deleted {len(ids)} CMF records from the database.")

    con.close()


if __name__ == "__main__":
    main()
