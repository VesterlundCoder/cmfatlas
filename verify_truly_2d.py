#!/usr/bin/env python3
"""
Verify all truly 2D telescope CMFs: symbolic + numerical path independence.

Path independence (flatness) for a 2D CMF means:
    K1(k, m) · K2(k+1, m) = K2(k, m) · K1(k, m+1)

For telescope CMFs built from f(x,y) with conjugacy fbar(x,y):
    g(k,m)    = f(k,m)
    gbar(k,m) = fbar(k,m)
    b(k)      = g(k,0) · gbar(k,0)
    a(k,m)    = g(k,m) - gbar(k+1,m)

    K1(k,m) = [[a(k,m), 1], [b(k+1), 0]]
    K2(k,m) = [[gbar(k,m), 1], [b(k), g(k,m)]]

This script:
  1. Extracts all truly 2D CMFs from the atlas DB
  2. Reconstructs K1, K2 symbolically from f_poly + conjugacy
  3. SYMBOLIC: verifies K1·K2(k+1) - K2·K1(m+1) = 0 via SymPy expand
  4. NUMERICAL: evaluates at many (k,m) points with mpmath high precision
  5. Reports pass/fail with details
  6. Saves results to JSON
"""

import json
import math
import sys
import time
from collections import Counter, defaultdict

import mpmath
import numpy as np
import sympy as sp
from sympy import Poly, expand, symbols

sys.path.insert(0, "/Users/davidsvensson/Desktop/cmf_atlas/src")

from cmf_atlas.db.models import CMF, Representation
from cmf_atlas.db.session import get_engine, get_session
from cmf_atlas.util.json import loads

# ═══════════════════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════════════════
DB_PATH = "/Users/davidsvensson/Desktop/cmf_atlas/data/atlas.db"

x, y, k, m = symbols("x y k m")

CONJUGACIES = {
    "negx":     lambda f: f.subs(x, -x),
    "negy":     lambda f: f.subs(y, -y),
    "neg_negx": lambda f: (-f).subs(x, -x),
    "neg_negy": lambda f: (-f).subs(y, -y),
}


def build_K_matrices(f_expr, fbar_expr):
    """Build symbolic K1(k,m) and K2(k,m) from f and fbar.

    CORRECT convention (matching certified CMF database):
        K1(k,m) = [[0, 1], [b(k+1), a(k,m)]]
        K2(k,m) = [[gbar(k,m), 1], [b(k), g(k,m)]]
    where:
        g(k,m) = f(k,m),  gbar(k,m) = fbar(k,m)
        a(k,m) = g(k,m) - gbar(k+1,m)
        b(k)   = g(k,0) * gbar(k,0)
    """
    g_km = f_expr.subs([(x, k), (y, m)])
    gbar_km = fbar_expr.subs([(x, k), (y, m)])
    b_k = expand(g_km.subs(m, 0) * gbar_km.subs(m, 0))
    a_km = expand(g_km - gbar_km.subs(k, k + 1))

    K1 = sp.Matrix([
        [0, 1],
        [b_k.subs(k, k + 1), a_km]
    ])
    K2 = sp.Matrix([
        [gbar_km, 1],
        [b_k, g_km]
    ])
    return K1, K2, a_km, b_k, g_km, gbar_km


# ═══════════════════════════════════════════════════════════════════════
#  Symbolic Verification
# ═══════════════════════════════════════════════════════════════════════
def verify_symbolic(f_expr, fbar_expr):
    """
    Symbolically verify path independence:
        K1(k,m) · K2(k+1,m) == K2(k,m) · K1(k,m+1)
    Returns (ok, details_dict).
    """
    K1, K2, a_km, b_k, g_km, gbar_km = build_K_matrices(f_expr, fbar_expr)

    K2_k1 = K2.subs(k, k + 1)       # K2(k+1, m)
    K1_m1 = K1.subs(m, m + 1)       # K1(k, m+1)

    LHS = K1 * K2_k1
    RHS = K2 * K1_m1

    diff = sp.Matrix([[expand(LHS[i, j] - RHS[i, j]) for j in range(2)] for i in range(2)])

    ok = all(diff[i, j] == 0 for i in range(2) for j in range(2))

    residuals = {}
    if not ok:
        for i in range(2):
            for j in range(2):
                if diff[i, j] != 0:
                    residuals[f"({i},{j})"] = str(diff[i, j])[:200]

    return ok, {
        "method": "symbolic_expand",
        "passed": ok,
        "residuals": residuals,
        "a_km": str(a_km),
        "b_k": str(b_k),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Numerical Verification
# ═══════════════════════════════════════════════════════════════════════
def verify_numerical(f_expr, fbar_expr, n_k=15, n_m=8, dps=50):
    """
    Numerically verify path independence at many (k,m) points.
    Returns (ok, details_dict).
    """
    _, _, a_km_sym, b_k_sym, g_km_sym, gbar_km_sym = build_K_matrices(f_expr, fbar_expr)

    a_fn = sp.lambdify((k, m), a_km_sym, "mpmath")
    b_fn = sp.lambdify(k, b_k_sym, "mpmath")
    g_fn = sp.lambdify((k, m), g_km_sym, "mpmath")
    gbar_fn = sp.lambdify((k, m), gbar_km_sym, "mpmath")

    mpmath.mp.dps = dps
    max_rel_error = mpmath.mpf(0)
    max_abs_error = mpmath.mpf(0)
    n_tested = 0
    n_passed = 0
    worst_point = None

    for kv in range(2, 2 + n_k):
        for mv in range(0, n_m):
            try:
                # K1(k,m) = [[0, 1], [b(k+1), a(k,m)]]
                a_val = mpmath.mpf(a_fn(kv, mv))
                b_val_k1 = mpmath.mpf(b_fn(kv + 1))
                K1_mat = mpmath.matrix([[0, 1], [b_val_k1, a_val]])

                # K2(k+1, m) = [[gbar(k+1,m), 1], [b(k+1), g(k+1,m)]]
                gbar_k1 = mpmath.mpf(gbar_fn(kv + 1, mv))
                g_k1 = mpmath.mpf(g_fn(kv + 1, mv))
                b_val_k1_2 = mpmath.mpf(b_fn(kv + 1))
                K2_k1 = mpmath.matrix([[gbar_k1, 1], [b_val_k1_2, g_k1]])

                # K2(k, m) = [[gbar(k,m), 1], [b(k), g(k,m)]]
                gbar_val = mpmath.mpf(gbar_fn(kv, mv))
                g_val = mpmath.mpf(g_fn(kv, mv))
                b_val = mpmath.mpf(b_fn(kv))
                K2_mat = mpmath.matrix([[gbar_val, 1], [b_val, g_val]])

                # K1(k, m+1) = [[0, 1], [b(k+1), a(k,m+1)]]
                a_val_m1 = mpmath.mpf(a_fn(kv, mv + 1))
                K1_m1 = mpmath.matrix([[0, 1], [b_val_k1, a_val_m1]])

                # LHS = K1(k,m) · K2(k+1,m)
                LHS = K1_mat * K2_k1
                # RHS = K2(k,m) · K1(k,m+1)
                RHS = K2_mat * K1_m1

                n_tested += 1
                point_max_err = mpmath.mpf(0)
                for i in range(2):
                    for j in range(2):
                        diff = abs(LHS[i, j] - RHS[i, j])
                        scale = max(abs(LHS[i, j]), abs(RHS[i, j]), mpmath.mpf(1))
                        rel_err = diff / scale
                        point_max_err = max(point_max_err, rel_err)
                        max_abs_error = max(max_abs_error, diff)

                if point_max_err > max_rel_error:
                    max_rel_error = point_max_err
                    worst_point = (kv, mv)

                if point_max_err < mpmath.power(10, -dps + 10):
                    n_passed += 1

            except Exception as e:
                n_tested += 1
                continue

    ok = n_passed == n_tested and n_tested > 0
    return ok, {
        "method": f"numerical_dps{dps}",
        "passed": ok,
        "n_tested": n_tested,
        "n_passed": n_passed,
        "max_rel_error": float(max_rel_error),
        "max_abs_error": float(max_abs_error),
        "worst_point": worst_point,
        "dps": dps,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main verification pipeline
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    engine = get_engine(DB_PATH)
    session = get_session(engine)

    # Extract all truly 2D CMFs
    print("=" * 72)
    print("  TRULY 2D CMF VERIFICATION — Symbolic + Numerical")
    print("=" * 72)

    reps = session.query(Representation).filter_by(primary_group="dfinite").all()

    truly_2d_entries = []
    for rep in reps:
        cp = loads(rep.canonical_payload)
        if cp.get("source_type") != "telescope" or cp.get("dimension", 1) != 2:
            continue
        cmf = session.query(CMF).filter_by(representation_id=rep.id).first()
        cmf_p = loads(cmf.cmf_payload) if cmf else {}

        truly_2d_entries.append({
            "rep_id": rep.id,
            "f_poly": cp.get("f_poly", ""),
            "fbar_poly": cp.get("fbar_poly", ""),
            "conjugacy": cp.get("conjugacy", ""),
            "total_degree": cp.get("total_degree", 0),
            "deg_x": cp.get("deg_x", 0),
            "deg_y": cp.get("deg_y", 0),
            "a_km": cmf_p.get("a_km", ""),
            "limit_m0": cmf_p.get("limit_m0"),
            "limit_m1": cmf_p.get("limit_m1"),
        })

    print(f"\n  Found {len(truly_2d_entries)} truly 2D telescope CMFs in DB")

    # Stratify: take a representative sample if too many
    by_deg = defaultdict(list)
    for e in truly_2d_entries:
        by_deg[(e["total_degree"], e["conjugacy"])].append(e)

    print(f"  Stratified into {len(by_deg)} (degree, conjugacy) groups:")
    for key, entries in sorted(by_deg.items()):
        print(f"    deg={key[0]}, conj={key[1]}: {len(entries)} entries")

    # For symbolic: verify ALL unique (f_poly, conjugacy) pairs
    # For numerical: verify a sample from each group
    seen_polys = set()
    symbolic_queue = []
    numerical_queue = []

    for e in truly_2d_entries:
        poly_key = (e["f_poly"], e["conjugacy"])
        if poly_key not in seen_polys:
            seen_polys.add(poly_key)
            symbolic_queue.append(e)

    # Numerical: up to 5 per (degree, conjugacy) group
    for key, entries in by_deg.items():
        for e in entries[:5]:
            numerical_queue.append(e)

    print(f"\n  Symbolic verification: {len(symbolic_queue)} unique polynomials")
    print(f"  Numerical verification: {len(numerical_queue)} sample entries")

    # ── Symbolic verification ───────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  PHASE 1: SYMBOLIC VERIFICATION")
    print(f"{'─' * 72}")

    sym_results = []
    sym_pass = 0
    sym_fail = 0
    sym_error = 0

    for i, e in enumerate(symbolic_queue):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(symbolic_queue)}] verifying deg={e['total_degree']} conj={e['conjugacy']}...")

        try:
            f_expr = sp.sympify(e["f_poly"].replace("^", "**"))
            conj_fn = CONJUGACIES.get(e["conjugacy"])
            if not conj_fn:
                sym_error += 1
                sym_results.append({"rep_id": e["rep_id"], "symbolic": {"passed": False, "error": "unknown conjugacy"}})
                continue

            fbar_expr = conj_fn(f_expr)
            ok, details = verify_symbolic(f_expr, fbar_expr)

            if ok:
                sym_pass += 1
            else:
                sym_fail += 1
                print(f"    ✗ FAIL rep={e['rep_id']} f={e['f_poly'][:40]} conj={e['conjugacy']}")
                for cell, res in details.get("residuals", {}).items():
                    print(f"      diff{cell} = {res[:80]}")

            sym_results.append({
                "rep_id": e["rep_id"],
                "f_poly": e["f_poly"],
                "conjugacy": e["conjugacy"],
                "total_degree": e["total_degree"],
                "symbolic": details,
            })

        except Exception as ex:
            sym_error += 1
            sym_results.append({"rep_id": e["rep_id"], "symbolic": {"passed": False, "error": str(ex)[:200]}})

    print(f"\n  Symbolic results: {sym_pass} PASS, {sym_fail} FAIL, {sym_error} ERROR")
    print(f"  out of {len(symbolic_queue)} unique polynomials")

    # ── Numerical verification ──────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  PHASE 2: NUMERICAL VERIFICATION (dps=50)")
    print(f"{'─' * 72}")

    num_results = []
    num_pass = 0
    num_fail = 0
    num_error = 0

    for i, e in enumerate(numerical_queue):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(numerical_queue)}] testing deg={e['total_degree']} conj={e['conjugacy']}...")

        try:
            f_expr = sp.sympify(e["f_poly"].replace("^", "**"))
            conj_fn = CONJUGACIES.get(e["conjugacy"])
            if not conj_fn:
                num_error += 1
                continue

            fbar_expr = conj_fn(f_expr)
            ok, details = verify_numerical(f_expr, fbar_expr, n_k=12, n_m=6, dps=50)

            if ok:
                num_pass += 1
            else:
                num_fail += 1
                print(f"    ✗ FAIL rep={e['rep_id']} f={e['f_poly'][:40]} "
                      f"max_err={details['max_rel_error']:.2e} "
                      f"({details['n_passed']}/{details['n_tested']} pts)")

            num_results.append({
                "rep_id": e["rep_id"],
                "f_poly": e["f_poly"],
                "conjugacy": e["conjugacy"],
                "total_degree": e["total_degree"],
                "numerical": details,
            })

        except Exception as ex:
            num_error += 1
            num_results.append({"rep_id": e["rep_id"], "numerical": {"passed": False, "error": str(ex)[:200]}})

    print(f"\n  Numerical results: {num_pass} PASS, {num_fail} FAIL, {num_error} ERROR")
    print(f"  out of {len(numerical_queue)} samples")

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═' * 72}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'═' * 72}")
    print(f"  Total truly 2D CMFs in DB:    {len(truly_2d_entries)}")
    print(f"  Unique (f_poly, conjugacy):    {len(symbolic_queue)}")
    print(f"  Symbolic PASS:                 {sym_pass}/{len(symbolic_queue)}")
    print(f"  Symbolic FAIL:                 {sym_fail}")
    print(f"  Numerical PASS:                {num_pass}/{len(numerical_queue)}")
    print(f"  Numerical FAIL:                {num_fail}")
    print(f"  Elapsed:                       {elapsed:.1f}s")

    if sym_fail == 0 and num_fail == 0:
        print(f"\n  ✓ ALL VERIFICATIONS PASSED — path independence confirmed")
    else:
        print(f"\n  ✗ SOME FAILURES — review details above")

    # ── Save results ────────────────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_truly_2d": len(truly_2d_entries),
        "unique_polynomials": len(symbolic_queue),
        "symbolic": {
            "total": len(symbolic_queue),
            "pass": sym_pass,
            "fail": sym_fail,
            "error": sym_error,
            "details": sym_results,
        },
        "numerical": {
            "total": len(numerical_queue),
            "pass": num_pass,
            "fail": num_fail,
            "error": num_error,
            "dps": 50,
            "details": num_results,
        },
        "stratification": {
            f"deg{k}__{v}": len(es)
            for (k, v), es in sorted(by_deg.items())
        },
        "elapsed_seconds": elapsed,
    }

    out_path = "/Users/davidsvensson/Desktop/cmf_atlas/data/truly_2d_verification.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    session.close()


if __name__ == "__main__":
    main()
