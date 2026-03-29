"""Gap-filling pipeline: generate → evaluate → recognize → attempt CMF construction.

Fills all 73 detected gaps with generated candidates, evaluates them,
runs PSLQ recognition, and attempts 2D/3D CMF construction from the best ones.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mpmath
import numpy as np
import sympy as sp

from cmf_atlas.db.session import get_engine, get_session, init_db
from cmf_atlas.db.models import Representation, Features, CMF, EvalRun, Series, Project
from cmf_atlas.density.gaps import build_gap_atlas
from cmf_atlas.generate.pcf_gen import generate_pcf_candidates
from cmf_atlas.generate.hypergeom_gen import generate_hypergeom_candidates
from cmf_atlas.generate.dfinite_gen import generate_dfinite_candidates
from cmf_atlas.eval.quick import evaluate_pcf_quick
from cmf_atlas.recognition.pslq import run_pslq
from cmf_atlas.canonical import canonicalize_and_fingerprint
from cmf_atlas.features.compute import compute_features
from cmf_atlas.util.json import dumps
from cmf_atlas.util.logging import get_logger

log = get_logger("gap_fill")

DB_PATH = Path(__file__).parent / "data" / "atlas.db"
RESULTS_PATH = Path(__file__).parent / "data" / "gap_fill_results.json"

# ── Configuration ────────────────────────────────────────────────────────
N_CANDIDATES_PER_GAP = 50     # candidates generated per gap cell
QUICK_EVAL_DEPTH = 150        # CF walk depth for quick eval
QUICK_EVAL_DPS = 40           # decimal digits for quick eval
HP_EVAL_DEPTH = 1000          # high-precision depth for promising candidates
HP_EVAL_DPS = 200             # high-precision digits
CONV_THRESHOLD = 0.3          # min convergence score to proceed
PSLQ_DPS = 100                # PSLQ working precision


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1: Generate candidates in gaps
# ══════════════════════════════════════════════════════════════════════════
def generate_gap_candidates(gaps_df):
    """Generate candidates for every gap in the atlas."""
    all_candidates = []

    for _, gap in gaps_df.iterrows():
        group = gap["group"]
        x, y = int(gap["x_coord"]), int(gap["y_coord"])

        if group == "pcf":
            cands = generate_pcf_candidates(
                deg_a=x, deg_b=y,
                n_candidates=N_CANDIDATES_PER_GAP,
                coeff_range=8,
                seed=1000 * x + y,
            )
            for c in cands:
                c["group"] = "pcf"
                c["gap"] = (x, y)
        elif group == "dfinite":
            cands = generate_dfinite_candidates(
                rec_order=x, max_poly_degree=y,
                n_candidates=N_CANDIDATES_PER_GAP,
                coeff_range=8,
                seed=2000 * x + y,
            )
            for c in cands:
                c["group"] = "dfinite"
                c["gap"] = (x, y)
        elif group == "hypergeometric":
            cands = generate_hypergeom_candidates(
                degP=max(x, 1), degQ=max(y, 1),
                n_candidates=N_CANDIDATES_PER_GAP,
                coeff_range=8,
                seed=3000 * x + y,
            )
            for c in cands:
                c["group"] = "hypergeometric"
                c["gap"] = (x, y)
        else:
            cands = []

        all_candidates.extend(cands)

    return all_candidates


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2: Quick evaluate PCF candidates
# ══════════════════════════════════════════════════════════════════════════
def quick_evaluate_candidates(candidates):
    """Quick-evaluate all PCF candidates. Skip dfinite/hypergeom for now
    (they need series term generation which is more complex)."""
    results = []
    n_eval = 0
    n_converging = 0

    for c in candidates:
        if c["group"] != "pcf":
            # For dfinite: we can't directly evaluate as CF, skip quick eval
            # but keep them for CMF construction attempts
            c["eval"] = None
            results.append(c)
            continue

        a = c.get("a_coeffs", [0])
        b = c.get("b_coeffs", [0])

        # Skip trivially zero
        if all(x == 0 for x in a) or all(x == 0 for x in b):
            continue

        try:
            ev = evaluate_pcf_quick(a, b, depth=QUICK_EVAL_DEPTH, dps=QUICK_EVAL_DPS)
            c["eval"] = ev
            n_eval += 1

            if ev["convergence_score"] >= CONV_THRESHOLD:
                n_converging += 1

            results.append(c)
        except Exception:
            continue

    log.info(f"Quick eval: {n_eval} evaluated, {n_converging} converging (>{CONV_THRESHOLD})")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3: PSLQ recognition on converging candidates
# ══════════════════════════════════════════════════════════════════════════
def recognize_candidates(candidates):
    """Run PSLQ on candidates with good convergence."""
    n_attempted = 0
    n_recognized = 0

    for c in candidates:
        ev = c.get("eval")
        if ev is None or ev.get("convergence_score", 0) < CONV_THRESHOLD:
            c["recognition"] = None
            continue
        if ev.get("limit_estimate") is None:
            c["recognition"] = None
            continue

        try:
            # High-precision re-evaluation for PSLQ
            if c["group"] == "pcf":
                mpmath.mp.dps = PSLQ_DPS + 50
                a, b = c["a_coeffs"], c["b_coeffs"]

                def a_fn(n):
                    return sum(mpmath.mpf(coeff) * mpmath.power(n, i) for i, coeff in enumerate(a))
                def b_fn(n):
                    return sum(mpmath.mpf(coeff) * mpmath.power(n, i) for i, coeff in enumerate(b))

                p_prev, p_curr = mpmath.mpf(1), a_fn(0)
                q_prev, q_curr = mpmath.mpf(0), mpmath.mpf(1)
                for n in range(1, HP_EVAL_DEPTH + 1):
                    an, bn = a_fn(n), b_fn(n)
                    p_new = an * p_curr + bn * p_prev
                    q_new = an * q_curr + bn * q_prev
                    p_prev, p_curr = p_curr, p_new
                    q_prev, q_curr = q_curr, q_new

                if q_curr != 0:
                    limit_hp = mpmath.nstr(p_curr / q_curr, PSLQ_DPS)
                else:
                    c["recognition"] = None
                    continue

                rec = run_pslq(limit_hp, dps=PSLQ_DPS)
                c["recognition"] = rec
                n_attempted += 1
                if rec.get("success"):
                    n_recognized += 1
            else:
                c["recognition"] = None

        except Exception as e:
            c["recognition"] = {"success": 0, "error": str(e)}

    log.info(f"PSLQ: {n_attempted} attempted, {n_recognized} recognized")
    return candidates


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4: Attempt 2D / 3D CMF construction
# ══════════════════════════════════════════════════════════════════════════
def attempt_cmf_construction(candidates):
    """Try to build 2D and 3D CMFs from the best candidates.

    For PCFs with a(n), b(n): build the companion matrix K1(k) and search
    for K2(k,m) that satisfies the flatness equation K1(k)K2(k+1,m) = K2(k,m)K1(k,m+1).

    For dfinite operators: use the recurrence to build telescope-style matrices.
    """
    k, m = sp.symbols("k m", integer=True)
    cmf_results = []
    n_attempted = 0
    n_flat_2d = 0
    n_flat_3d = 0

    # Select top candidates: converging PCFs + all dfinite gap candidates
    top_pcf = sorted(
        [c for c in candidates if c["group"] == "pcf" and c.get("eval") and
         c["eval"].get("convergence_score", 0) >= CONV_THRESHOLD],
        key=lambda c: -c["eval"]["convergence_score"],
    )[:200]

    # Also include dfinite candidates (for telescope CMF attempts)
    dfinite_cands = [c for c in candidates if c["group"] == "dfinite"][:200]

    log.info(f"CMF construction: {len(top_pcf)} PCF + {len(dfinite_cands)} dfinite candidates")

    # ── 2D CMF from PCF (companion matrix approach) ──────────────────
    for c in top_pcf:
        n_attempted += 1
        a_coeffs = c["a_coeffs"]
        b_coeffs = c["b_coeffs"]

        try:
            # Build a(k,m) and b(k) symbolically
            a_sym = sum(sp.Integer(coeff) * k**i for i, coeff in enumerate(a_coeffs))
            b_sym = sum(sp.Integer(coeff) * k**i for i, coeff in enumerate(b_coeffs))

            # K1(k) = [[a(k), 1], [b(k+1), 0]]  (companion matrix)
            K1 = sp.Matrix([[a_sym, 1], [b_sym.subs(k, k + 1), 0]])

            # For 2D: need K2(k,m) such that K1(k)*K2(k+1,m) = K2(k,m)*K1_m
            # where K1_m means K1 with some m-dependence.
            #
            # Strategy: try conjugate polynomial approach.
            # Let f(x,y) be a polynomial. K1 entries become functions of (k,m).
            # If a(k) = p(k) and b(k) = q(k), try a(k,m) = p(k) + c*m for small c.
            # Then check flatness: K1(k)*K2(k+1,m) = K2(k,m)*K1(k,m+1)

            found_2d = False
            for c_m in range(-3, 4):
                if c_m == 0:
                    continue
                # Modify a(k) → a(k) + c_m * m
                a_2d = a_sym + c_m * m

                # K1(k,m) = [[a(k,m), 1], [b(k+1), 0]]
                K1_km = sp.Matrix([[a_2d, 1], [b_sym.subs(k, k + 1), 0]])

                # K2(k,m) using conjugate: try K2 = [[g(k,m), 1], [b(k), f(k,m)]]
                # where g = fbar, f = original poly
                # Flatness: K1(k,m) * K2(k+1,m) = K2(k,m) * K1(k,m+1)
                f_sym = a_2d  # f = a(k,m)
                g_sym = a_2d.subs(k, -k)  # fbar = a(-k,m) (negation conjugacy)
                b_k = b_sym

                K2_km = sp.Matrix([[g_sym, 1], [b_k, f_sym]])

                # Check flatness
                K1_k = K1_km
                K2_k1 = K2_km.subs(k, k + 1)
                K1_m1 = K1_km.subs(m, m + 1)
                K2_k_m = K2_km

                LHS = K1_k * K2_k1
                RHS = K2_k_m * K1_m1

                diff = sp.simplify(LHS - RHS)
                if diff == sp.zeros(2, 2):
                    found_2d = True
                    n_flat_2d += 1
                    cmf_results.append({
                        "type": "2D",
                        "source": c["provenance"],
                        "gap": c["gap"],
                        "a_coeffs": a_coeffs,
                        "b_coeffs": b_coeffs,
                        "c_m": c_m,
                        "a_2d": str(a_2d),
                        "limit": c["eval"].get("limit_estimate"),
                        "conv_score": c["eval"].get("convergence_score"),
                        "recognized": c.get("recognition", {}).get("identified_as") if c.get("recognition") else None,
                    })
                    break

            # Also try b(k) + d*m modification
            if not found_2d:
                for d_m in range(-3, 4):
                    if d_m == 0:
                        continue
                    b_2d = b_sym + d_m * m
                    K1_km = sp.Matrix([[a_sym, 1], [b_2d.subs(k, k + 1), 0]])
                    K2_km = sp.Matrix([[a_sym.subs(k, -k), 1], [b_2d, a_sym]])

                    LHS = K1_km * K2_km.subs(k, k + 1)
                    RHS = K2_km * K1_km.subs(m, m + 1)

                    diff = sp.simplify(LHS - RHS)
                    if diff == sp.zeros(2, 2):
                        n_flat_2d += 1
                        cmf_results.append({
                            "type": "2D",
                            "source": c["provenance"],
                            "gap": c["gap"],
                            "a_coeffs": a_coeffs,
                            "b_coeffs": b_coeffs,
                            "d_m": d_m,
                            "b_2d": str(b_2d),
                            "limit": c["eval"].get("limit_estimate"),
                            "conv_score": c["eval"].get("convergence_score"),
                            "recognized": c.get("recognition", {}).get("identified_as") if c.get("recognition") else None,
                        })
                        break

        except Exception:
            continue

    # ── 2D CMF from dfinite (telescope construction) ─────────────────
    for c in dfinite_cands:
        n_attempted += 1
        op = c.get("operator", [])
        if not op or len(op) < 2:
            continue

        try:
            # For a recurrence operator, build telescope-style CMF
            # p_0(n)*a_n + p_1(n)*a_{n+1} + ... + p_r(n)*a_{n+r} = 0
            # Use the companion matrix approach

            # Build symbolic coefficient polynomials
            polys = []
            for p_coeffs in op:
                p_sym = sum(sp.Integer(c_val) * k**i for i, c_val in enumerate(p_coeffs))
                polys.append(p_sym)

            if len(polys) < 2:
                continue

            # 2-term recurrence → companion matrix
            if len(polys) == 2:
                # a_{n+1} = -p_0(n)/p_1(n) * a_n
                ratio = -polys[0] / polys[1]
                ratio_simplified = sp.simplify(ratio)

                # K1(k) = [[ratio, 0], [1, 0]] or similar
                # Try to extend to 2D by adding m-dependence
                for c_m in range(-2, 3):
                    if c_m == 0:
                        continue
                    ratio_2d = ratio_simplified + c_m * m
                    K1 = sp.Matrix([[ratio_2d, 0], [1, 0]])
                    K2 = sp.Matrix([[ratio_2d.subs(k, -k), 0], [1, 0]])

                    LHS = K1 * K2.subs(k, k + 1)
                    RHS = K2 * K1.subs(m, m + 1)
                    diff = sp.simplify(LHS - RHS)
                    if diff == sp.zeros(2, 2):
                        n_flat_2d += 1
                        cmf_results.append({
                            "type": "2D_dfinite",
                            "source": c["provenance"],
                            "gap": c["gap"],
                            "operator": [[int(x) for x in p] for p in op],
                            "c_m": c_m,
                            "ratio_2d": str(ratio_2d),
                        })
                        break

            # 3-term recurrence → 2x2 companion
            elif len(polys) == 3:
                a_entry = -polys[1] / polys[2]
                b_entry = -polys[0] / polys[2]
                a_simp = sp.simplify(a_entry)
                b_simp = sp.simplify(b_entry)

                # K1(k) = [[a(k), 1], [b(k), 0]]
                # Try 2D extension
                for c_m in range(-2, 3):
                    if c_m == 0:
                        continue
                    a_2d = a_simp + c_m * m
                    K1_km = sp.Matrix([[a_2d, 1], [b_simp.subs(k, k + 1), 0]])
                    fbar = a_2d.subs(k, -k)
                    K2_km = sp.Matrix([[fbar, 1], [b_simp, a_2d]])

                    LHS = K1_km * K2_km.subs(k, k + 1)
                    RHS = K2_km * K1_km.subs(m, m + 1)
                    diff = sp.simplify(LHS - RHS)
                    if diff == sp.zeros(2, 2):
                        n_flat_2d += 1
                        cmf_results.append({
                            "type": "2D_dfinite",
                            "source": c["provenance"],
                            "gap": c["gap"],
                            "operator": [[int(x) for x in p] for p in op],
                            "c_m": c_m,
                            "a_2d": str(a_2d),
                        })
                        break

        except Exception:
            continue

    # ── 3D CMF attempts ──────────────────────────────────────────────
    # For any 2D CMF found, try to extend to 3D by finding K3(k,m,n)
    n_var = sp.Symbol("n", integer=True)
    for cmf_r in list(cmf_results):
        if cmf_r["type"] not in ("2D", "2D_dfinite"):
            continue

        try:
            # Parse the 2D CMF K1, K2 and try to find K3
            a_2d_str = cmf_r.get("a_2d") or cmf_r.get("ratio_2d", "")
            if not a_2d_str:
                continue

            a_2d_expr = sp.sympify(a_2d_str)

            # 3D attempt: a(k,m,n) = a(k,m) + d*n
            for d_n in range(-2, 3):
                if d_n == 0:
                    continue
                a_3d = a_2d_expr + d_n * n_var

                # Quick numeric check (exact symbolic is expensive for 3D)
                flat_3d = True
                for kv in range(2, 6):
                    for mv in range(1, 4):
                        for nv in range(1, 4):
                            try:
                                # K1(k,m,n), K2, K3 — check all 3 flatness conditions
                                a_val = float(a_3d.subs([(k, kv), (m, mv), (n_var, nv)]))
                                a_val_k1 = float(a_3d.subs([(k, kv + 1), (m, mv), (n_var, nv)]))
                                a_val_m1 = float(a_3d.subs([(k, kv), (m, mv + 1), (n_var, nv)]))
                                a_val_n1 = float(a_3d.subs([(k, kv), (m, mv), (n_var, nv + 1)]))

                                # For companion: K_i = [[a_i, 1], [b, 0]]
                                # Flatness K_i * K_j(shift_i) = K_j * K_i(shift_j)
                                # This requires a(k,m,n) to be linear in k,m,n
                                # with very specific coefficient relationships

                                # Check: is a(k+1,m,n)*a(k,m,n) symmetric in appropriate sense?
                                pass
                            except Exception:
                                flat_3d = False
                                break
                        if not flat_3d:
                            break
                    if not flat_3d:
                        break

                # For a proper 3D check, we need the full matrix equation.
                # Use symbolic for small cases.
                if "b_coeffs" in cmf_r:
                    b_coeffs = cmf_r["b_coeffs"]
                    b_sym = sum(sp.Integer(coeff) * k**i for i, coeff in enumerate(b_coeffs))
                else:
                    continue

                K1 = sp.Matrix([[a_3d, 1], [b_sym.subs(k, k + 1), 0]])
                K2 = sp.Matrix([[a_3d.subs(k, -k), 1], [b_sym, a_3d]])
                K3_candidate = sp.Matrix([[a_3d.subs(m, -m), 1], [b_sym, a_3d]])

                # Check K1*K3(k+1) = K3*K1(n+1)  and K2*K3(m+1) = K3*K2(n+1)
                check1 = sp.simplify(K1 * K3_candidate.subs(k, k + 1) - K3_candidate * K1.subs(n_var, n_var + 1))
                if check1 == sp.zeros(2, 2):
                    check2 = sp.simplify(K2 * K3_candidate.subs(m, m + 1) - K3_candidate * K2.subs(n_var, n_var + 1))
                    if check2 == sp.zeros(2, 2):
                        n_flat_3d += 1
                        cmf_results.append({
                            "type": "3D",
                            "parent_2d": cmf_r,
                            "d_n": d_n,
                            "a_3d": str(a_3d),
                        })
                        break

        except Exception:
            continue

    log.info(f"CMF construction: {n_attempted} attempted, {n_flat_2d} flat 2D, {n_flat_3d} flat 3D")
    return cmf_results


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5: Ingest results into atlas DB
# ══════════════════════════════════════════════════════════════════════════
def ingest_results(session, candidates, cmf_results):
    """Store evaluated candidates and CMF results in the database."""
    # Find or create project
    project = session.query(Project).filter_by(name="Gap Fill Run").first()
    if not project:
        project = Project(name="Gap Fill Run")
        session.add(project)
        session.flush()

    n_stored = 0
    n_dup = 0

    # Store converging PCF candidates
    for c in candidates:
        if c["group"] != "pcf":
            continue
        ev = c.get("eval")
        if ev is None or ev.get("convergence_score", 0) < CONV_THRESHOLD:
            continue

        try:
            payload = {"a_coeffs": c["a_coeffs"], "b_coeffs": c["b_coeffs"]}
            fp, canonical_json = canonicalize_and_fingerprint("pcf", payload)

            existing = session.query(Representation).filter_by(
                primary_group="pcf", canonical_fingerprint=fp
            ).first()
            if existing:
                n_dup += 1
                continue

            series = Series(
                project_id=project.id,
                name=f"gap_pcf_{c['gap'][0]}_{c['gap'][1]}",
                definition=c.get("provenance", ""),
                generator_type="gap_fill_pcf",
                provenance=f"gap_fill:pcf:{c['gap']}",
            )
            session.add(series)
            session.flush()

            rep = Representation(
                series_id=series.id,
                primary_group="pcf",
                canonical_fingerprint=fp,
                canonical_payload=canonical_json,
            )
            session.add(rep)
            session.flush()

            # Store features
            feat = compute_features("pcf", canonical_json)
            # Update with eval results
            feat["conv_score"] = ev.get("convergence_score")
            feat["stability_score"] = ev.get("stability_score")
            import math
            err = ev.get("error_estimate")
            if err and err > 0 and err < float("inf"):
                feat["log10_error"] = math.log10(err)

            rec = c.get("recognition")
            if rec:
                feat["recognized"] = rec.get("success", 0)
                feat["best_residual_log10"] = rec.get("residual_log10")

            from cmf_atlas.db.models import Features as FeaturesModel
            feat_obj = FeaturesModel(
                representation_id=rep.id,
                feature_json=dumps(feat),
                feature_version="1.0.0",
            )
            session.add(feat_obj)

            # CMF object
            cmf_payload = {
                "a_coeffs": c["a_coeffs"],
                "b_coeffs": c["b_coeffs"],
                "gap": c["gap"],
                "limit": ev.get("limit_estimate"),
                "convergence_score": ev.get("convergence_score"),
                "stability_score": ev.get("stability_score"),
                "recognized_as": rec.get("identified_as") if rec else None,
            }
            cmf_obj = CMF(
                representation_id=rep.id,
                cmf_payload=dumps(cmf_payload),
                dimension=1,
            )
            session.add(cmf_obj)
            session.flush()

            # Eval run
            eval_run = EvalRun(
                cmf_id=cmf_obj.id,
                run_type="quick",
                precision_digits=QUICK_EVAL_DPS,
                steps=QUICK_EVAL_DEPTH,
                limit_estimate=ev.get("limit_estimate"),
                error_estimate=ev.get("error_estimate"),
                convergence_score=ev.get("convergence_score"),
                stability_score=ev.get("stability_score"),
                runtime_ms=ev.get("runtime_ms"),
            )
            session.add(eval_run)

            n_stored += 1
        except Exception as e:
            session.rollback()
            project = session.query(Project).filter_by(name="Gap Fill Run").first()
            continue

    session.commit()
    log.info(f"Stored {n_stored} converging candidates ({n_dup} duplicates skipped)")
    return n_stored


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    engine = get_engine(DB_PATH)
    init_db(engine)
    session = get_session(engine)

    # Step 0: Load gaps
    log.info("Loading gap atlas...")
    gaps = build_gap_atlas(session, max_density=2)
    log.info(f"Found {len(gaps)} gaps")

    # Step 1: Generate
    log.info("Generating candidates...")
    candidates = generate_gap_candidates(gaps)
    log.info(f"Generated {len(candidates)} candidates across {len(gaps)} gaps")

    # Step 2: Quick evaluate
    log.info("Quick evaluating...")
    candidates = quick_evaluate_candidates(candidates)

    converging = [c for c in candidates if c.get("eval") and c["eval"].get("convergence_score", 0) >= CONV_THRESHOLD]
    log.info(f"{len(converging)} converging candidates")

    # Step 3: Recognize
    log.info("Running PSLQ recognition on top converging candidates...")
    # Only do PSLQ on top 100 by convergence score (expensive)
    converging_sorted = sorted(converging, key=lambda c: -c["eval"]["convergence_score"])
    top_for_pslq = converging_sorted[:100]
    recognize_candidates(top_for_pslq)

    # Merge recognition results back
    pslq_map = {}
    for c in top_for_pslq:
        key = (tuple(c.get("a_coeffs", [])), tuple(c.get("b_coeffs", [])))
        pslq_map[key] = c.get("recognition")
    for c in candidates:
        key = (tuple(c.get("a_coeffs", [])), tuple(c.get("b_coeffs", [])))
        if key in pslq_map:
            c["recognition"] = pslq_map[key]

    # Step 4: CMF construction
    log.info("Attempting 2D/3D CMF construction...")
    cmf_results = attempt_cmf_construction(candidates)

    # Step 5: Ingest results
    log.info("Ingesting results into atlas DB...")
    n_stored = ingest_results(session, candidates, cmf_results)

    # Step 6: Save full results to JSON
    results_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_gaps": len(gaps),
        "total_candidates_generated": len(candidates),
        "total_converging": len(converging),
        "pslq_attempted": len(top_for_pslq),
        "pslq_recognized": sum(1 for c in top_for_pslq if c.get("recognition", {}).get("success")),
        "cmf_2d_found": sum(1 for r in cmf_results if r["type"] in ("2D", "2D_dfinite")),
        "cmf_3d_found": sum(1 for r in cmf_results if r["type"] == "3D"),
        "stored_in_db": n_stored,
        "runtime_sec": round(time.time() - t0, 1),
        "cmf_results": cmf_results,
        "top_converging": [
            {
                "gap": c["gap"],
                "a_coeffs": c.get("a_coeffs"),
                "b_coeffs": c.get("b_coeffs"),
                "conv_score": c["eval"]["convergence_score"],
                "stability": c["eval"].get("stability_score"),
                "limit": c["eval"].get("limit_estimate"),
                "recognized": c.get("recognition", {}).get("identified_as") if c.get("recognition") else None,
            }
            for c in converging_sorted[:50]
        ],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  GAP-FILLING PIPELINE — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Gaps processed:          {len(gaps)}")
    print(f"  Candidates generated:    {len(candidates)}")
    print(f"  Converging (>{CONV_THRESHOLD}):     {len(converging)}")
    print(f"  PSLQ recognized:         {sum(1 for c in top_for_pslq if c.get('recognition', {}).get('success'))}")
    print(f"  2D CMFs found:           {sum(1 for r in cmf_results if r['type'] in ('2D', '2D_dfinite'))}")
    print(f"  3D CMFs found:           {sum(1 for r in cmf_results if r['type'] == '3D')}")
    print(f"  Stored in DB:            {n_stored}")
    print(f"  Runtime:                 {elapsed:.1f}s")
    print("=" * 70)

    # Print recognized constants
    recognized = [c for c in top_for_pslq if c.get("recognition", {}).get("success")]
    if recognized:
        print("\n  RECOGNIZED CONSTANTS:")
        for c in recognized[:20]:
            rec = c["recognition"]
            print(f"    gap={c['gap']} a={c['a_coeffs']} b={c['b_coeffs']}")
            print(f"      → {rec['identified_as']} (residual: 1e{rec.get('residual_log10', 0):.1f})")

    # Print CMF results
    if cmf_results:
        print("\n  CMF CONSTRUCTION RESULTS:")
        for r in cmf_results[:20]:
            print(f"    {r['type']}: gap={r.get('gap')} source={r.get('source', '')[:40]}")
            if r.get("a_2d"):
                print(f"      a(k,m) = {r['a_2d']}")
            if r.get("recognized"):
                print(f"      limit → {r['recognized']}")

    session.close()
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
