#!/usr/bin/env python3
"""
CMF_Distiller — Three-stage mathematical rigor pipeline
========================================================
Ingests data/irrational_candidates.json from CMF_Scout and produces
publication-ready, deduplicated mathematical structures.

STAGE 1 — verify_transcendence
    • mp.dps = 1000 walks at depth 10 000
    • 4-Gate strict filter (same as reward_engine.classify_limit_strict):
        G1: Zero/Divergence/±1 trap → FATAL
        G2: Algebraic purge (prime power roots) → FATAL
        G3: PSLQ coefficient cap > 50 → FATAL
        G4: True transcendental PSLQ residual < 1e-50 → MASSIVE BONUS

STAGE 2 — cluster_families
    • Eigenvalue + Tr-of-product structural fingerprint (gauge-invariant)
    • Groups CMFs into distinct mathematical families

STAGE 3 — export_paper_data
    • Per-family: structured JSON + LaTeX theorem draft snippet

Usage:
    python3 cmf_distiller.py
    python3 cmf_distiller.py --stage 1          # only Stage 1
    python3 cmf_distiller.py --stage 2          # only Stage 2 (uses Stage 1 output)
    python3 cmf_distiller.py --stage 3          # only Stage 3 (uses Stage 2 output)
    python3 cmf_distiller.py --max 10           # limit to first N candidates
    python3 cmf_distiller.py --resume           # skip already-verified fingerprints
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE            = Path(__file__).parent
GAUGE           = HERE / "gauge_agents"
CANDIDATES_JSON = HERE / "data" / "irrational_candidates.json"
STAGE1_OUT      = HERE / "data" / "distiller_stage1.jsonl"
STAGE2_OUT      = HERE / "data" / "distiller_stage2.json"
STAGE3_OUT_JSON = HERE / "data" / "distiller_paper_data.json"
STAGE3_OUT_TEX  = HERE / "data" / "distiller_theorems.tex"

sys.path.insert(0, str(GAUGE))
from run_all_agents import build_eval_fns, AGENT_CONFIGS
from reward_engine import classify_limit_strict

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DPS_HIGH        = 1000   # decimal places for deep verification
WALK_DEPTH      = 10_000 # lattice walk depth


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _load_store_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for sf in sorted(GAUGE.glob("store_*.jsonl")):
        for line in sf.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fp  = (rec.get("fingerprint") or "")[:16]
                if fp:
                    idx[fp] = rec
            except Exception:
                pass
    return idx


def _parse_off(raw: dict) -> dict:
    result = {}
    for k, v in raw.items():
        k2 = k.strip("()")
        parts = [x.strip() for x in k2.split(",")]
        result[tuple(int(p) for p in parts)] = float(v)
    return result


def _rebuild_fns(rec: dict):
    p = rec.get("params", {})
    if not p:
        return None, None
    dim    = int(p.get("dim", rec.get("dim", 3)))
    n_vars = int(p.get("n_vars", rec.get("n_vars",
                  AGENT_CONFIGS.get(rec.get("agent","A"), {}).get("n_vars", 3))))
    params = {
        "dim":      dim,
        "n_vars":   n_vars,
        "D_params": p["D_params"],
        "L_off":    _parse_off(p.get("L_off", {})),
        "U_off":    _parse_off(p.get("U_off", {})),
    }
    try:
        fns = build_eval_fns(params)
        return fns, params
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — verify_transcendence
# ══════════════════════════════════════════════════════════════════════════════

def _walk_mp(fn, dim: int, start: list[int], depth: int, dps: int):
    """High-precision walk. Returns mpf ratio or None."""
    mpmath.mp.dps = dps + 20
    pos = list(start)
    v   = mpmath.zeros(dim, 1)
    v[0] = mpmath.mpf(1)
    for _ in range(depth):
        pos[0] += 1
        try:
            raw = np.asarray(fn(*pos), dtype=float)
            M   = mpmath.matrix([[mpmath.mpf(str(raw[r][c]))
                                  for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            return None
        scale = max(abs(v[i]) for i in range(dim))
        if scale > mpmath.power(10, dps // 2):
            v /= scale
        elif scale < mpmath.power(10, -(dps // 2)):
            return None
    if abs(v[dim - 1]) < mpmath.power(10, -(dps - 10)):
        return None
    return v[0] / v[dim - 1]


def _is_rational(val: mpmath.mpf, dps: int) -> tuple[bool, str | None]:
    """Return (True, 'p/q') if value looks rational at this precision."""
    mpmath.mp.dps = dps + 20
    f = Fraction(float(val)).limit_denominator(RATIONAL_DENOM)
    approx = mpmath.mpf(f.numerator) / mpmath.mpf(f.denominator)
    diff = abs(val - approx)
    if diff < mpmath.power(10, -(dps // 3)):
        return True, f"{f.numerator}/{f.denominator}"
    return False, None


def verify_transcendence(candidates: list[dict], store_idx: dict,
                         resume_fps: set[str]) -> list[dict]:
    """Stage 1: Re-verify all candidates at 1000 dps, classify via 4-gate system."""
    print("\n" + "═" * 70)
    print("  STAGE 1 — Extreme-Precision Transcendence Verification")
    print(f"  mp.dps={DPS_HIGH}, depth={WALK_DEPTH:,}")
    print("═" * 70)

    results = []
    total   = len(candidates)

    for i, cand in enumerate(candidates):
        fp = cand.get("fingerprint", "")
        if fp in resume_fps:
            print(f"  [{i+1}/{total}] {fp} — skipped (done)")
            continue

        store_rec = store_idx.get(fp)
        if not store_rec:
            print(f"  [{i+1}/{total}] {fp} — NO STORE RECORD")
            results.append({**cand, "stage1": {"error": "no_store_record"}})
            continue

        fns, params = _rebuild_fns(store_rec)
        if fns is None:
            print(f"  [{i+1}/{total}] {fp} — could not rebuild fns")
            results.append({**cand, "stage1": {"error": "rebuild_failed"}})
            continue

        dim = params["dim"]
        fn0 = fns[0]
        start = [5] * params["n_vars"]

        print(f"\n  [{i+1}/{total}] fp={fp}  agent={cand.get('agent_type')}  "
              f"{dim}×{dim}  {cand.get('limit_label','')[:55]}", flush=True)
        t0 = time.time()

        # ── Deep walk at 1000 dps ─────────────────────────────────────────
        print(f"      Walking depth={WALK_DEPTH:,} at dps={DPS_HIGH} …", flush=True)
        hp_val = _walk_mp(fn0, dim, start, WALK_DEPTH, DPS_HIGH)
        elapsed_walk = time.time() - t0

        if hp_val is None:
            print(f"      ✗ Walk failed  ({elapsed_walk:.1f}s)")
            results.append({**cand, "stage1": {
                "walk_ok": False, "classification": "WALK_FAILED"}})
            continue

        hp_str = mpmath.nstr(hp_val, 50)
        print(f"      ✓ Walk done ({elapsed_walk:.1f}s)  L≈{hp_str[:40]}", flush=True)

        # ── 4-Gate strict classification ───────────────────────────────────────────────
        print(f"      Running 4-gate strict classifier …", flush=True)
        t1 = time.time()
        hp_float = float(mpmath.nstr(hp_val, 30))
        gate = classify_limit_strict(hp_float, hp_val=hp_val, verbose=True)
        elapsed_gate = time.time() - t1

        classif = gate["label"]
        print(f"      → {classif}  ({elapsed_gate:.1f}s)")

        # Map to stage1 classification names expected by Stage 2/3
        if gate["label"].startswith("FATAL"):
            s1_class = gate["label"]        # e.g. FATAL_ALGEBRAIC_ESCAPE
        elif gate["label"] == "TRUE_TRANSCENDENTAL":
            s1_class = "IDENTIFIED_KNOWN"
        else:
            s1_class = "UNKNOWN_IRRATIONAL"

        stage1 = {
            "walk_ok":        True,
            "hp_value":       hp_str,
            "classification": s1_class,
            "relation":       gate.get("reason"),
            "residual_log10": (
                round(math.log10(gate["pslq_residual"]), 1)
                if gate.get("pslq_residual") and gate["pslq_residual"] > 0
                else None
            ),
            "gate_label":     gate["label"],
            "gate_passed":    gate["gate_passed"],
            "walk_time_s":    round(elapsed_walk, 1),
            "gate_time_s":    round(elapsed_gate, 1),
        }
        results.append({**cand, "stage1": stage1})

    print(f"\n  Stage 1 complete: {len(results)} CMFs processed")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — cluster_families
# ══════════════════════════════════════════════════════════════════════════════

def _matrix_eigenvalue_fingerprint(matrices_rows: list[list[list]], dim: int) -> np.ndarray:
    """
    Gauge-invariant fingerprint = sorted eigenvalues of the product
    M_inf = prod(M_i) evaluated at a standard point.
    """
    # Build numpy matrices from stored symbolic rows (string entries → float eval)
    mats = []
    for rows in matrices_rows:
        try:
            M = np.zeros((dim, dim), dtype=complex)
            for r, row in enumerate(rows[:dim]):
                for c, entry in enumerate(row[:dim]):
                    if isinstance(entry, (int, float)):
                        M[r][c] = float(entry)
                    elif isinstance(entry, str):
                        # Evaluate string expression with x=5,y=5,z=5,w=5,v=5
                        try:
                            M[r][c] = float(eval(entry, {
                                "x": 5, "y": 5, "z": 5, "w": 5, "v": 5,
                                "n": 5, "__builtins__": {}
                            }))
                        except Exception:
                            M[r][c] = 0.0
            mats.append(M)
        except Exception:
            pass

    if not mats:
        return np.zeros(dim)

    # Product of all matrices
    P = mats[0].copy()
    for M in mats[1:]:
        P = P @ M

    # Eigenvalues, sorted by absolute value
    eigs = np.linalg.eigvals(P)
    eigs_sorted = sorted(eigs, key=lambda x: (abs(x), x.real, x.imag))

    fp = []
    for e in eigs_sorted:
        fp.extend([round(e.real, 4), round(e.imag, 4)])
    return np.array(fp)


def _trace_loop_fingerprint(matrices_rows: list[list[list]], dim: int) -> list[float]:
    """
    Gauge-invariant: traces of A·B, A·B·C, B·C·A (cyclic permutations).
    Trace is invariant under cyclic gauge transforms.
    """
    mats = []
    for rows in matrices_rows:
        try:
            M = np.zeros((dim, dim), dtype=float)
            for r, row in enumerate(rows[:dim]):
                for c, entry in enumerate(row[:dim]):
                    if isinstance(entry, (int, float)):
                        M[r][c] = float(entry)
                    elif isinstance(entry, str):
                        try:
                            M[r][c] = float(eval(entry, {
                                "x": 5, "y": 5, "z": 5, "w": 5, "v": 5,
                                "n": 5, "__builtins__": {}
                            }))
                        except Exception:
                            M[r][c] = 0.0
            mats.append(M)
        except Exception:
            pass

    if not mats:
        return []

    traces = []
    n = len(mats)
    for length in [2, 3]:
        if n >= length:
            for start in range(min(n, 3)):
                P = mats[start % n].copy()
                for k in range(1, length):
                    P = P @ mats[(start + k) % n]
                traces.append(round(np.trace(P).real, 3))

    return traces


def _fp_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """L2 distance between two fingerprint vectors (zero-padded)."""
    la, lb = len(fp_a), len(fp_b)
    if la != lb:
        maxl = max(la, lb)
        fp_a = np.pad(fp_a, (0, maxl - la))
        fp_b = np.pad(fp_b, (0, maxl - lb))
    return float(np.linalg.norm(fp_a - fp_b))


CLUSTER_THRESHOLD = 0.5  # eigenvalue L2 distance below this → same family


def cluster_families(stage1_results: list[dict]) -> dict:
    """Stage 2: Group CMFs into gauge-equivalence families."""
    print("\n" + "═" * 70)
    print("  STAGE 2 — Gauge-Equivalence Clustering")
    print(f"  Threshold: eigenvalue L2 < {CLUSTER_THRESHOLD}")
    print("═" * 70)

    # Only cluster CMFs that passed Stage 1 verification
    valid = [r for r in stage1_results
             if r.get("stage1", {}).get("walk_ok") and
                r.get("stage1", {}).get("classification") != "RATIONAL"]

    print(f"  Valid candidates: {len(valid)}")

    # Compute fingerprints
    fps_computed = []
    for cand in valid:
        mats_data = cand.get("matrices") or []
        dim = int(cand.get("matrix_size") or cand.get("dimension") or 3)

        # Extract rows from canonical matrix format
        matrices_rows = []
        for m in mats_data:
            if isinstance(m, dict) and "rows" in m:
                matrices_rows.append(m["rows"])
            elif isinstance(m, list):
                matrices_rows.append(m)

        eig_fp   = _matrix_eigenvalue_fingerprint(matrices_rows, dim)
        trace_fp = np.array(_trace_loop_fingerprint(matrices_rows, dim))
        combined = np.concatenate([eig_fp, trace_fp]) if len(trace_fp) else eig_fp

        fps_computed.append(combined)
        print(f"  fp={cand['fingerprint']}  eig_dim={len(eig_fp)}  "
              f"trace_dim={len(trace_fp)}", flush=True)

    # Greedy clustering
    family_id = [-1] * len(valid)
    next_fam   = 0

    for i in range(len(valid)):
        if family_id[i] != -1:
            continue
        family_id[i] = next_fam
        for j in range(i + 1, len(valid)):
            if family_id[j] != -1:
                continue
            # Same matrix size required
            if valid[i].get("matrix_size") != valid[j].get("matrix_size"):
                continue
            dist = _fp_distance(fps_computed[i], fps_computed[j])
            if dist < CLUSTER_THRESHOLD:
                family_id[j] = next_fam
        next_fam += 1

    # Build family map
    families: dict[int, list[dict]] = defaultdict(list)
    for i, cand in enumerate(valid):
        fam = family_id[i]
        families[fam].append({
            **cand,
            "stage2": {
                "family_id":  fam,
                "fp_vector":  fps_computed[i].tolist(),
            }
        })

    # Also include Stage-1-failed CMFs as singleton families
    failed = [r for r in stage1_results
              if not r.get("stage1", {}).get("walk_ok") or
                 r.get("stage1", {}).get("classification") == "RATIONAL"]
    for cand in failed:
        families[next_fam] = [{**cand, "stage2": {"family_id": next_fam, "excluded": True}}]
        next_fam += 1

    n_families = sum(1 for f in families.values() if not f[0].get("stage2", {}).get("excluded"))
    print(f"\n  Distinct mathematical families: {n_families}")
    for fam_id, members in sorted(families.items()):
        if members[0].get("stage2", {}).get("excluded"):
            continue
        classifs = [m.get("stage1", {}).get("classification","?") for m in members]
        sizes = [str(m.get("matrix_size","?")) for m in members]
        print(f"    Family {fam_id:3d}: {len(members)} members  "
              f"size={sizes[0]}×{sizes[0]}  "
              f"classifications={set(classifs)}")

    return {
        "families": {k: v for k, v in families.items()},
        "n_families":  n_families,
        "n_excluded":  len(failed),
        "total_valid": len(valid),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — export_paper_data
# ══════════════════════════════════════════════════════════════════════════════

def _matrix_rows_to_latex(label: str, rows: list[list], dim: int) -> str:
    """Render a matrix as a LaTeX pmatrix."""
    latex_rows = []
    for row in rows[:dim]:
        cells = []
        for entry in row[:dim]:
            s = str(entry)
            # Basic Python → LaTeX cosmetics
            s = s.replace("**", "^").replace("*", " \\cdot ")
            s = s.replace("sqrt(", "\\sqrt{").replace(")", "}")
            cells.append(s)
        latex_rows.append(" & ".join(cells))
    body = " \\\\\n    ".join(latex_rows)
    return f"K_{{{label}}} = \\begin{{pmatrix}}\n    {body}\n\\end{{pmatrix}}"


def _make_theorem_snippet(family_id: int, representative: dict) -> str:
    """Generate a LaTeX theorem draft for one family representative."""
    s1  = representative.get("stage1", {})
    dim = int(representative.get("matrix_size") or 3)
    agent   = representative.get("agent_type", "?")
    classif = s1.get("classification", "UNKNOWN")
    lval    = s1.get("hp_value", "?")
    relation = s1.get("relation") or "No algebraic relation found in extended basis"

    mats_data = representative.get("matrices") or []
    mat_lines = []
    for i, m in enumerate(mats_data[:5]):
        rows = m["rows"] if isinstance(m, dict) and "rows" in m else m
        label = (m.get("label", f"K{i+1}") if isinstance(m, dict) else f"K{i+1}")
        mat_lines.append(_matrix_rows_to_latex(label, rows, dim))

    mat_block = "\n\\quad\n".join(mat_lines) if mat_lines else "\\text{(matrices not available)}"
    fp = representative.get("fingerprint", "?")

    tex = rf"""
%% ─── Family {family_id} ───────────────────────────────────────────────────
%% Agent: {agent}  |  Matrix size: {dim}×{dim}  |  Classification: {classif}
%% Fingerprint: {fp}

\begin{{theorem}}[CMF Family {family_id}]
The continued matrix fraction defined by the recurrence matrices
\[
{mat_block}
\]
evaluated on a \({dim}\)-dimensional lattice walk, converges to the value
\[
L_{{ {family_id} }} \approx {lval[:60]}
\]
which is classified as \textbf{{{classif.replace("_", "\\_")}}}.

\noindent Algebraic relation search (extended PSLQ, $\mathrm{{dps}} = {PSLQ_DPS}$):
\[
\text{{{relation[:120]}}}
\]
\end{{theorem}}

"""
    return tex


def export_paper_data(clustering: dict) -> tuple[dict, str]:
    """Stage 3: Export per-family JSON + LaTeX for paper."""
    print("\n" + "═" * 70)
    print("  STAGE 3 — Auto-Theorem Export")
    print("═" * 70)

    families = clustering.get("families", {})
    paper_json: dict[str, list] = {
        "metadata": {
            "generated_by":   "CMF_Distiller",
            "n_families":     clustering["n_families"],
            "n_total_valid":  clustering["total_valid"],
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pslq_dps":       PSLQ_DPS,
            "walk_depth":     WALK_DEPTH,
        },
        "families": [],
    }
    latex_sections = [
        r"""\documentclass{amsart}
\usepackage{amsmath,amssymb}
\title{CMF Irrational Limit Families \\ (Auto-generated by CMF\_Distiller)}
\date{\today}
\begin{document}
\maketitle
\section{Introduction}
This document contains theorem sketches for each gauge-equivalence family
identified among the CMF Scout irrational candidates. Each theorem was
automatically generated from high-precision ($\mathrm{dps}=1000$) matrix
walks and extended PSLQ identification.
\section{Theorem Drafts}
"""
    ]

    for fam_id_raw, members in sorted(families.items(), key=lambda x: int(x[0])):
        fam_id = int(fam_id_raw)
        if members[0].get("stage2", {}).get("excluded"):
            continue
        # Pick representative: highest best_delta (most convergent)
        rep = max(members, key=lambda m: float(m.get("best_delta") or 0))
        classif  = rep.get("stage1", {}).get("classification", "?")
        lval     = rep.get("stage1", {}).get("hp_value", "?")
        relation = rep.get("stage1", {}).get("relation")
        res_log  = rep.get("stage1", {}).get("residual_log10")

        fam_entry = {
            "family_id":    fam_id,
            "n_members":    len(members),
            "classification": classif,
            "representative_fingerprint": rep.get("fingerprint"),
            "limit_value_hp": lval,
            "limit_label":    rep.get("limit_label"),
            "algebraic_relation": relation,
            "residual_log10":  res_log,
            "agent_type":     rep.get("agent_type"),
            "matrix_size":    rep.get("matrix_size"),
            "matrices":       rep.get("matrices"),
            "all_members": [
                {
                    "fingerprint": m.get("fingerprint"),
                    "agent_type":  m.get("agent_type"),
                    "best_delta":  m.get("best_delta"),
                    "classification": m.get("stage1", {}).get("classification"),
                    "limit_label": m.get("limit_label"),
                }
                for m in members
            ],
        }
        paper_json["families"].append(fam_entry)

        # LaTeX snippet
        latex_sections.append(_make_theorem_snippet(fam_id, rep))
        print(f"  Family {fam_id:3d}: {classif:<30}  {len(members)} members  "
              f"L≈{str(lval)[:30]}")

    latex_sections.append(r"\end{document}")
    latex_full = "\n".join(latex_sections)

    print(f"\n  Exported {len(paper_json['families'])} families.")
    return paper_json, latex_full


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="CMF_Distiller — 3-stage pipeline")
    ap.add_argument("--stage", type=int, default=0,
                    help="Run only this stage (0 = all)")
    ap.add_argument("--max", type=int, default=0,
                    help="Limit to first N candidates (0=all)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip fingerprints already in Stage 1 output")
    args = ap.parse_args()

    t_start = time.time()

    print("═" * 70)
    print("  CMF_Distiller  —  Publication-Grade Mathematical Rigor Pipeline")
    print("═" * 70)

    # Load candidates
    candidates = json.loads(CANDIDATES_JSON.read_text())
    if args.max:
        candidates = candidates[:args.max]
    print(f"  Candidates loaded: {len(candidates)}")

    # Load store index
    print("  Building store index …")
    store_idx = _load_store_index()
    print(f"  Store index: {len(store_idx)} records")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    stage1_results: list[dict] = []

    if args.stage in (0, 1):
        resume_fps: set[str] = set()
        if args.resume and STAGE1_OUT.exists():
            prev = []
            for line in STAGE1_OUT.read_text().splitlines():
                try:
                    r = json.loads(line)
                    resume_fps.add(r.get("fingerprint", ""))
                    prev.append(r)
                except Exception:
                    pass
            print(f"  Resuming Stage 1 — {len(resume_fps)} already done")
            stage1_results = prev

        new_results = verify_transcendence(candidates, store_idx, resume_fps)
        stage1_results.extend(new_results)

        STAGE1_OUT.parent.mkdir(exist_ok=True)
        with open(STAGE1_OUT, "w") as f:
            for r in stage1_results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  Stage 1 output → {STAGE1_OUT}")

    elif STAGE1_OUT.exists():
        stage1_results = [json.loads(l) for l in STAGE1_OUT.read_text().splitlines() if l.strip()]
        print(f"  Loaded Stage 1 output: {len(stage1_results)} records")
    else:
        print("  ERROR: No Stage 1 output found. Run with --stage 1 or --stage 0 first.")
        sys.exit(1)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    clustering: dict = {}

    if args.stage in (0, 2):
        clustering = cluster_families(stage1_results)
        STAGE2_OUT.parent.mkdir(exist_ok=True)
        # Serialize (converting numpy arrays to lists)
        def _json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")
        STAGE2_OUT.write_text(json.dumps(clustering, indent=2, default=_json_safe))
        print(f"\n  Stage 2 output → {STAGE2_OUT}")

    elif STAGE2_OUT.exists():
        clustering = json.loads(STAGE2_OUT.read_text())
        print(f"  Loaded Stage 2 output: {clustering.get('n_families')} families")
    else:
        print("  ERROR: No Stage 2 output found. Run with --stage 2 or --stage 0 first.")
        sys.exit(1)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    if args.stage in (0, 3):
        paper_json, latex_str = export_paper_data(clustering)

        STAGE3_OUT_JSON.parent.mkdir(exist_ok=True)
        STAGE3_OUT_JSON.write_text(json.dumps(paper_json, indent=2))
        STAGE3_OUT_TEX.write_text(latex_str)

        print(f"\n  Stage 3 JSON  → {STAGE3_OUT_JSON}")
        print(f"  Stage 3 LaTeX → {STAGE3_OUT_TEX}")

    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  CMF_Distiller complete in {elapsed:.1f}s")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
