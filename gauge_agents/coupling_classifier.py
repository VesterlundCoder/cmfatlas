#!/usr/bin/env python3
"""
coupling_classifier.py — Classify all CMFs into 4 coupling buckets
===================================================================

Bucket 1 — Fully separable
  All X_i diagonal; each depends only on n_i. Tensor product of 1D objects.
  → Archive / reject.

Bucket 2 — Weakly coupled
  Some off-diagonal entries, but only in ONE triangular direction across all
  X_i (lower OR upper, not both), OR bidir_ratio < 0.10.
  → Exploratory edge cases.

Bucket 3 — Pairwise nontrivial
  Bidirectional off-diagonal entries exist in some but not all axis pairs.
  bidir_ratio ∈ [0.10, 0.50).  Valid multidimensional candidates.

Bucket 4 — Fully bidirectionally coupled
  Dense bidirectional coupling: bidir_ratio ≥ 0.50 AND multi-coord entries
  present. These are the genuine higher-dimensional CMF candidates.

Metrics per CMF (all numerical):
  off_diag_density  — fraction of off-diagonal entries non-zero (across all X_i)
  bidir_ratio       — fraction of off-diag pairs that are bidirectional
  coord_dep_ratio   — fraction of X_i entries sensitive to coords ≠ n_i
  max_offdiag       — largest |off-diagonal| entry (magnitude)
  coupling_score    — weighted composite [0,1]

Output:
  coupling_report.md     — full table + per-bucket breakdown
  coupling_summary.json  — machine-readable counts + per-record bucket
  store_{A,B,C}_bucket{1..4}.jsonl  — per-bucket filtered stores
"""
from __future__ import annotations
import json, math, warnings
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent
TOL  = 1e-7     # threshold for "non-zero"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Eval-function rebuilders for each agent format
# ══════════════════════════════════════════════════════════════════════════════

def _pk(s: str) -> tuple:
    return tuple(int(x.strip()) for x in str(s).strip("()").split(","))


def _build_fns_ldu(rec: dict):
    """
    Rebuild eval_fns for Agent B and Agent C records (LDU param format).
    eval_fns[ax](*pos) → dim×dim ndarray   (pos has `dim` elements)
    """
    dim     = rec["dim"]
    params  = rec["params"]

    L_off   = {_pk(k): float(v) for k, v in params.get("L_off", {}).items()}
    U_off   = {_pk(k): float(v) for k, v in params.get("U_off", {}).items()}
    D_pars  = [(float(a), float(b)) for a, b in params["D_params"]]

    def G_fn(pos):
        L = np.eye(dim, dtype=float)
        for (i, j), v in L_off.items():
            L[i, j] = v
        d_diag = np.array(
            [a * (pos[k % dim] + b) for k, (a, b) in enumerate(D_pars)]
        )
        U = np.eye(dim, dtype=float)
        for (i, j), v in U_off.items():
            U[i, j] = v
        return L @ np.diag(d_diag) @ U

    fns = []
    for ax in range(dim):
        def make_fn(axis=ax):
            def fn(*pos):
                pos = list(pos)
                G_n  = G_fn(pos)
                pos_sh = [p + (1 if k == axis else 0) for k, p in enumerate(pos)]
                G_sh = G_fn(pos_sh)
                Di   = np.diag([pos[axis] + k for k in range(dim)])
                det  = np.linalg.det(G_n)
                if abs(det) < 1e-9:
                    raise ValueError("singular G")
                return G_sh @ Di @ np.linalg.inv(G_n)
            return fn
        fns.append(make_fn())
    return fns, dim


def _build_fns_agent_a(rec: dict):
    """
    Rebuild eval_fns for Agent A records (diag/offdiag/d_params format).
    Note: Agent A only has 3 coordinate axes regardless of matrix dim.
    eval_fns[ax](x,y,z) → dim×dim ndarray
    """
    dim      = rec["dim"]
    params   = rec["params"]
    diag     = params["diag"]       # list of [a_i, b_i]
    offdiag  = params["offdiag"]    # list of [i,j,c,var_idx]
    d_params = rec.get("d_params", [])  # list of 3 axis-param lists

    def G_fn(x, y, z):
        coords = [x, y, z]
        G = np.zeros((dim, dim), dtype=float)
        for i, (a, b) in enumerate(diag):
            G[i, i] = float(a) * (coords[i % 3] + float(b))
        for entry in offdiag:
            i, j, c, vi = int(entry[0]), int(entry[1]), float(entry[2]), int(entry[3])
            G[i, j] = c * coords[vi % 3]
        return G

    shifts3 = [(1,0,0), (0,1,0), (0,0,1)]

    def Di_fn(axis, coord_val):
        if d_params and axis < len(d_params):
            dp = d_params[axis]
            return np.diag([float(a)*(coord_val + float(b) + k)
                            for k, (a, b) in enumerate(dp)])
        return np.diag([coord_val + k for k in range(dim)])

    # Agent A has exactly 3 X_i (one per coordinate)
    n_ax = 3
    fns = []
    for ax in range(n_ax):
        def make_fn(axis=ax):
            sx, sy, sz = shifts3[axis]
            def fn(x, y, z):
                G_n  = G_fn(x, y, z)
                G_sh = G_fn(x+sx, y+sy, z+sz)
                Di   = Di_fn(axis, [x,y,z][axis])
                det  = np.linalg.det(G_n)
                if abs(det) < 1e-9:
                    raise ValueError("singular G")
                return G_sh @ Di @ np.linalg.inv(G_n)
            return fn
        fns.append(make_fn())
    return fns, 3  # returns n_axes=3 for Agent A


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Coupling metrics
# ══════════════════════════════════════════════════════════════════════════════

def _test_points(n_ax: int, dim: int, n_pts: int = 8):
    """Generate test lattice points."""
    rng = np.random.default_rng(99)
    pts = []
    for _ in range(n_pts):
        if n_ax == 3:
            pts.append(list(int(v) for v in rng.integers(3, 14, 3)))
        else:
            pts.append(list(int(v) for v in rng.integers(3, 10, n_ax)))
    return pts


def compute_coupling_metrics(fns, n_ax: int, dim: int) -> dict:
    """
    fns   — eval_fns list (each fn(*pos) → dim×dim ndarray)
    n_ax  — number of axes (= len(fns))
    dim   — matrix size

    Returns dict with all coupling metrics.
    """
    pts = _test_points(n_ax, dim)

    # ── 1. Evaluate all matrices ──────────────────────────────────────────────
    mats = []   # list of (axis, matrix_ndarray)
    for pt in pts:
        for ax in range(n_ax):
            try:
                M = np.array(fns[ax](*pt), float)
                if M.shape == (dim, dim) and np.all(np.isfinite(M)):
                    mats.append((ax, M))
            except Exception:
                pass

    if not mats:
        return {
            "off_diag_density": 0.0, "bidir_ratio": 0.0,
            "coord_dep_ratio": 0.0, "max_offdiag": 0.0,
            "coupling_score": 0.0, "bucket": 0, "bucket_reason": "eval_failed",
            "n_mats_evaluated": 0,
        }

    # ── 2. Off-diagonal density ───────────────────────────────────────────────
    total_offdiag = len(mats) * dim * (dim - 1)
    n_nonzero_offdiag = sum(
        1 for _, M in mats
        for i in range(dim) for j in range(dim)
        if i != j and abs(M[i, j]) > TOL
    )
    off_diag_density = n_nonzero_offdiag / max(1, total_offdiag)

    max_offdiag = max(
        (abs(M[i, j]) for _, M in mats
         for i in range(dim) for j in range(dim) if i != j),
        default=0.0,
    )

    # ── 3. Bidirectionality ───────────────────────────────────────────────────
    # For each (axis, i, j) pair with i<j: bidirectional if both [i,j] and [j,i]
    # are non-zero in ANY of the evaluations for that axis.
    pair_has_ij  = {}   # (ax, i, j) -> bool
    pair_has_ji  = {}
    for ax, M in mats:
        for i in range(dim):
            for j in range(i + 1, dim):
                k = (ax, i, j)
                pair_has_ij[k]  = pair_has_ij.get(k, False)  or (abs(M[i, j]) > TOL)
                pair_has_ji[k]  = pair_has_ji.get(k, False)  or (abs(M[j, i]) > TOL)

    n_asym_pairs = sum(1 for k in pair_has_ij
                       if pair_has_ij[k] or pair_has_ji[k])
    n_bidir      = sum(1 for k in pair_has_ij
                       if pair_has_ij[k] and pair_has_ji[k])
    bidir_ratio  = n_bidir / max(1, n_asym_pairs)

    # ── 4. Upper vs lower triangularity bias ─────────────────────────────────
    n_lower = sum(1 for _, M in mats
                  for i in range(dim) for j in range(i)
                  if abs(M[i, j]) > TOL)
    n_upper = sum(1 for _, M in mats
                  for i in range(dim) for j in range(i + 1, dim)
                  if abs(M[i, j]) > TOL)
    total_tri = n_lower + n_upper
    # "one-sided" if strongly skewed
    if total_tri > 0:
        skew = abs(n_lower - n_upper) / total_tri
    else:
        skew = 0.0

    # ── 5. Coordinate dependency ratio ───────────────────────────────────────
    # For each axis ax, check whether X_ax[i,j] changes when varying coordinate
    # OTHER than ax (i.e., multi-coordinate dependence).
    n_dep = 0
    n_dep_total = 0
    for ax in range(n_ax):
        base_pt = [6] * n_ax
        try:
            M_base = np.array(fns[ax](*base_pt), float)
        except Exception:
            continue
        for other in range(n_ax):
            if other == ax:
                continue
            alt_pt = list(base_pt)
            alt_pt[other] = 10
            try:
                M_alt = np.array(fns[ax](*alt_pt), float)
            except Exception:
                continue
            for i in range(dim):
                for j in range(dim):
                    n_dep_total += 1
                    if abs(M_base[i, j] - M_alt[i, j]) > TOL:
                        n_dep += 1

    coord_dep_ratio = n_dep / max(1, n_dep_total)

    # ── 6. Coupling score ─────────────────────────────────────────────────────
    coupling_score = (
        0.40 * off_diag_density
        + 0.35 * bidir_ratio
        + 0.25 * coord_dep_ratio
    )

    # ── 7. Bucket assignment ──────────────────────────────────────────────────
    if off_diag_density < 1e-9:
        bucket = 1
        reason = "all X_i diagonal"
    elif bidir_ratio < 0.10 or (total_tri > 0 and skew > 0.85):
        bucket = 2
        reason = f"one-sided triangular (skew={skew:.2f}) or bidir<0.10"
    elif bidir_ratio < 0.50:
        bucket = 3
        reason = f"bidir={bidir_ratio:.2f}, partial coupling"
    else:
        bucket = 4
        reason = f"bidir={bidir_ratio:.2f}, coord_dep={coord_dep_ratio:.2f}"

    return {
        "off_diag_density":  round(off_diag_density, 5),
        "bidir_ratio":       round(bidir_ratio, 4),
        "coord_dep_ratio":   round(coord_dep_ratio, 4),
        "max_offdiag":       round(float(max_offdiag), 6),
        "coupling_score":    round(coupling_score, 4),
        "n_lower_offdiag":   n_lower,
        "n_upper_offdiag":   n_upper,
        "tri_skew":          round(skew, 4),
        "n_bidir_pairs":     n_bidir,
        "n_asym_pairs":      n_asym_pairs,
        "n_mats_evaluated":  len(mats),
        "bucket":            bucket,
        "bucket_reason":     reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Process all store files
# ══════════════════════════════════════════════════════════════════════════════

BUCKET_LABELS = {
    1: "Fully separable",
    2: "Weakly coupled",
    3: "Pairwise nontrivial",
    4: "Fully bidirectionally coupled",
}
BUCKET_VERDICT = {
    1: "Archive / reject",
    2: "Exploratory edge case",
    3: "Valid multidimensional candidate",
    4: "BEST — genuine CMF candidate",
}


def classify_all():
    stores = sorted(HERE.glob("store_[ABC]_*.jsonl"))
    if not stores:
        print("No store files found.")
        return

    all_records = []     # list of enriched dicts
    bucket_fh    = {}    # bucket -> open file handle

    # Open per-bucket output files
    out_dir = HERE
    for b in range(1, 5):
        p = out_dir / f"classified_bucket{b}.jsonl"
        bucket_fh[b] = open(p, "w")

    counts = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}  # 0 = eval_failed

    print(f"\n{'═'*72}")
    print("  Coupling classifier — all Agents A, B, C")
    print(f"{'═'*72}")

    for store_path in stores:
        lines = [l for l in store_path.read_text().splitlines() if l.strip()]
        if not lines:
            continue

        agent = "A" if "_A_" in store_path.name else \
                "B" if "_B_" in store_path.name else "C"
        dim = int(store_path.stem.split("_")[-1].split("x")[0])

        print(f"\n  {store_path.name}  ({len(lines)} records, agent={agent}, dim={dim})")

        for idx, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except Exception:
                continue

            rec_dim = rec.get("dim", dim)

            # Build eval functions
            try:
                if agent == "A":
                    fns, n_ax = _build_fns_agent_a(rec)
                else:
                    fns, n_ax = _build_fns_ldu(rec)
            except Exception as e:
                counts[0] += 1
                print(f"    [{idx+1}] build_fns error: {e}")
                continue

            # Compute metrics
            metrics = compute_coupling_metrics(fns, n_ax, rec_dim)
            b = metrics["bucket"]
            counts[b] += 1

            # Enrich record
            enriched = dict(rec)
            enriched.update({
                "coupling_bucket":       b,
                "bucket_label":          BUCKET_LABELS.get(b, "?"),
                "coupling_off_diag_density": metrics["off_diag_density"],
                "coupling_bidir_ratio":  metrics["bidir_ratio"],
                "coupling_coord_dep":    metrics["coord_dep_ratio"],
                "coupling_max_offdiag":  metrics["max_offdiag"],
                "coupling_score":        metrics["coupling_score"],
                "coupling_tri_skew":     metrics["tri_skew"],
                "coupling_n_bidir_pairs": metrics["n_bidir_pairs"],
                "coupling_reason":       metrics["bucket_reason"],
                "source_store":          store_path.name,
            })

            all_records.append(enriched)
            bucket_fh[b].write(json.dumps(enriched) + "\n")
            bucket_fh[b].flush()

            print(f"    [{idx+1:>4}] fp={rec.get('fingerprint','?')[:12]}  "
                  f"B{b}  offdiag={metrics['off_diag_density']:.3f}  "
                  f"bidir={metrics['bidir_ratio']:.2f}  "
                  f"coord_dep={metrics['coord_dep_ratio']:.2f}  "
                  f"score={metrics['coupling_score']:.3f}")

    for fh in bucket_fh.values():
        fh.close()

    return all_records, counts


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_records: list, counts: dict):
    total = sum(counts[b] for b in range(1, 5))

    lines = [
        "# CMF Coupling Classification Report\n",
        f"Total classified: **{total}**  |  eval failed: {counts[0]}\n",
        "",
        "## Bucket Summary\n",
        "| Bucket | Label | Verdict | Count | % |",
        "|--------|-------|---------|-------|---|",
    ]
    for b in range(1, 5):
        pct = 100 * counts[b] / max(1, total)
        lines.append(
            f"| {b} | {BUCKET_LABELS[b]} | {BUCKET_VERDICT[b]} "
            f"| {counts[b]} | {pct:.1f}% |"
        )

    lines += ["", "## Per-record Table\n",
              "| Source | fp | dim | B | off_diag | bidir | coord_dep | score | reason |",
              "|--------|----|-----|---|---------|-------|-----------|-------|--------|"]

    for r in sorted(all_records, key=lambda x: (-x["coupling_bucket"],
                                                  -x["coupling_score"])):
        lines.append(
            f"| {r.get('source_store','?')} "
            f"| `{r.get('fingerprint','?')[:10]}` "
            f"| {r.get('dim','?')} "
            f"| **B{r['coupling_bucket']}** "
            f"| {r['coupling_off_diag_density']:.3f} "
            f"| {r['coupling_bidir_ratio']:.2f} "
            f"| {r['coupling_coord_dep']:.2f} "
            f"| {r['coupling_score']:.3f} "
            f"| {r['coupling_reason']} |"
        )

    # ── Per-bucket detail sections ────────────────────────────────────────────
    for b in range(4, 0, -1):
        bucket_recs = [r for r in all_records if r["coupling_bucket"] == b]
        if not bucket_recs:
            continue
        lines += [
            "",
            f"## Bucket {b} — {BUCKET_LABELS[b]} ({len(bucket_recs)} records)\n",
            f"*{BUCKET_VERDICT[b]}*\n",
        ]
        for r in sorted(bucket_recs, key=lambda x: -x["coupling_score"])[:50]:
            lines.append(
                f"- `{r.get('fingerprint','?')[:16]}`  "
                f"dim={r.get('dim','?')}  agent={r.get('agent','?')}  "
                f"best_delta={r.get('best_delta',0):.2f}  "
                f"score={r['coupling_score']:.3f}  "
                f"bidir={r['coupling_bidir_ratio']:.2f}  "
                f"coord_dep={r['coupling_coord_dep']:.2f}"
            )
        if len(bucket_recs) > 50:
            lines.append(f"  … and {len(bucket_recs)-50} more (see classified_bucket{b}.jsonl)")

    report_path = HERE / "coupling_report.md"
    report_path.write_text("\n".join(lines))

    # JSON summary
    summary = {
        "total": total,
        "eval_failed": counts[0],
        "buckets": {str(b): counts[b] for b in range(1, 5)},
        "records": [
            {
                "fp": r.get("fingerprint", "")[:16],
                "dim": r.get("dim"),
                "agent": r.get("agent"),
                "bucket": r["coupling_bucket"],
                "score": r["coupling_score"],
                "bidir_ratio": r["coupling_bidir_ratio"],
                "coord_dep": r["coupling_coord_dep"],
                "source": r.get("source_store", ""),
            }
            for r in all_records
        ],
    }
    (HERE / "coupling_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'═'*72}")
    print("  BUCKET SUMMARY")
    print(f"{'═'*72}")
    for b in range(1, 5):
        pct = 100 * counts[b] / max(1, total)
        bar = "█" * int(pct / 2)
        print(f"  B{b}  {BUCKET_LABELS[b]:<36}  {counts[b]:>5}  {pct:5.1f}%  {bar}")
    print(f"  {'TOTAL':<38}  {total:>5}")
    print(f"\n  Report  : {report_path}")
    print(f"  Summary : {HERE / 'coupling_summary.json'}")
    for b in range(1, 5):
        print(f"  Bucket {b}: {HERE / f'classified_bucket{b}.jsonl'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = classify_all()
    if result:
        all_records, counts = result
        write_report(all_records, counts)
