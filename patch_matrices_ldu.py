#!/usr/bin/env python3
"""
patch_matrices_ldu.py
=====================
For gauge_agent CMFs that still have no matrices:
build L, D(n), U string matrices directly from params (no SymPy, no inversion).
Stores three matrices per CMF: L, D(n), U — fully renderable by the frontend.
Commits every 1000 rows. Finishes in < 30s for all 19k CMFs.
"""
import json, sqlite3, time
from fractions import Fraction
from pathlib import Path

HERE  = Path(__file__).parent
GAUGE = HERE / "gauge_agents"
DB    = HERE / "data" / "atlas_2d.db"

VAR_NAMES = ["x", "y", "z", "w", "v", "u", "t", "s"]


def _rat_str(v: float) -> str:
    """Float → clean fraction string, e.g. 0.5 → '1/2'."""
    try:
        f = Fraction(v).limit_denominator(1000)
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        return str(round(v, 6))


def build_ldu_matrices(params: dict, dim: int, axes: list[str]) -> list[dict] | None:
    """Return [L_mat, D_mat, U_mat] as list of canonical matrix dicts."""
    D_pars = params.get("D_params", [])
    L_off  = {eval(str(k)) if isinstance(k, str) else tuple(k): v
              for k, v in params.get("L_off", {}).items()}
    U_off  = {eval(str(k)) if isinstance(k, str) else tuple(k): v
              for k, v in params.get("U_off", {}).items()}

    if not D_pars:
        return None

    n_axes = len(D_pars)
    n_axes = min(n_axes, dim, len(VAR_NAMES))
    if n_axes == 0:
        return None

    var_labels = (axes[:n_axes] if axes and len(axes) >= n_axes else VAR_NAMES[:n_axes])

    # ── L matrix (lower triangular) ───────────────────────────────────────
    L_rows = [["0"] * dim for _ in range(dim)]
    for i in range(dim):
        L_rows[i][i] = "1"
    for (i, j), v in L_off.items():
        if 0 <= i < dim and 0 <= j < dim:
            L_rows[i][j] = _rat_str(v)

    # ── D(n) matrix (diagonal, linear in variables) ───────────────────────
    D_rows = [["0"] * dim for _ in range(dim)]
    for k in range(dim):
        if k < len(D_pars):
            a, b = float(D_pars[k][0]), float(D_pars[k][1])
            var  = var_labels[k % n_axes]
            a_s  = _rat_str(a)
            b_s  = _rat_str(b)
            if a == 0:
                D_rows[k][k] = b_s
            elif b == 0:
                D_rows[k][k] = f"{a_s}*{var}"
            elif a == 1:
                D_rows[k][k] = f"{var} + {b_s}" if float(b) > 0 else f"{var} - {_rat_str(-float(b))}"
            elif a == -1:
                D_rows[k][k] = f"-{var} + {b_s}" if float(b) > 0 else f"-{var} - {_rat_str(-float(b))}"
            else:
                sign = "+" if float(b) > 0 else "-"
                D_rows[k][k] = f"{a_s}*{var} {sign} {_rat_str(abs(float(b)))}"

    # ── U matrix (upper triangular) ───────────────────────────────────────
    U_rows = [["0"] * dim for _ in range(dim)]
    for i in range(dim):
        U_rows[i][i] = "1"
    for (i, j), v in U_off.items():
        if 0 <= i < dim and 0 <= j < dim:
            U_rows[i][j] = _rat_str(v)

    return [
        {"label": "L",    "axis": "lower",    "index": 0, "source": "ldu_L",    "rows": L_rows},
        {"label": "D(n)", "axis": "diagonal", "index": 1, "source": "ldu_D",    "rows": D_rows},
        {"label": "U",    "axis": "upper",    "index": 2, "source": "ldu_U",    "rows": U_rows},
    ]


# ── Build store index (params only, no X0 required) ──────────────────────
print("Step 1/3 — building store index (params) …")
t0 = time.time()
idx: dict[str, dict] = {}
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
        existing = idx.get(fp)
        if existing is None:
            idx[fp] = rec
        elif "X0" not in existing and "X0" in rec:
            idx[fp] = rec
print(f"  {len(idx)} fingerprints  ({time.time()-t0:.1f}s)")

# ── Load still-empty representations ─────────────────────────────────────
print("Step 2/3 — loading remaining empty-matrix representations …")
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = con.execute("""
    SELECT r.id, r.canonical_fingerprint AS fp, r.canonical_payload
    FROM representation r
    WHERE r.primary_group = 'gauge_agent'
      AND (json_array_length(json_extract(r.canonical_payload,'$.matrices'))=0
           OR json_extract(r.canonical_payload,'$.matrices') IS NULL)
""").fetchall()
print(f"  {len(rows)} still need patching")

# ── Patch ─────────────────────────────────────────────────────────────────
print("Step 3/3 — patching with LDU matrices …")
t1 = time.time()
cur = con.cursor()
updated = skipped = 0
BATCH = 1000

for i, row in enumerate(rows):
    fp  = (row["fp"] or "")[:16]
    rec = idx.get(fp)
    if not rec:
        skipped += 1
        continue

    try:
        cp = json.loads(row["canonical_payload"])
    except Exception:
        skipped += 1
        continue

    params = rec.get("params", {})
    if "D_params" not in params:
        skipped += 1
        continue

    dim    = int(rec.get("dim", 3))
    axes   = cp.get("axes", [])
    mats   = build_ldu_matrices(params, dim, axes)
    if not mats:
        skipped += 1
        continue

    cp["matrices"] = mats
    cur.execute("UPDATE representation SET canonical_payload=? WHERE id=?",
                (json.dumps(cp), row["id"]))
    updated += 1

    if updated % BATCH == 0:
        con.commit()
        elapsed = time.time() - t1
        rate = updated / elapsed
        print(f"  {updated} updated, {skipped} skipped  ({rate:.0f}/s)")

con.commit()
con.close()

print(f"\nDone in {time.time()-t1:.1f}s")
print(f"  Updated with LDU form: {updated}")
print(f"  Skipped:               {skipped}")
