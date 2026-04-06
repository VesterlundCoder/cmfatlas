#!/usr/bin/env python3
"""
patch_matrices_fast.py
======================
Fast pass: copy pre-computed X0/X1/X2 from store files into canonical_payload.
No SymPy. Commits every 500 rows. Finishes in ~30s.
"""
import json, sqlite3, time
from pathlib import Path

HERE  = Path(__file__).parent
GAUGE = HERE / "gauge_agents"
DB    = HERE / "data" / "atlas_2d.db"

_K   = ["Kx","Ky","Kz","Kw","Kv"]
_AX  = ["x","y","z","w","v"]

print("Step 1/3 — scanning store files …")
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
        # prefer record that has X0
        if existing is None:
            idx[fp] = rec
        elif "X0" not in existing and "X0" in rec:
            idx[fp] = rec
print(f"  {len(idx)} fingerprints  |  {sum(1 for r in idx.values() if 'X0' in r)} have X0  ({time.time()-t0:.1f}s)")

print("Step 2/3 — loading empty-matrix representations …")
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = con.execute("""
    SELECT r.id, r.canonical_fingerprint AS fp, r.canonical_payload
    FROM representation r
    WHERE r.primary_group = 'gauge_agent'
      AND (json_array_length(json_extract(r.canonical_payload,'$.matrices'))=0
           OR json_extract(r.canonical_payload,'$.matrices') IS NULL)
""").fetchall()
print(f"  {len(rows)} representations need patching")

print("Step 3/3 — patching …")
cur = con.cursor()
t1 = time.time()
updated = skipped = 0
BATCH = 500

for i, row in enumerate(rows):
    fp  = (row["fp"] or "")[:16]
    rec = idx.get(fp)
    if not rec or "X0" not in rec:
        skipped += 1
        continue

    dim    = int(rec.get("dim", 3))
    n_axes = 0
    for k in range(dim):
        if f"X{k}" in rec:
            n_axes += 1
        else:
            break
    n_axes = min(n_axes, len(_K))
    if n_axes == 0:
        skipped += 1
        continue

    matrices = []
    for k in range(n_axes):
        matrices.append({
            "label":  _K[k],
            "axis":   _AX[k],
            "index":  k,
            "source": "symbolic",
            "rows":   rec[f"X{k}"],
        })

    try:
        cp = json.loads(row["canonical_payload"])
    except Exception:
        skipped += 1
        continue

    cp["matrices"] = matrices
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
print(f"  Updated: {updated}")
print(f"  Skipped (no X0 in store): {skipped}")
