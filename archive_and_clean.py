"""
archive_and_clean.py
====================
1. Export all Euler2AI CMFs to Desktop/cmf_atlas/euler2ai_archive/ as JSON.
2. Remove Euler2AI CMFs from atlas_2d.db.
3. Remove 2D CMFs whose f_poly only depends on x (no y variable).
4. Cascade-clean orphans and VACUUM.
"""
import json, re, shutil, sqlite3
from pathlib import Path

DB      = Path("data/atlas_2d.db")
ARCHIVE = Path("/Users/davidsvensson/Desktop/cmf_atlas/euler2ai_archive")
ARCHIVE.mkdir(exist_ok=True)

# ── Backup ────────────────────────────────────────────────────────────────────
shutil.copy2(DB, DB.with_suffix(".db.pre_euler2ai"))
print(f"Backup → {DB.with_suffix('.db.pre_euler2ai')}")

con = sqlite3.connect(DB)
cur = con.cursor()
rows = cur.execute("SELECT id, cmf_payload, dimension FROM cmf").fetchall()
print(f"CMFs before cleanup: {len(rows)}")

# ── 1. Collect IDs to remove ──────────────────────────────────────────────────
euler2ai_ids = []
one_var_ids  = []
euler2ai_export = []

for (cid, p, dim) in rows:
    d   = json.loads(p) if p else {}
    src = d.get("source", "")
    fpoly = d.get("f_poly", "") or ""

    if src == "euler2ai":
        euler2ai_ids.append(cid)
        euler2ai_export.append({"cmf_id": cid, "dimension": dim, **d})

    elif dim == 2 and fpoly and not re.search(r"\by\b", fpoly):
        one_var_ids.append(cid)

print(f"Euler2AI to archive+remove : {len(euler2ai_ids)}")
print(f"Single-variable (x only)   : {len(one_var_ids)}")

# ── 2. Write Euler2AI archive ─────────────────────────────────────────────────
archive_file = ARCHIVE / "euler2ai_cmfs.json"
with open(archive_file, "w") as f:
    json.dump(euler2ai_export, f, indent=2)
print(f"Euler2AI archive → {archive_file}  ({len(euler2ai_export)} entries)")

# ── 3. Delete both sets ───────────────────────────────────────────────────────
all_remove = euler2ai_ids + one_var_ids
ph = ",".join("?" * len(all_remove))
cur.execute(f"DELETE FROM cmf       WHERE id IN ({ph})", all_remove)
cur.execute(f"DELETE FROM eval_run  WHERE cmf_id IN ({ph})", all_remove)

# ── 4. Cascade-clean orphans ──────────────────────────────────────────────────
cur.execute("DELETE FROM recognition_attempt WHERE eval_run_id NOT IN (SELECT id FROM eval_run)")
cur.execute("DELETE FROM features WHERE representation_id NOT IN (SELECT DISTINCT representation_id FROM cmf)")
cur.execute("DELETE FROM representation_equivalence WHERE representation_id NOT IN (SELECT DISTINCT representation_id FROM cmf)")
cur.execute("DELETE FROM equivalence_class WHERE id NOT IN (SELECT DISTINCT equivalence_class_id FROM representation_equivalence)")
cur.execute("DELETE FROM representation WHERE id NOT IN (SELECT DISTINCT representation_id FROM cmf)")
cur.execute("DELETE FROM series WHERE id NOT IN (SELECT DISTINCT series_id FROM representation)")
cur.execute("DELETE FROM project WHERE id NOT IN (SELECT DISTINCT project_id FROM series)")
con.commit()

# ── 5. VACUUM ─────────────────────────────────────────────────────────────────
print("Running VACUUM…")
con.execute("VACUUM")
con.close()

remaining = sqlite3.connect(DB).execute("SELECT COUNT(*) FROM cmf").fetchone()[0]
sz_mb = DB.stat().st_size / 1_048_576
print(f"\nDone. Remaining CMFs: {remaining}  DB size: {sz_mb:.2f} MB")

# ── 6. Source breakdown of survivors ─────────────────────────────────────────
con2 = sqlite3.connect(DB)
src_rows = con2.execute(
    "SELECT json_extract(cmf_payload,'$.source_category'), COUNT(*) "
    "FROM cmf GROUP BY 1 ORDER BY 2 DESC"
).fetchall()
cert_rows = con2.execute(
    "SELECT json_extract(cmf_payload,'$.certification_level'), COUNT(*) "
    "FROM cmf GROUP BY 1 ORDER BY 2 DESC"
).fetchall()
con2.close()

print("\nSource categories remaining:")
for s, c in src_rows:
    print(f"  {s}: {c}")
print("\nCertification levels remaining:")
for s, c in cert_rows:
    print(f"  {s}: {c}")
