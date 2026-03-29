"""
clean_and_upgrade_db.py
=======================
1. Remove empty CMF entries (no f_poly and no primary_constant).
2. Add A+ certification to all RamanujanTools entries.
3. Remap source → one of 4 canonical categories in payload.
4. Export full verification list (CSV, grouped C → B → A → A+).
5. Cascade-clean orphaned rows and VACUUM.

Source mapping:
  euler2ai                                → Euler2AI
  ramanujantools                          → RamanujanTools  (also → A_plus)
  cmf_hunter, ml_loop                     → CMF Hunter
  telescope, verified_2d, ore_algebra,
  training_seed_*, oeis_bank, known_family→ Gauge Transformed
"""
import csv
import json
import shutil
import sqlite3
from pathlib import Path

DB   = Path("data/atlas_2d.db")
BACK = Path("data/atlas_2d.db.bak")

SOURCE_MAP = {
    "euler2ai":              "Euler2AI",
    "ramanujantools":        "RamanujanTools",
    "cmf_hunter":            "CMF Hunter",
    "ml_loop":               "CMF Hunter",
    "telescope":             "Gauge Transformed",
    "verified_2d":           "Gauge Transformed",
    "ore_algebra":           "Gauge Transformed",
    "training_seed_series":  "Gauge Transformed",
    "training_seed_contfrac":"Gauge Transformed",
    "training_seed_exotic":  "Gauge Transformed",
    "oeis_bank":             "Gauge Transformed",
    "known_family":          "Gauge Transformed",
}


def main():
    # ── Backup ────────────────────────────────────────────────────────────────
    print(f"Backing up {DB} → {BACK}")
    shutil.copy2(DB, BACK)

    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.cursor()

    rows = cur.execute("SELECT id, cmf_payload FROM cmf").fetchall()
    print(f"Total CMFs before cleanup: {len(rows)}")

    # ── 1. Identify empty CMFs ────────────────────────────────────────────────
    empty_ids = []
    for (cid, payload) in rows:
        d = json.loads(payload) if payload else {}
        if not d.get("f_poly") and not d.get("primary_constant"):
            empty_ids.append(cid)

    print(f"Empty CMFs to remove: {len(empty_ids)}  ids={empty_ids[:10]}{'...' if len(empty_ids)>10 else ''}")

    if empty_ids:
        placeholders = ",".join("?" * len(empty_ids))
        cur.execute(f"DELETE FROM cmf WHERE id IN ({placeholders})", empty_ids)
        cur.execute(f"DELETE FROM eval_run WHERE cmf_id IN ({placeholders})", empty_ids)
        cur.execute(f"""DELETE FROM recognition_attempt WHERE eval_run_id IN (
            SELECT id FROM eval_run WHERE cmf_id IN ({placeholders}))""", empty_ids)

    # ── 2. Reload rows after deletion ─────────────────────────────────────────
    rows = cur.execute("SELECT id, cmf_payload FROM cmf").fetchall()
    print(f"CMFs after removal: {len(rows)}")

    # ── 3. Update payloads: A+ + source_category ─────────────────────────────
    updated = 0
    aplus_count = 0
    for (cid, payload) in rows:
        d = json.loads(payload) if payload else {}
        raw_src   = d.get("source", "")
        category  = SOURCE_MAP.get(raw_src, "CMF Hunter")

        changed = False
        if d.get("source_category") != category:
            d["source_category"] = category
            changed = True

        if raw_src == "ramanujantools" and d.get("certification_level") != "A_plus":
            d["certification_level"] = "A_plus"
            aplus_count += 1
            changed = True

        if changed:
            cur.execute("UPDATE cmf SET cmf_payload=? WHERE id=?",
                        (json.dumps(d), cid))
            updated += 1

    print(f"Payloads updated: {updated}  (A+ assigned: {aplus_count})")

    # ── 4. Cascade-clean orphans ──────────────────────────────────────────────
    cur.execute("DELETE FROM recognition_attempt WHERE eval_run_id NOT IN (SELECT id FROM eval_run)")
    cur.execute("DELETE FROM eval_run WHERE cmf_id NOT IN (SELECT id FROM cmf)")
    cur.execute("DELETE FROM features WHERE representation_id NOT IN (SELECT DISTINCT representation_id FROM cmf)")
    cur.execute("DELETE FROM representation_equivalence WHERE representation_id NOT IN (SELECT DISTINCT representation_id FROM cmf)")
    cur.execute("DELETE FROM equivalence_class WHERE id NOT IN (SELECT DISTINCT equivalence_class_id FROM representation_equivalence)")
    cur.execute("DELETE FROM representation WHERE id NOT IN (SELECT DISTINCT representation_id FROM cmf)")
    cur.execute("DELETE FROM series WHERE id NOT IN (SELECT DISTINCT series_id FROM representation)")
    cur.execute("DELETE FROM project WHERE id NOT IN (SELECT DISTINCT project_id FROM series)")
    con.commit()

    # ── 5. Certification breakdown ────────────────────────────────────────────
    cert_rows = cur.execute(
        "SELECT json_extract(cmf_payload,'$.certification_level'), COUNT(*) "
        "FROM cmf GROUP BY 1 ORDER BY 1"
    ).fetchall()
    print("\nCertification breakdown after upgrade:")
    for cert, cnt in cert_rows:
        print(f"  {cert}: {cnt}")

    # ── 6. Source category breakdown ──────────────────────────────────────────
    src_rows = cur.execute(
        "SELECT json_extract(cmf_payload,'$.source_category'), COUNT(*) "
        "FROM cmf GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    print("\nSource category breakdown:")
    for src, cnt in src_rows:
        print(f"  {src}: {cnt}")

    # ── 7. Export verification list ───────────────────────────────────────────
    ORDER = {"C_scouting": 0, "B_verified_numeric": 1, "A_certified": 2, "A_plus": 3}
    all_rows = cur.execute("SELECT id, cmf_payload, dimension FROM cmf").fetchall()

    records = []
    for (cid, payload, dim) in all_rows:
        d = json.loads(payload) if payload else {}
        records.append({
            "id":            cid,
            "dimension":     dim,
            "cert":          d.get("certification_level", "none"),
            "source":        d.get("source", ""),
            "source_category": d.get("source_category", ""),
            "f_poly":        d.get("f_poly", ""),
            "primary_constant": d.get("primary_constant", ""),
            "degree":        d.get("degree", ""),
        })

    records.sort(key=lambda x: (ORDER.get(x["cert"], -1), x["id"]))

    out_csv = Path("data/verification_list.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","cert","source_category","dimension","degree","f_poly","primary_constant","source"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nVerification list written → {out_csv.resolve()}")
    for cert in ["C_scouting","B_verified_numeric","A_certified","A_plus"]:
        n = sum(1 for r in records if r["cert"]==cert)
        print(f"  {cert}: {n}")

    # ── 8. VACUUM ─────────────────────────────────────────────────────────────
    print("\nRunning VACUUM…")
    con.execute("VACUUM")
    con.close()

    sz_mb = DB.stat().st_size / 1_048_576
    print(f"\nDone. atlas_2d.db → {sz_mb:.1f} MB")


if __name__ == "__main__":
    main()
