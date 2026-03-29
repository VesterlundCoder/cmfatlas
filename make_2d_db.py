"""
make_2d_db.py
=============
Creates a clean atlas_2d.db from atlas.db containing only CMFs with dimension >= 2.
All orphaned rows in dependent tables are removed, then the DB is VACUUM-ed.

Usage:
    python make_2d_db.py
"""
import shutil
import sqlite3
from pathlib import Path

SRC = Path("data/atlas.db")
DST = Path("data/atlas_2d.db")


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source DB not found: {SRC}")

    print(f"Copying {SRC} → {DST} …")
    shutil.copy2(SRC, DST)

    con = sqlite3.connect(DST)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.cursor()

    # ── Count before ──────────────────────────────────────────────────────────
    def count(table):
        return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    before = {t: count(t) for t in [
        "project", "series", "representation", "cmf",
        "eval_run", "recognition_attempt"
    ]}
    print("\nBefore:")
    for t, n in before.items():
        print(f"  {t}: {n}")

    # ── 1. Remove 1D CMFs ─────────────────────────────────────────────────────
    cur.execute("DELETE FROM cmf WHERE dimension IS NULL OR dimension < 2")
    removed_cmfs = con.total_changes
    print(f"\nDeleted {removed_cmfs} 1D CMF rows")

    # ── 2. Cascade: recognition_attempt → eval_run → cmf ─────────────────────
    cur.execute("""
        DELETE FROM recognition_attempt
        WHERE eval_run_id NOT IN (SELECT id FROM eval_run)
           OR eval_run_id IN (
               SELECT er.id FROM eval_run er
               WHERE er.cmf_id NOT IN (SELECT id FROM cmf)
           )
    """)
    cur.execute("""
        DELETE FROM eval_run
        WHERE cmf_id NOT IN (SELECT id FROM cmf)
    """)

    # ── 3. Cascade: features / representation_equivalence / equivalence_class ─
    cur.execute("""
        DELETE FROM features
        WHERE representation_id NOT IN (
            SELECT DISTINCT representation_id FROM cmf
        )
    """)

    cur.execute("""
        DELETE FROM representation_equivalence
        WHERE representation_id NOT IN (
            SELECT DISTINCT representation_id FROM cmf
        )
    """)

    # Remove equivalence_classes with no remaining members
    cur.execute("""
        DELETE FROM equivalence_class
        WHERE id NOT IN (
            SELECT DISTINCT equivalence_class_id FROM representation_equivalence
        )
    """)

    # ── 4. Remove orphaned representations ────────────────────────────────────
    cur.execute("""
        DELETE FROM representation
        WHERE id NOT IN (
            SELECT DISTINCT representation_id FROM cmf
        )
    """)

    # ── 5. Remove orphaned series ─────────────────────────────────────────────
    cur.execute("""
        DELETE FROM series
        WHERE id NOT IN (
            SELECT DISTINCT series_id FROM representation
        )
    """)

    # ── 6. Remove orphaned projects ───────────────────────────────────────────
    cur.execute("""
        DELETE FROM project
        WHERE id NOT IN (
            SELECT DISTINCT project_id FROM series
        )
    """)

    con.commit()

    # ── Count after ───────────────────────────────────────────────────────────
    after = {t: count(t) for t in [
        "project", "series", "representation", "cmf",
        "eval_run", "recognition_attempt"
    ]}
    print("\nAfter:")
    for t, n in after.items():
        print(f"  {t}: {n}")

    # ── Dimension breakdown ───────────────────────────────────────────────────
    dims = cur.execute(
        "SELECT dimension, COUNT(*) FROM cmf GROUP BY dimension ORDER BY dimension"
    ).fetchall()
    print("\nDimension breakdown:")
    for dim, cnt in dims:
        print(f"  {dim}D: {cnt}")

    # ── Vacuum ────────────────────────────────────────────────────────────────
    print("\nRunning VACUUM …")
    con.execute("VACUUM")
    con.close()

    src_mb = SRC.stat().st_size / 1_048_576
    dst_mb = DST.stat().st_size / 1_048_576
    print(f"\nDone. {SRC.name}: {src_mb:.1f} MB → {DST.name}: {dst_mb:.1f} MB")
    print(f"atlas_2d.db written to {DST.resolve()}")


if __name__ == "__main__":
    main()
