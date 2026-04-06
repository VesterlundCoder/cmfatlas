#!/usr/bin/env python3
"""
purge_empty_cmfs.py — Remove all CMF entries that have no matrix data.
Operates on data/atlas_2d.db (local). Push to Railway after verifying.
"""
import sqlite3, json, argparse, sys
from pathlib import Path

DB = Path(__file__).parent / "data" / "atlas_2d.db"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts only, make no changes")
    parser.add_argument("--db", default=str(DB))
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    # Find CMF IDs with no matrices
    rows = con.execute("""
        SELECT c.id as cmf_id, r.id as rep_id,
               json_extract(r.canonical_payload, '$.matrices') as mx
        FROM cmf c
        JOIN representation r ON c.representation_id = r.id
    """).fetchall()

    empty_cmf_ids   = []
    empty_rep_ids   = set()
    for row in rows:
        mx = row["mx"]
        if mx is None or mx == "[]" or mx == "null":
            empty_cmf_ids.append(row["cmf_id"])
            empty_rep_ids.add(row["rep_id"])

    print(f"Total CMFs:        {len(rows)}")
    print(f"CMFs to DELETE:    {len(empty_cmf_ids)}")
    print(f"CMFs to KEEP:      {len(rows) - len(empty_cmf_ids)}")
    print(f"Representations affected: {len(empty_rep_ids)}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    print("\nDeleting…")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=OFF")

    # Delete dependent eval_run / recognition records first
    chunk = 500
    for i in range(0, len(empty_cmf_ids), chunk):
        ids = empty_cmf_ids[i:i+chunk]
        ph  = ",".join("?" * len(ids))
        # recognition → eval_run → cmf
        eval_ids = [r[0] for r in con.execute(
            f"SELECT id FROM eval_run WHERE cmf_id IN ({ph})", ids).fetchall()]
        if eval_ids:
            eph = ",".join("?" * len(eval_ids))
            con.execute(f"DELETE FROM recognition WHERE eval_run_id IN ({eph})", eval_ids)
            con.execute(f"DELETE FROM eval_run WHERE id IN ({eph})", eval_ids)
        con.execute(f"DELETE FROM cmf WHERE id IN ({ph})", ids)

    # Delete orphaned representations (no remaining cmf references)
    con.execute("""
        DELETE FROM representation
        WHERE id NOT IN (SELECT DISTINCT representation_id FROM cmf)
    """)

    # Delete orphaned series
    con.execute("""
        DELETE FROM series
        WHERE id NOT IN (SELECT DISTINCT series_id FROM representation)
    """)

    # Delete orphaned projects
    con.execute("""
        DELETE FROM project
        WHERE id NOT IN (SELECT DISTINCT project_id FROM series)
    """)

    con.commit()

    # Vacuum to reclaim space
    print("Running VACUUM…")
    con.execute("VACUUM")
    con.close()

    print(f"\n✓ Done. {len(empty_cmf_ids)} CMFs purged.")
    print("  Verify locally, then push data/atlas_2d.db to Railway.")

if __name__ == "__main__":
    main()
