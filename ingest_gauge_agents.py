#!/usr/bin/env python3
"""
ingest_gauge_agents.py — Add all gauge_agents CMFs to atlas_2d.db
===================================================================
Reads store_*.jsonl files from gauge_agents/ and inserts every CMF
into the existing atlas_2d.db under project "Gauge Bootstrap Research".

Enriches with analysis scores from global_summary.csv where available.
Marks the 5 rational-limit CMFs as A_plus / primary_constant set.

Usage:
    python3 ingest_gauge_agents.py [--db data/atlas_2d.db] [--dry-run]
"""
import argparse, csv, json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
GAUGE_DIR = HERE / "gauge_agents"
NOW = datetime.now(timezone.utc).isoformat()

IDENTIFIED_LIMITS: dict = {
    "gauge_A_c02c792c009aa6b8": "-2",
    "gauge_A_4341aad295915872": "5/4",
    "gauge_A_af50bd587fe67d57": "1/7",
    "gauge_A_6ce2fcaa88fbd632": "-5/2",
    "gauge_A_35e44a13db2f46e6": "1/3",
}

# ── Load analysis results from global_summary.csv ─────────────────────────────
def load_analysis(cmf_harvester_root: Path) -> dict:
    csv_path = cmf_harvester_root / "cmf_analysis_results" / "global_summary.csv"
    if not csv_path.exists():
        print(f"  [warn] global_summary.csv not found at {csv_path}")
        return {}
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("cmf_id", "")
            if cid:
                result[cid] = row
    print(f"  Loaded analysis for {len(result)} CMFs from global_summary.csv")
    return result

# ── Determine certification level ─────────────────────────────────────────────
def cert_level(cmf_id: str, score: float, analysis: dict) -> str:
    if cmf_id in IDENTIFIED_LIMITS:
        return "A_plus"
    row = analysis.get(cmf_id, {})
    ts = float(row.get("total_score", score) or score)
    if ts >= 0.75:
        return "A_certified"
    if ts >= 0.50:
        return "B_verified_numeric"
    return "C_scouting"

# ── Build compact cmf_payload ──────────────────────────────────────────────────
def build_payload(rec: dict, cmf_id: str, analysis: dict) -> dict:
    agent = rec.get("agent", "A")
    dim   = int(rec.get("dim", 3))
    fp    = rec.get("fingerprint", "")[:16]
    score = float(rec.get("score", 0) or 0)
    row   = analysis.get(cmf_id, {})

    def _f(k):
        try: return float(row.get(k) or rec.get(k) or 0)
        except: return None

    primary_const = IDENTIFIED_LIMITS.get(cmf_id)
    cert          = cert_level(cmf_id, score, analysis)
    src_cat       = f"Gauge Agent {agent}"

    return {
        "gauge_id":          cmf_id,
        "fingerprint":       fp,
        "matrix_size":       dim,
        "dimension":         3,
        "agent_type":        agent,
        "coupling_bucket":   rec.get("coupling_bucket") or row.get("coupling_bucket"),
        "best_delta":        _f("best_delta"),
        "total_score":       _f("total_score") or _f("score"),
        "A_score":           _f("A_score"),
        "B_score":           _f("B_score"),
        "C_score":           _f("C_score"),
        "P_score":           _f("P_score"),
        "DIO_score":         _f("DIO_score"),
        "congruence_depth":  _f("congruence_depth"),
        "ore_compression":   _f("ore_compression"),
        "bidir_ratio":       _f("bidir_ratio"),
        "certification_level": cert,
        "source":            "gauge_agents",
        "source_category":   src_cat,
        "primary_constant":  primary_const,
        "identified_constant": primary_const,
        "flatness_verified": row.get("symbolic_flat", "False") == "True",
        "symbolic_flat":     row.get("symbolic_flat", "False") == "True",
        "degree":            0,
        "hidden":            False,
        "construction_type": "matrix_explicit",
        "proof_status":      "verified" if primary_const else "numeric_only",
        "identification_status": "pslq_identified" if primary_const else "unidentified",
        "source_family":     f"gauge_agent_{agent.lower()}_{dim}x{dim}",
    }

# ── Build canonical_payload for representation ────────────────────────────────
def build_canon_payload(rec: dict, fp: str) -> dict:
    agent = rec.get("agent", "A")
    dim   = int(rec.get("dim", 3))
    return {
        "fingerprint":   fp,
        "matrix_size":   dim,
        "axes":          3,
        "source_type":   "gauge_agent",
        "agent_type":    agent,
        "dim":           f"{dim}x{dim}",
    }

# ── Main ingest ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=HERE / "data" / "atlas_2d.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--harvester-root", type=Path,
                    default=Path("/Users/davidsvensson/Desktop/rd-lumi-z3") /
                            "CMF_Z5_SAGEMATH/Math Paper material/cmf_harvester")
    args = ap.parse_args()

    print(f"DB: {args.db}")
    analysis = load_analysis(args.harvester_root)

    if args.dry_run:
        print("DRY RUN — no writes")

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # ── Ensure category column exists ────────────────────────────────────────
    cols = [r[1] for r in cur.execute("PRAGMA table_info(cmf)").fetchall()]
    if "category" not in cols:
        cur.execute("ALTER TABLE cmf ADD COLUMN category TEXT")

    # ── Get or create project ────────────────────────────────────────────────
    existing_proj = cur.execute(
        "SELECT id FROM project WHERE name = ?", ("Gauge Bootstrap Research",)
    ).fetchone()
    if existing_proj:
        proj_id = existing_proj["id"]
        print(f"  Using existing project id={proj_id}")
    else:
        if not args.dry_run:
            cur.execute("INSERT INTO project (name, created_at) VALUES (?, ?)",
                        ("Gauge Bootstrap Research", NOW))
            proj_id = cur.lastrowid
        else:
            proj_id = 9999
        print(f"  Created project id={proj_id}")

    # ── Series registry: one per (agent, dim) ───────────────────────────────
    series_cache: dict = {}

    def get_or_create_series(agent: str, dim: int) -> int:
        key = (agent, dim)
        if key in series_cache:
            return series_cache[key]
        sname = f"Gauge Agent {agent} {dim}×{dim}"
        sdef  = f"Matrix CMF with {dim}×{dim} polynomial matrices along 3 lattice axes; Agent {agent}"
        row = cur.execute(
            "SELECT id FROM series WHERE name = ? AND project_id = ?",
            (sname, proj_id)
        ).fetchone()
        if row:
            series_cache[key] = row["id"]
            return row["id"]
        if not args.dry_run:
            cur.execute(
                "INSERT INTO series (project_id, name, definition, generator_type, "
                "provenance, created_at) VALUES (?,?,?,?,?,?)",
                (proj_id, sname, sdef, "gauge_matrix",
                 "Gauge Bootstrap Research pipeline (Vesterlund 2026)", NOW)
            )
            sid = cur.lastrowid
        else:
            sid = 10000 + len(series_cache)
        series_cache[key] = sid
        return sid

    # ── Load existing fingerprints to skip duplicates ─────────────────────
    existing_fps = set(
        r[0] for r in cur.execute(
            "SELECT canonical_fingerprint FROM representation WHERE primary_group = 'gauge_agent'"
        ).fetchall()
    )
    print(f"  Already in DB (gauge_agent group): {len(existing_fps)}")

    # ── Iterate store files ───────────────────────────────────────────────
    total_inserted = 0
    total_skipped  = 0
    store_files = sorted(GAUGE_DIR.glob("store_*.jsonl"))
    print(f"  Store files found: {len(store_files)}")

    for sf in store_files:
        fname = sf.name
        inserted_file = 0

        for line in sf.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            fp16    = (rec.get("fingerprint") or "")[:16]
            agent   = rec.get("agent", "A")
            dim     = int(rec.get("dim", 3))
            cmf_id  = f"gauge_{agent}_{fp16}"

            if fp16 in existing_fps:
                total_skipped += 1
                continue

            series_id     = get_or_create_series(agent, dim)
            payload       = build_payload(rec, cmf_id, analysis)
            canon_payload = build_canon_payload(rec, fp16)
            cert          = payload["certification_level"]

            # Assign category (same logic as API startup)
            primary_const = payload.get("primary_constant")
            if primary_const:
                category = "discovery"
            elif float(payload.get("total_score") or 0) >= 0.60:
                category = "interesting"
            else:
                category = "reference"

            if not args.dry_run:
                cur.execute(
                    "INSERT INTO representation "
                    "(series_id, primary_group, canonical_fingerprint, "
                    "canonical_payload, created_at) VALUES (?,?,?,?,?)",
                    (series_id, "gauge_agent", fp16,
                     json.dumps(canon_payload), NOW)
                )
                repr_id = cur.lastrowid

                cur.execute(
                    "INSERT INTO cmf "
                    "(representation_id, cmf_payload, dimension, "
                    "direction_policy, created_at, category) VALUES (?,?,?,?,?,?)",
                    (repr_id, json.dumps(payload), 3, None, NOW, category)
                )

            existing_fps.add(fp16)
            inserted_file += 1
            total_inserted += 1

        print(f"  {fname}: +{inserted_file} (skipped {total_skipped} so far)")

        if not args.dry_run and inserted_file > 0:
            con.commit()

    if not args.dry_run:
        con.commit()

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Done.")
    print(f"  Inserted: {total_inserted}")
    print(f"  Skipped:  {total_skipped}")

    # Final counts
    total_cmfs = cur.execute("SELECT COUNT(*) FROM cmf").fetchone()[0]
    print(f"  Total CMFs in DB: {total_cmfs}")
    db_size = args.db.stat().st_size
    print(f"  DB size: {db_size / 1024 / 1024:.1f} MB")

    con.close()


if __name__ == "__main__":
    main()
