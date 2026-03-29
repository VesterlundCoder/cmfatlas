"""Import existing data sources into the atlas database.

Supports:
    - CMF database JSON (cmf_database.json)
    - Euler2AI PCF collections (pcfs.json / cmf_pcfs.json)
    - Generic JSONL files
"""

import json
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from cmf_atlas.db.models import CMF, Project, Representation, Series
from cmf_atlas.db.session import get_session, init_db
from cmf_atlas.canonical import canonicalize_and_fingerprint
from cmf_atlas.ingest.parsers import parse_cmf_db_entry, parse_euler2ai_pcf, parse_jsonl_line
from cmf_atlas.util.json import dumps
from cmf_atlas.util.logging import get_logger

log = get_logger("ingest")


def _upsert_representation(session, series, primary_group, payload, cmf_payload) -> Representation | None:
    """Insert a representation, skipping if fingerprint already exists."""
    try:
        fingerprint, canonical_json = canonicalize_and_fingerprint(primary_group, payload)
    except Exception as e:
        log.warning(f"Canonicalization failed for {series.name}: {e}")
        return None

    # Check for existing
    existing = (
        session.query(Representation)
        .filter_by(primary_group=primary_group, canonical_fingerprint=fingerprint)
        .first()
    )
    if existing:
        return None  # Duplicate

    rep = Representation(
        series_id=series.id,
        primary_group=primary_group,
        canonical_fingerprint=fingerprint,
        canonical_payload=canonical_json,
    )
    session.add(rep)
    session.flush()

    # Add CMF object if payload provided
    if cmf_payload:
        cmf = CMF(
            representation_id=rep.id,
            cmf_payload=dumps(cmf_payload),
            dimension=cmf_payload.get("dimension") or cmf_payload.get("dim"),
        )
        session.add(cmf)

    return rep


def import_cmf_database(
    db_path: str | Path,
    project_name: str = "CMF Database v2.2",
    atlas_db_path: str | Path | None = None,
) -> dict:
    """Import the CMF database JSON into the atlas.

    Returns stats dict.
    """
    engine = init_db(db_path=atlas_db_path)
    session = get_session(engine)

    stats = {"total": 0, "imported": 0, "skipped_dup": 0, "skipped_err": 0}

    try:
        # Create or find project
        project = session.query(Project).filter_by(name=project_name).first()
        if not project:
            project = Project(name=project_name)
            session.add(project)
            session.flush()

        # Load CMF database
        with open(db_path) as f:
            data = json.load(f)

        entries = data.get("cmfs", data.get("entries", []))
        if isinstance(data, list):
            entries = data

        log.info(f"Importing {len(entries)} CMF database entries...")
        stats["total"] = len(entries)

        for entry in entries:
            try:
                parsed = parse_cmf_db_entry(entry)

                series = Series(
                    project_id=project.id,
                    name=parsed["metadata"].get("cmf_db_id", ""),
                    definition=parsed["series_definition"],
                    generator_type=parsed["generator_type"],
                    provenance=parsed["provenance"],
                )
                session.add(series)
                session.flush()

                rep = _upsert_representation(
                    session, series,
                    parsed["primary_group"],
                    parsed["payload"],
                    parsed["cmf_payload"],
                )
                if rep:
                    stats["imported"] += 1
                else:
                    stats["skipped_dup"] += 1

            except Exception as e:
                stats["skipped_err"] += 1
                log.debug(f"Error importing {entry.get('id', '?')}: {e}")
                session.rollback()
                # Re-fetch project after rollback
                project = session.query(Project).filter_by(name=project_name).first()
                continue

        session.commit()
        log.info(f"Import complete: {stats}")

    finally:
        session.close()

    return stats


def import_euler2ai_pcfs(
    pcf_path: str | Path,
    project_name: str = "Euler2AI PCFs",
    atlas_db_path: str | Path | None = None,
) -> dict:
    """Import Euler2AI PCF JSON file."""
    engine = init_db(db_path=atlas_db_path)
    session = get_session(engine)

    stats = {"total": 0, "imported": 0, "skipped_dup": 0, "skipped_err": 0}

    try:
        project = session.query(Project).filter_by(name=project_name).first()
        if not project:
            project = Project(name=project_name)
            session.add(project)
            session.flush()

        with open(pcf_path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            entries = data.get("pcfs", data.get("entries", []))
        else:
            entries = data

        log.info(f"Importing {len(entries)} PCFs...")
        stats["total"] = len(entries)

        for entry in entries:
            try:
                parsed = parse_euler2ai_pcf(entry)

                series = Series(
                    project_id=project.id,
                    name=f"PCF_{parsed['metadata'].get('pcf_id', '')}",
                    definition=parsed["series_definition"],
                    generator_type="pcf",
                    provenance=parsed["provenance"],
                )
                session.add(series)
                session.flush()

                rep = _upsert_representation(
                    session, series,
                    "pcf",
                    parsed["payload"],
                    parsed["cmf_payload"],
                )
                if rep:
                    stats["imported"] += 1
                else:
                    stats["skipped_dup"] += 1

            except Exception as e:
                stats["skipped_err"] += 1
                log.debug(f"Error: {e}")
                session.rollback()
                project = session.query(Project).filter_by(name=project_name).first()
                continue

        session.commit()
        log.info(f"Import complete: {stats}")

    finally:
        session.close()

    return stats


def import_jsonl(
    jsonl_path: str | Path,
    project_name: str = "JSONL Import",
    atlas_db_path: str | Path | None = None,
) -> dict:
    """Import a generic JSONL file."""
    engine = init_db(db_path=atlas_db_path)
    session = get_session(engine)

    stats = {"total": 0, "imported": 0, "skipped_dup": 0, "skipped_err": 0}

    try:
        project = session.query(Project).filter_by(name=project_name).first()
        if not project:
            project = Project(name=project_name)
            session.add(project)
            session.flush()

        with open(jsonl_path) as f:
            for line in f:
                stats["total"] += 1
                parsed = parse_jsonl_line(line)
                if not parsed:
                    stats["skipped_err"] += 1
                    continue

                try:
                    series = Series(
                        project_id=project.id,
                        definition=parsed["series_definition"],
                        generator_type=parsed["generator_type"],
                        provenance=parsed["provenance"],
                    )
                    session.add(series)
                    session.flush()

                    rep = _upsert_representation(
                        session, series,
                        parsed["primary_group"],
                        parsed["payload"],
                        parsed.get("cmf_payload", {}),
                    )
                    if rep:
                        stats["imported"] += 1
                    else:
                        stats["skipped_dup"] += 1

                except Exception as e:
                    stats["skipped_err"] += 1
                    session.rollback()
                    project = session.query(Project).filter_by(name=project_name).first()

        session.commit()
        log.info(f"JSONL import complete: {stats}")

    finally:
        session.close()

    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m cmf_atlas.ingest.import_existing <cmf_database.json> [atlas.db]")
        sys.exit(1)

    src = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) > 2 else None

    if src.endswith(".jsonl"):
        stats = import_jsonl(src, atlas_db_path=dest)
    elif "pcf" in src.lower() or "euler" in src.lower():
        stats = import_euler2ai_pcfs(src, atlas_db_path=dest)
    else:
        stats = import_cmf_database(src, atlas_db_path=dest)

    print(json.dumps(stats, indent=2))
