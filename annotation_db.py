"""
CMF Atlas Classification — Annotation Database
===============================================
Standalone SQLite store for the annotation system.
Lives at data/annotation.db (separate from atlas.db so it can be cleared
without touching the CMF data).

Tables:
  users, protocol_versions,
  cmf_annotations, matrix_annotations,
  cmf_comparisons, family_hypotheses
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ANNOTATION_DB = Path(os.getenv("ANNOTATION_DB", Path(__file__).parent / "data" / "annotation.db"))

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    email        TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'annotator',
    password_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS protocol_versions (
    id           TEXT PRIMARY KEY,
    version_name TEXT NOT NULL UNIQUE,
    description  TEXT,
    is_active    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cmf_annotations (
    id                              TEXT PRIMARY KEY,
    cmf_id                          INTEGER NOT NULL,
    annotator_id                    TEXT NOT NULL REFERENCES users(id),
    protocol_version_id             TEXT NOT NULL REFERENCES protocol_versions(id),

    visible_variables               TEXT,
    visible_variable_count          INTEGER,
    has_denominator                 INTEGER,
    has_variable_in_denominator     INTEGER,
    highest_visible_exponent        INTEGER,
    highest_numerator_total_degree  INTEGER,
    highest_denominator_total_degree INTEGER,
    matrix_count                    INTEGER,
    matrix_size_signature           TEXT,

    system_symmetry_badge           TEXT,
    notes                           TEXT,
    confidence                      INTEGER CHECK (confidence BETWEEN 1 AND 5),
    status                          TEXT NOT NULL DEFAULT 'draft',
    created_at                      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at                      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ann_cmf ON cmf_annotations(cmf_id);
CREATE INDEX IF NOT EXISTS idx_ann_status ON cmf_annotations(status);
CREATE INDEX IF NOT EXISTS idx_ann_annotator ON cmf_annotations(annotator_id);

CREATE TABLE IF NOT EXISTS matrix_annotations (
    id                              TEXT PRIMARY KEY,
    cmf_annotation_id               TEXT NOT NULL REFERENCES cmf_annotations(id) ON DELETE CASCADE,
    matrix_label                    TEXT NOT NULL,
    matrix_index                    INTEGER NOT NULL,

    rows_count                      INTEGER,
    cols_count                      INTEGER,
    visible_variables               TEXT,
    visible_variable_count          INTEGER,
    has_denominator                 INTEGER,
    has_variable_in_denominator     INTEGER,
    highest_visible_exponent        INTEGER,
    highest_numerator_total_degree  INTEGER,
    highest_denominator_total_degree INTEGER,
    entry_type_profile              TEXT,
    contains_zero_entries           INTEGER,
    symmetry_impression             TEXT,
    notes                           TEXT,
    confidence                      INTEGER CHECK (confidence BETWEEN 1 AND 5),

    UNIQUE (cmf_annotation_id, matrix_label)
);

CREATE TABLE IF NOT EXISTS cmf_comparisons (
    id                                  TEXT PRIMARY KEY,
    cmf_a_id                            INTEGER NOT NULL,
    cmf_b_id                            INTEGER NOT NULL,
    annotator_id                        TEXT NOT NULL REFERENCES users(id),
    protocol_version_id                 TEXT NOT NULL REFERENCES protocol_versions(id),

    same_matrix_count                   INTEGER,
    same_size_signature                 INTEGER,
    same_variable_count_profile         INTEGER,
    same_denominator_profile            INTEGER,
    same_variable_in_denominator_profile INTEGER,
    same_highest_exponent_profile       TEXT,
    same_degree_profile                 TEXT,
    same_entry_type_profile             TEXT,
    same_zero_entry_profile             TEXT,

    surface_similarity_score            INTEGER,
    family_similarity_score             INTEGER,

    human_judgment                      TEXT,
    rationale                           TEXT,
    confidence                          INTEGER CHECK (confidence BETWEEN 1 AND 5),

    created_at                          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cmp_ab ON cmf_comparisons(cmf_a_id, cmf_b_id);

CREATE TABLE IF NOT EXISTS family_hypotheses (
    id           TEXT PRIMARY KEY,
    family_label TEXT NOT NULL,
    cmf_id       INTEGER NOT NULL,
    source       TEXT NOT NULL DEFAULT 'manual',
    evidence     TEXT,
    confidence   INTEGER CHECK (confidence BETWEEN 1 AND 5),
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SEED_SQL = """
INSERT OR IGNORE INTO protocol_versions (id, version_name, description, is_active)
VALUES ('proto_v01', 'CMF-Struct-v0.1',
        'First structural annotation protocol. Surface-level visible properties only.',
        1);
"""


def _conn() -> sqlite3.Connection:
    ANNOTATION_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ANNOTATION_DB), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


MIGRATIONS = [
    "ALTER TABLE cmf_annotations ADD COLUMN system_symmetry_badge TEXT",
]


def init_db():
    """Create tables, seed default data, and run migrations."""
    with _conn() as con:
        con.executescript(SCHEMA_SQL)
        con.executescript(SEED_SQL)
        for sql in MIGRATIONS:
            try:
                con.execute(sql)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(email: str, display_name: str, password: str, role: str = "annotator") -> dict:
    uid = "u_" + secrets.token_hex(8)
    with _conn() as con:
        con.execute(
            "INSERT INTO users (id, email, display_name, role, password_hash) VALUES (?,?,?,?,?)",
            (uid, email, display_name, role, _hash_pw(password)),
        )
    return {"id": uid, "email": email, "display_name": display_name, "role": role}


def authenticate(email: str, password: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT id, display_name, role FROM users WHERE email=? AND password_hash=?",
            (email, _hash_pw(password)),
        ).fetchone()
    if not row:
        return None
    return dict(row)


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # sessions valid for 30 days
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,datetime('now','+30 days'))",
            (token, user_id),
        )
    return token


def get_session_user(token: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            """SELECT u.id, u.display_name, u.role FROM sessions s
               JOIN users u ON u.id = s.user_id
               WHERE s.token=? AND s.expires_at > datetime('now')""",
            (token,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str):
    with _conn() as con:
        con.execute("DELETE FROM sessions WHERE token=?", (token,))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def get_active_protocol() -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM protocol_versions WHERE is_active=1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def list_protocols() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM protocol_versions ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Annotations (CRUD)
# ---------------------------------------------------------------------------

def upsert_annotation(data: dict) -> str:
    """Insert or replace a cmf_annotation record. Returns annotation id."""
    now = datetime.now(timezone.utc).isoformat()
    proto = get_active_protocol()
    proto_id = proto["id"] if proto else "proto_v01"

    # Resolve annotation id: explicit > existing by (cmf_id, annotator_id) > new
    ann_id = data.get("id")
    if not ann_id:
        with _conn() as con:
            existing_row = con.execute(
                "SELECT id FROM cmf_annotations WHERE cmf_id=? AND annotator_id=? "
                "ORDER BY updated_at DESC LIMIT 1",
                (data["cmf_id"], data["annotator_id"]),
            ).fetchone()
        ann_id = existing_row[0] if existing_row else ("ann_" + secrets.token_hex(10))

    fields = [
        "id", "cmf_id", "annotator_id", "protocol_version_id",
        "visible_variables", "visible_variable_count",
        "has_denominator", "has_variable_in_denominator",
        "highest_visible_exponent", "highest_numerator_total_degree",
        "highest_denominator_total_degree", "matrix_count", "matrix_size_signature",
        "system_symmetry_badge", "notes", "confidence", "status", "updated_at",
    ]
    vals = (
        ann_id,
        data["cmf_id"],
        data["annotator_id"],
        data.get("protocol_version_id", proto_id),
        _json_list(data.get("visible_variables")),
        data.get("visible_variable_count"),
        _bool_int(data.get("has_denominator")),
        _bool_int(data.get("has_variable_in_denominator")),
        data.get("highest_visible_exponent"),
        data.get("highest_numerator_total_degree"),
        data.get("highest_denominator_total_degree"),
        data.get("matrix_count"),
        data.get("matrix_size_signature"),
        data.get("system_symmetry_badge"),
        data.get("notes"),
        data.get("confidence"),
        data.get("status", "draft"),
        now,
    )

    with _conn() as con:
        existing = con.execute("SELECT id FROM cmf_annotations WHERE id=?", (ann_id,)).fetchone()
        if existing:
            set_clause = ", ".join(f"{f}=?" for f in fields[1:])
            con.execute(f"UPDATE cmf_annotations SET {set_clause} WHERE id=?", vals[1:] + (ann_id,))
        else:
            placeholders = ", ".join("?" * len(fields))
            con.execute(f"INSERT INTO cmf_annotations ({', '.join(fields)}) VALUES ({placeholders})", vals)

        if "matrices" in data:
            con.execute("DELETE FROM matrix_annotations WHERE cmf_annotation_id=?", (ann_id,))
            for i, mat in enumerate(data["matrices"]):
                mid = "mat_" + secrets.token_hex(8)
                con.execute(
                    """INSERT INTO matrix_annotations
                       (id, cmf_annotation_id, matrix_label, matrix_index,
                        rows_count, cols_count, visible_variables, visible_variable_count,
                        has_denominator, has_variable_in_denominator,
                        highest_visible_exponent, highest_numerator_total_degree,
                        highest_denominator_total_degree, entry_type_profile,
                        contains_zero_entries, symmetry_impression, notes, confidence)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mid, ann_id, mat.get("matrix_label", f"K{i+1}"), i,
                        mat.get("rows_count"), mat.get("cols_count"),
                        _json_list(mat.get("visible_variables")), mat.get("visible_variable_count"),
                        _bool_int(mat.get("has_denominator")),
                        _bool_int(mat.get("has_variable_in_denominator")),
                        mat.get("highest_visible_exponent"),
                        mat.get("highest_numerator_total_degree"),
                        mat.get("highest_denominator_total_degree"),
                        mat.get("entry_type_profile"),
                        _bool_int(mat.get("contains_zero_entries")),
                        mat.get("symmetry_impression"),
                        mat.get("notes"),
                        mat.get("confidence"),
                    ),
                )
    return ann_id


def get_annotation(ann_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM cmf_annotations WHERE id=?", (ann_id,)).fetchone()
        if not row:
            return None
        ann = dict(row)
        mats = con.execute(
            "SELECT * FROM matrix_annotations WHERE cmf_annotation_id=? ORDER BY matrix_index",
            (ann_id,),
        ).fetchall()
        ann["matrices"] = [dict(m) for m in mats]
    ann["visible_variables"] = _parse_json_list(ann.get("visible_variables"))
    for m in ann["matrices"]:
        m["visible_variables"] = _parse_json_list(m.get("visible_variables"))
    return ann


def get_annotation_for_cmf(cmf_id: int, annotator_id: str | None = None) -> dict | None:
    with _conn() as con:
        q = "SELECT * FROM cmf_annotations WHERE cmf_id=?"
        args: list = [cmf_id]
        if annotator_id:
            q += " AND annotator_id=?"
            args.append(annotator_id)
        q += " ORDER BY updated_at DESC LIMIT 1"
        row = con.execute(q, args).fetchone()
        if not row:
            return None
        ann = dict(row)
        mats = con.execute(
            "SELECT * FROM matrix_annotations WHERE cmf_annotation_id=? ORDER BY matrix_index",
            (ann["id"],),
        ).fetchall()
        ann["matrices"] = [dict(m) for m in mats]
    ann["visible_variables"] = _parse_json_list(ann.get("visible_variables"))
    for m in ann["matrices"]:
        m["visible_variables"] = _parse_json_list(m.get("visible_variables"))
    return ann


def list_annotations(
    annotator_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    with _conn() as con:
        clauses = []
        args: list = []
        if annotator_id:
            clauses.append("a.annotator_id=?"); args.append(annotator_id)
        if status:
            clauses.append("a.status=?"); args.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        total = con.execute(f"SELECT COUNT(*) FROM cmf_annotations a {where}", args).fetchone()[0]
        rows = con.execute(
            f"SELECT a.*, u.display_name AS annotator_name FROM cmf_annotations a "
            f"LEFT JOIN users u ON u.id=a.annotator_id "
            f"{where} ORDER BY a.updated_at DESC LIMIT ? OFFSET ?",
            args + [limit, offset],
        ).fetchall()
    return total, [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Queue helpers (which CMF IDs need annotation)
# ---------------------------------------------------------------------------

def annotated_cmf_ids(annotator_id: str | None = None) -> set[int]:
    with _conn() as con:
        q = "SELECT DISTINCT cmf_id FROM cmf_annotations"
        args: list = []
        if annotator_id:
            q += " WHERE annotator_id=?"; args.append(annotator_id)
        rows = con.execute(q, args).fetchall()
    return {r[0] for r in rows}


def annotation_status_map() -> dict[int, str]:
    """Returns {cmf_id: best_status} for all annotated CMFs."""
    with _conn() as con:
        rows = con.execute(
            "SELECT cmf_id, MAX(CASE status "
            "WHEN 'reviewed' THEN 3 WHEN 'submitted' THEN 2 WHEN 'draft' THEN 1 ELSE 0 END) AS rank, "
            "status FROM cmf_annotations GROUP BY cmf_id"
        ).fetchall()
    return {r["cmf_id"]: r["status"] for r in rows}


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def upsert_comparison(data: dict) -> str:
    cmp_id = data.get("id") or ("cmp_" + secrets.token_hex(8))
    proto = get_active_protocol()
    proto_id = proto["id"] if proto else "proto_v01"

    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO cmf_comparisons
               (id, cmf_a_id, cmf_b_id, annotator_id, protocol_version_id,
                same_matrix_count, same_size_signature, same_variable_count_profile,
                same_denominator_profile, same_variable_in_denominator_profile,
                same_highest_exponent_profile, same_degree_profile,
                same_entry_type_profile, same_zero_entry_profile,
                surface_similarity_score, family_similarity_score,
                human_judgment, rationale, confidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                cmp_id, data["cmf_a_id"], data["cmf_b_id"],
                data["annotator_id"], data.get("protocol_version_id", proto_id),
                _bool_int(data.get("same_matrix_count")),
                _bool_int(data.get("same_size_signature")),
                _bool_int(data.get("same_variable_count_profile")),
                _bool_int(data.get("same_denominator_profile")),
                _bool_int(data.get("same_variable_in_denominator_profile")),
                data.get("same_highest_exponent_profile"),
                data.get("same_degree_profile"),
                data.get("same_entry_type_profile"),
                data.get("same_zero_entry_profile"),
                data.get("surface_similarity_score"),
                data.get("family_similarity_score"),
                data.get("human_judgment"),
                data.get("rationale"),
                data.get("confidence"),
            ),
        )
    return cmp_id


def list_comparisons(annotator_id: str | None = None, limit: int = 200) -> list[dict]:
    with _conn() as con:
        q = "SELECT * FROM cmf_comparisons"
        args: list = []
        if annotator_id:
            q += " WHERE annotator_id=?"; args.append(annotator_id)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = con.execute(q, args).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def annotation_stats() -> dict:
    with _conn() as con:
        total       = con.execute("SELECT COUNT(*) FROM cmf_annotations").fetchone()[0]
        draft       = con.execute("SELECT COUNT(*) FROM cmf_annotations WHERE status='draft'").fetchone()[0]
        submitted   = con.execute("SELECT COUNT(*) FROM cmf_annotations WHERE status='submitted'").fetchone()[0]
        reviewed    = con.execute("SELECT COUNT(*) FROM cmf_annotations WHERE status='reviewed'").fetchone()[0]
        comparisons = con.execute("SELECT COUNT(*) FROM cmf_comparisons").fetchone()[0]
        annotators  = con.execute("SELECT COUNT(DISTINCT annotator_id) FROM cmf_annotations").fetchone()[0]
    return {
        "total": total, "draft": draft, "submitted": submitted,
        "reviewed": reviewed, "comparisons": comparisons, "annotators": annotators,
    }


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

def compute_similarity(ann_a: dict, ann_b: dict) -> dict:
    """Compute surface and family similarity scores from two annotations."""
    def _get(d, k, default=None):
        return d.get(k, default)

    checks: dict[str, bool | None] = {}
    checks["same_matrix_count"] = (
        _get(ann_a, "matrix_count") == _get(ann_b, "matrix_count")
        if None not in (_get(ann_a, "matrix_count"), _get(ann_b, "matrix_count"))
        else None
    )
    checks["same_size_signature"] = (
        _get(ann_a, "matrix_size_signature") == _get(ann_b, "matrix_size_signature")
        if None not in (_get(ann_a, "matrix_size_signature"), _get(ann_b, "matrix_size_signature"))
        else None
    )
    checks["same_denominator_profile"] = (
        _get(ann_a, "has_denominator") == _get(ann_b, "has_denominator")
    )
    checks["same_variable_in_denominator_profile"] = (
        _get(ann_a, "has_variable_in_denominator") == _get(ann_b, "has_variable_in_denominator")
    )
    checks["same_variable_count_profile"] = (
        _get(ann_a, "visible_variable_count") == _get(ann_b, "visible_variable_count")
    )
    checks["same_highest_exponent_profile"] = (
        "yes" if _get(ann_a, "highest_visible_exponent") == _get(ann_b, "highest_visible_exponent")
        else ("partial" if None not in (_get(ann_a, "highest_visible_exponent"), _get(ann_b, "highest_visible_exponent")) else "unknown")
    )
    checks["same_degree_profile"] = (
        "yes" if (
            _get(ann_a, "highest_numerator_total_degree") == _get(ann_b, "highest_numerator_total_degree") and
            _get(ann_a, "highest_denominator_total_degree") == _get(ann_b, "highest_denominator_total_degree")
        ) else "partial"
    )

    # Surface score (0-100)
    surface = 0
    if checks.get("same_matrix_count"):          surface += 20
    if checks.get("same_size_signature"):        surface += 20
    if checks.get("same_variable_count_profile"): surface += 10
    va_a = set(_get(ann_a, "visible_variables") or [])
    va_b = set(_get(ann_b, "visible_variables") or [])
    if va_a and va_b and va_a == va_b:           surface += 10
    if checks.get("same_denominator_profile"):   surface += 10
    if checks.get("same_variable_in_denominator_profile"): surface += 10
    if checks.get("same_highest_exponent_profile") == "yes": surface += 10
    if checks.get("same_degree_profile") == "yes": surface += 10

    # Family score (0-100, variable names ignored)
    family = 0
    if checks.get("same_matrix_count"):          family += 20
    if checks.get("same_size_signature"):        family += 20
    if checks.get("same_degree_profile") == "yes": family += 15
    if (checks.get("same_denominator_profile") and
            checks.get("same_variable_in_denominator_profile")): family += 15
    if checks.get("same_variable_count_profile"): family += 10

    return {
        **checks,
        "surface_similarity_score": surface,
        "family_similarity_score": family,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_json() -> list[dict]:
    import json as _json
    with _conn() as con:
        rows = con.execute(
            "SELECT a.*, u.display_name AS annotator_name "
            "FROM cmf_annotations a LEFT JOIN users u ON u.id=a.annotator_id "
            "ORDER BY a.updated_at DESC"
        ).fetchall()
        result = []
        for row in rows:
            ann = dict(row)
            mats = con.execute(
                "SELECT * FROM matrix_annotations WHERE cmf_annotation_id=? ORDER BY matrix_index",
                (ann["id"],),
            ).fetchall()
            ann["matrices"] = [dict(m) for m in mats]
            ann["visible_variables"] = _parse_json_list(ann.get("visible_variables"))
            for m in ann["matrices"]:
                m["visible_variables"] = _parse_json_list(m.get("visible_variables"))
            result.append(ann)
    return result


def export_csv_rows() -> tuple[list[str], list[list]]:
    headers = [
        "id", "cmf_id", "annotator_name", "status", "protocol_version_id",
        "matrix_count", "matrix_size_signature", "visible_variables", "visible_variable_count",
        "has_denominator", "has_variable_in_denominator",
        "highest_visible_exponent", "highest_numerator_total_degree",
        "highest_denominator_total_degree", "confidence", "updated_at",
    ]
    with _conn() as con:
        rows = con.execute(
            "SELECT a.id, a.cmf_id, u.display_name, a.status, a.protocol_version_id, "
            "a.matrix_count, a.matrix_size_signature, a.visible_variables, a.visible_variable_count, "
            "a.has_denominator, a.has_variable_in_denominator, "
            "a.highest_visible_exponent, a.highest_numerator_total_degree, "
            "a.highest_denominator_total_degree, a.confidence, a.updated_at "
            "FROM cmf_annotations a LEFT JOIN users u ON u.id=a.annotator_id "
            "ORDER BY a.updated_at DESC"
        ).fetchall()
    return headers, [list(r) for r in rows]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bool_int(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return 1 if v.lower() in ("true", "yes", "1") else 0
    return None


def _json_list(v) -> str | None:
    import json as _json
    if v is None:
        return None
    if isinstance(v, list):
        return _json.dumps(v)
    return v


def _parse_json_list(v) -> list:
    import json as _json
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        parsed = _json.loads(v)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [s.strip() for s in v.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Bootstrap on import
# ---------------------------------------------------------------------------

init_db()
