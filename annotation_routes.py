"""
CMF Atlas Classification — Annotation API Routes
=================================================
Mounted at /annotate by api.py.

All write endpoints require a session cookie (token=...).
Auth is deliberately simple: single-tenant, invited accounts only.
"""

import csv
import io
import json
from typing import Any, Optional

from fastapi import APIRouter, Cookie, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import annotation_db as db

router = APIRouter(prefix="/annotate", tags=["annotate"])


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _require_user(
    cookie_token: Optional[str],
    header_token: Optional[str] = None,
) -> dict:
    """Accept token from cookie or X-Token header."""
    t = cookie_token or header_token
    if not t:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get_session_user(t)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return user


def _require_admin(cookie_token: Optional[str], header_token: Optional[str] = None) -> dict:
    user = _require_user(cookie_token, header_token)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    email: str
    display_name: str
    password: str
    role: str = "annotator"


class AnnotationRequest(BaseModel):
    id: Optional[str] = None
    cmf_id: int
    matrix_count: Optional[int] = None
    matrix_size_signature: Optional[str] = None
    visible_variables: Optional[list[str]] = None
    visible_variable_count: Optional[int] = None
    has_denominator: Optional[bool] = None
    has_variable_in_denominator: Optional[bool] = None
    highest_visible_exponent: Optional[int] = None
    highest_numerator_total_degree: Optional[int] = None
    highest_denominator_total_degree: Optional[int] = None
    notes: Optional[str] = None
    confidence: Optional[int] = None
    status: str = "draft"
    matrices: Optional[list[dict]] = None


class ComparisonRequest(BaseModel):
    id: Optional[str] = None
    cmf_a_id: int
    cmf_b_id: int
    same_matrix_count: Optional[bool] = None
    same_size_signature: Optional[bool] = None
    same_variable_count_profile: Optional[bool] = None
    same_denominator_profile: Optional[bool] = None
    same_variable_in_denominator_profile: Optional[bool] = None
    same_highest_exponent_profile: Optional[str] = None
    same_degree_profile: Optional[str] = None
    same_entry_type_profile: Optional[str] = None
    same_zero_entry_profile: Optional[str] = None
    surface_similarity_score: Optional[int] = None
    family_similarity_score: Optional[int] = None
    human_judgment: str
    rationale: Optional[str] = None
    confidence: Optional[int] = None


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@router.post("/login")
def login(req: LoginRequest):
    user = db.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = db.create_session(user["id"])
    resp = JSONResponse({"ok": True, "user": user, "token": token})
    resp.set_cookie("token", token, max_age=86400 * 30, httponly=True, samesite="lax")
    return resp


@router.post("/logout")
def logout(token: Optional[str] = Cookie(None)):
    if token:
        db.delete_session(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("token")
    return resp


@router.get("/me")
def get_me(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    user = _require_user(token, x_token)
    return user


@router.post("/users", summary="Admin: create a new user account")
def create_user(req: CreateUserRequest, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_admin(token, x_token)
    try:
        user = db.create_user(req.email, req.display_name, req.password, req.role)
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@router.get("/protocols")
def list_protocols(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    return db.list_protocols()


@router.get("/protocols/active")
def active_protocol(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    p = db.get_active_protocol()
    if not p:
        raise HTTPException(status_code=404, detail="No active protocol")
    return p


# ---------------------------------------------------------------------------
# Queue — list CMFs for annotation
# ---------------------------------------------------------------------------

@router.get("/queue")
def get_queue(
    token: Optional[str] = Cookie(None),
    x_token: Optional[str] = Header(None, alias="x-token"),
    matrix_size: Optional[int] = Query(None),
    status_filter: Optional[str] = Query(None, description="unlabeled|draft|submitted|reviewed"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    request: Request = None,
):
    """
    Returns CMFs from the main atlas database, annotated with annotation status.
    Filters by matrix_size bucket and annotation status.
    """
    _require_user(token, x_token)

    status_map = db.annotation_status_map()
    annotated_ids = set(status_map.keys())

    try:
        from sqlalchemy import create_engine, text
        import os
        from pathlib import Path
        _default_db = Path(__file__).parent / "data" / "atlas_2d.db"
        if not _default_db.exists():
            _default_db = Path(__file__).parent / "data" / "atlas.db"
        db_path = Path(os.getenv("CMF_ATLAS_DB", _default_db))
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

        with eng.connect() as con:
            clauses = ["c.dimension >= 2"]
            params: dict[str, Any] = {}

            if matrix_size is not None:
                clauses.append("CAST(json_extract(c.cmf_payload,'$.matrix_size') AS INTEGER) = :ms")
                params["ms"] = matrix_size

            where = "WHERE " + " AND ".join(clauses)

            total_row = con.execute(
                text(f"SELECT COUNT(*) FROM cmf c {where}"), params
            ).fetchone()
            total = total_row[0] if total_row else 0

            rows = con.execute(
                text(f"""
                    SELECT c.id,
                           json_extract(c.cmf_payload,'$.matrix_size') AS matrix_size,
                           json_extract(c.cmf_payload,'$.n_matrices') AS n_matrices,
                           c.dimension,
                           json_extract(c.cmf_payload,'$.primary_constant') AS primary_constant,
                           json_extract(c.cmf_payload,'$.certification_level') AS cert_level,
                           json_extract(c.cmf_payload,'$.source_category') AS source_category,
                           json_extract(c.cmf_payload,'$.degree') AS degree
                    FROM cmf c {where}
                    ORDER BY c.id
                    LIMIT :lim OFFSET :off
                """),
                {**params, "lim": limit * 3, "off": offset},
            ).fetchall()
    except Exception as e:
        return {"total": 0, "offset": offset, "limit": limit, "items": [], "error": str(e)}

    items = []
    for row in rows:
        cmf_id = row[0]
        ann_status = status_map.get(cmf_id, "unlabeled")
        if status_filter and status_filter != "all":
            if status_filter == "unlabeled" and ann_status != "unlabeled":
                continue
            elif status_filter not in ("unlabeled",) and ann_status != status_filter:
                continue
        items.append({
            "id": cmf_id,
            "matrix_size": row[1],
            "n_matrices": row[2],
            "dimension": row[3],
            "primary_constant": row[4],
            "cert_level": row[5],
            "source_category": row[6],
            "degree": row[7],
            "annotation_status": ann_status,
        })
        if len(items) >= limit:
            break

    return {"total": total, "offset": offset, "limit": limit, "items": items}


# ---------------------------------------------------------------------------
# Single CMF — fetch payload for annotation page
# ---------------------------------------------------------------------------

@router.get("/cmf/{cmf_id}")
def get_cmf(cmf_id: int, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    try:
        from sqlalchemy import create_engine, text
        import os
        from pathlib import Path
        _default_db = Path(__file__).parent / "data" / "atlas_2d.db"
        if not _default_db.exists():
            _default_db = Path(__file__).parent / "data" / "atlas.db"
        db_path = Path(os.getenv("CMF_ATLAS_DB", _default_db))
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        with eng.connect() as con:
            row = con.execute(
                text("SELECT c.id, c.cmf_payload, c.dimension, r.canonical_fingerprint "
                     "FROM cmf c LEFT JOIN representation r ON r.id=c.representation_id "
                     "WHERE c.id=:cid"),
                {"cid": cmf_id},
            ).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(status_code=404, detail=f"CMF {cmf_id} not found")

    payload = {}
    try:
        payload = json.loads(row[1]) if row[1] else {}
    except Exception:
        pass

    existing_annotation = db.get_annotation_for_cmf(cmf_id)

    return {
        "id": row[0],
        "dimension": row[2],
        "fingerprint": row[3],
        "payload": payload,
        "annotation": existing_annotation,
    }


# ---------------------------------------------------------------------------
# Annotations CRUD
# ---------------------------------------------------------------------------

@router.post("/annotation")
def save_annotation(req: AnnotationRequest, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    user = _require_user(token, x_token)
    data = req.model_dump()
    data["annotator_id"] = user["id"]
    ann_id = db.upsert_annotation(data)
    return {"ok": True, "id": ann_id}


@router.get("/annotation/by-cmf/{cmf_id}")
def get_annotation_for_cmf(cmf_id: int, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    user = _require_user(token, x_token)
    ann = db.get_annotation_for_cmf(cmf_id, user["id"])
    return ann or {}


@router.get("/annotation/{ann_id}")
def get_annotation(ann_id: str, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    ann = db.get_annotation(ann_id)
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann


@router.get("/annotations")
def list_annotations(
    token: Optional[str] = Cookie(None),
    x_token: Optional[str] = Header(None, alias="x-token"),
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    user = _require_user(token, x_token)
    annotator_id = user["id"] if user["role"] != "admin" else None
    total, items = db.list_annotations(annotator_id=annotator_id, status=status, limit=limit, offset=offset)
    return {"total": total, "offset": offset, "limit": limit, "items": items}


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

@router.post("/comparison")
def save_comparison(req: ComparisonRequest, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    user = _require_user(token, x_token)
    data = req.model_dump()
    data["annotator_id"] = user["id"]
    cmp_id = db.upsert_comparison(data)
    return {"ok": True, "id": cmp_id}


@router.get("/comparisons")
def list_comparisons(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    user = _require_user(token, x_token)
    annotator_id = user["id"] if user["role"] != "admin" else None
    return db.list_comparisons(annotator_id=annotator_id)


@router.post("/similarity")
def compute_similarity(body: dict, token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    """Compute auto-similarity between two annotation objects (sent in body)."""
    _require_user(token, x_token)
    ann_a = body.get("ann_a", {})
    ann_b = body.get("ann_b", {})
    return db.compute_similarity(ann_a, ann_b)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_stats(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    return db.annotation_stats()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@router.get("/export/json")
def export_json(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    data = db.export_json()
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=cmf_annotations.json"},
    )


@router.get("/export/csv")
def export_csv(token: Optional[str] = Cookie(None), x_token: Optional[str] = Header(None, alias="x-token")):
    _require_user(token, x_token)
    headers, rows = db.export_csv_rows()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.read().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cmf_annotations.csv"},
    )


# ---------------------------------------------------------------------------
# Admin bootstrap — create first admin user via env vars
# ---------------------------------------------------------------------------

@router.post("/bootstrap", include_in_schema=False)
def bootstrap(body: dict):
    """
    One-time admin setup. Requires BOOTSTRAP_SECRET env var to match body.secret.
    POST /annotate/bootstrap  {"secret":"...", "email":"...", "password":"...", "name":"..."}
    """
    import os
    expected = os.getenv("BOOTSTRAP_SECRET", "")
    if not expected or body.get("secret") != expected:
        raise HTTPException(status_code=403, detail="Invalid bootstrap secret")
    try:
        user = db.create_user(
            body["email"], body.get("name", "Admin"), body["password"], role="admin"
        )
        return {"ok": True, "user": user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
