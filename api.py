"""
CMF Atlas — Read-only FastAPI
============================
Serves the SQLite atlas.db over HTTP with filtering, pagination, and search.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Env vars:
    CMF_ATLAS_DB  — path to atlas.db  (default: data/atlas.db)
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union
import math
import mpmath
import sympy as _sp
from sympy import symbols as _sym, sympify as _sympify, lambdify as _lambdify, expand as _expand
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# DB setup (read-only, single connection)
# ---------------------------------------------------------------------------

_default_db = Path(__file__).parent / "data" / "atlas_2d.db"
if not _default_db.exists():
    _default_db = Path(__file__).parent / "data" / "atlas.db"
DB_PATH = Path(os.getenv("CMF_ATLAS_DB", _default_db))

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class ProjectOut(BaseModel):
    id: int
    name: str
    created_at: Optional[str] = None
    series_count: Optional[int] = None

class SeriesOut(BaseModel):
    id: int
    project_id: int
    name: Optional[str] = None
    definition: Optional[str] = None
    generator_type: str
    provenance: Optional[str] = None
    created_at: Optional[str] = None
    representation_count: Optional[int] = None

class RepresentationOut(BaseModel):
    id: int
    series_id: int
    primary_group: str
    canonical_fingerprint: str
    canonical_payload: Optional[Any] = None
    overlap_groups: Optional[Any] = None
    created_at: Optional[str] = None

class FeaturesOut(BaseModel):
    representation_id: int
    feature_json: Optional[Any] = None
    feature_version: str
    computed_at: Optional[str] = None

class CMFOut(BaseModel):
    id: int
    representation_id: int
    cmf_payload: Optional[Any] = None
    dimension: Optional[int] = None
    direction_policy: Optional[str] = None
    created_at: Optional[str] = None

class EvalRunOut(BaseModel):
    id: int
    cmf_id: int
    run_type: str
    precision_digits: int
    steps: Optional[int] = None
    limit_estimate: Optional[str] = None
    error_estimate: Optional[float] = None
    convergence_score: Optional[float] = None
    stability_score: Optional[float] = None
    runtime_ms: Optional[int] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None

class RecognitionOut(BaseModel):
    id: int
    eval_run_id: int
    method: str
    basis_name: Optional[str] = None
    basis_payload: Optional[Any] = None
    success: bool
    identified_as: Optional[str] = None
    relation_height: Optional[float] = None
    residual_log10: Optional[float] = None
    attempt_log: Optional[str] = None
    created_at: Optional[str] = None

class StatsOut(BaseModel):
    projects: int
    series: int
    representations: int
    cmfs: int
    eval_runs: int
    recognition_attempts: int
    recognition_successes: int
    generator_types: dict
    primary_groups: dict
    run_types: dict

class Paginated(BaseModel):
    total: int
    offset: int
    limit: int
    items: list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(raw: Optional[str]):
    """Try to parse a JSON string; return raw if it fails."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _str(dt) -> Optional[str]:
    return str(dt) if dt else None


# ---------------------------------------------------------------------------
# CMF Atlas math helpers
# ---------------------------------------------------------------------------

KNOWN_CONSTANTS: dict[str, float] = {
    "\u03c0":              math.pi,
    "e":               math.e,
    "ln(2)":           math.log(2),
    "\u03b6(3)": 1.20205690315959,
    "\u03b6(5)": 1.03692775514336,
    "\u03b6(2) = \u03c0\u00b2/6": math.pi**2 / 6,
    "G (Catalan)": 0.91596559417721,
    "4/\u03c0":             4 / math.pi,
    "1/\u03c0":             1 / math.pi,
    "\u03c0\u00b2":              math.pi**2,
    "ln(3)":           math.log(3),
    "\u221a2":              math.sqrt(2),
    "\u03c6 = (1+\u221a5)/2": (1 + math.sqrt(5)) / 2,
    "\u221a3":              math.sqrt(3),
    "1":               1.0,
    "2":               2.0,
    "1/2":             0.5,
}


def _compare_constants(value: float) -> list:
    """Compare value against known constants; return top 6 matches by abs error."""
    if not math.isfinite(value):
        return []
    results = []
    for name, c in KNOWN_CONSTANTS.items():
        if c == 0:
            continue
        abs_err = abs(value - c)
        rel_err = abs_err / abs(c)
        results.append({
            "name": name,
            "constant": round(c, 10),
            "abs_error": abs_err,
            "rel_error": rel_err,
            "log10_abs_error": round(math.log10(abs_err), 2) if abs_err > 0 else -15,
        })
    results.sort(key=lambda x: x["abs_error"])
    return results[:6]


def _build_walk_fns(f_poly_str: str, fbar_poly_str: str):
    """Build K1, K2 numerical matrix functions from CMF polynomial strings.

    Convention (verify_truly_2d.py):
        g(k,m)    = f(k,m)
        gbar(k,m) = fbar(k,m)
        b(k)      = g(k,0) * gbar(k,0)
        a(k,m)    = g(k,m) - gbar(k+1, m)
        K1(k,m)   = [[0, 1], [b(k+1), a(k,m)]]
        K2(k,m)   = [[gbar(k,m), 1], [b(k), g(k,m)]]
    """
    k_s, m_s, x_s, y_s = _sym("k m x y")
    f_expr    = _sympify(f_poly_str)
    fbar_expr = _sympify(fbar_poly_str)

    g_km    = f_expr.subs([(x_s, k_s), (y_s, m_s)])
    gbar_km = fbar_expr.subs([(x_s, k_s), (y_s, m_s)])
    b_expr  = _expand(g_km.subs(m_s, 0) * gbar_km.subs(m_s, 0))
    a_expr  = _expand(g_km - gbar_km.subs(k_s, k_s + 1))

    g_fn    = _lambdify([k_s, m_s], g_km,    modules="mpmath")
    gbar_fn = _lambdify([k_s, m_s], gbar_km, modules="mpmath")
    b_fn    = _lambdify([k_s],       b_expr,  modules="mpmath")
    a_fn    = _lambdify([k_s, m_s], a_expr,  modules="mpmath")

    def K1(k, m):
        return mpmath.matrix([[0, 1], [b_fn(k + 1), a_fn(k, m)]])

    def K2(k, m):
        return mpmath.matrix([[gbar_fn(k, m), 1], [b_fn(k), g_fn(k, m)]])

    return K1, K2


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}")
    yield

app = FastAPI(
    title="CMF Atlas API",
    version="1.0.0",
    description="Read-only API for the Conservative Matrix Field atlas database.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://davidvesterlund.com",
        "https://www.davidvesterlund.com",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"], include_in_schema=False)
def root():
    """Redirect to frontend."""
    return RedirectResponse(url="/app/index.html")


@app.get("/api", tags=["meta"])
def api_info():
    """API info."""
    return {
        "name": "CMF Atlas API",
        "version": "1.0.0",
        "docs": "/docs",
        "db": str(DB_PATH),
    }


@app.get("/stats", response_model=StatsOut, tags=["meta"])
def stats():
    """Aggregate statistics about the atlas."""
    db: Session = next(get_db())
    try:
        projects = db.execute(text("SELECT COUNT(*) FROM project")).scalar()
        series = db.execute(text("SELECT COUNT(*) FROM series")).scalar()
        representations = db.execute(text("SELECT COUNT(*) FROM representation")).scalar()
        cmfs = db.execute(text("SELECT COUNT(*) FROM cmf")).scalar()
        eval_runs = db.execute(text("SELECT COUNT(*) FROM eval_run")).scalar()
        rec_total = db.execute(text("SELECT COUNT(*) FROM recognition_attempt")).scalar()
        rec_success = db.execute(text("SELECT COUNT(*) FROM recognition_attempt WHERE success = 1")).scalar()

        gen_rows = db.execute(text("SELECT generator_type, COUNT(*) FROM series GROUP BY generator_type")).fetchall()
        pg_rows = db.execute(text("SELECT primary_group, COUNT(*) FROM representation GROUP BY primary_group")).fetchall()
        rt_rows = db.execute(text("SELECT run_type, COUNT(*) FROM eval_run GROUP BY run_type")).fetchall()

        return StatsOut(
            projects=projects or 0,
            series=series or 0,
            representations=representations or 0,
            cmfs=cmfs or 0,
            eval_runs=eval_runs or 0,
            recognition_attempts=rec_total or 0,
            recognition_successes=rec_success or 0,
            generator_types={r[0]: r[1] for r in gen_rows},
            primary_groups={r[0]: r[1] for r in pg_rows},
            run_types={r[0]: r[1] for r in rt_rows},
        )
    finally:
        db.close()


@app.get("/stats/detailed", tags=["meta"])
def stats_detailed():
    """Detailed breakdown for the dashboard: dimension, degree, certification, constants."""
    db: Session = next(get_db())
    try:
        dims = db.execute(text(
            "SELECT dimension, COUNT(*) FROM cmf GROUP BY dimension ORDER BY dimension"
        )).fetchall()

        certs: dict = {}
        sources: dict = {}
        degrees: dict = {}
        constants: dict = {}

        rows = db.execute(text("SELECT cmf_payload FROM cmf")).fetchall()
        for (p,) in rows:
            d = json.loads(p) if p else {}
            cl  = d.get("certification_level") or "unknown"
            src = d.get("source") or "unknown"
            deg = str(d.get("degree", 0))
            pc  = d.get("primary_constant")
            certs[cl]   = certs.get(cl, 0) + 1
            sources[src]= sources.get(src, 0) + 1
            degrees[deg]= degrees.get(deg, 0) + 1
            if pc:
                constants[pc] = constants.get(pc, 0) + 1

        walkable = db.execute(text(
            "SELECT COUNT(*) FROM cmf WHERE "
            "json_extract(cmf_payload,'$.f_poly') IS NOT NULL AND "
            "json_extract(cmf_payload,'$.f_poly') != ''"
        )).scalar() or 0

        return {
            "total_cmfs": sum(v for _, v in dims),
            "by_dimension": {str(d): c for d, c in dims},
            "by_certification": certs,
            "by_source": dict(sorted(sources.items(), key=lambda x: -x[1])[:15]),
            "by_degree": dict(sorted(degrees.items(), key=lambda x: int(x[0])))  ,
            "known_constants": dict(sorted(constants.items(), key=lambda x: -x[1])[:20]),
            "walkable_cmfs": walkable,
        }
    finally:
        db.close()


# --- Projects ---

@app.get("/projects", tags=["projects"])
def list_projects(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    db: Session = next(get_db())
    try:
        total = db.execute(text("SELECT COUNT(*) FROM project")).scalar()
        rows = db.execute(
            text("""
                SELECT p.id, p.name, p.created_at,
                       (SELECT COUNT(*) FROM series s WHERE s.project_id = p.id) AS series_count
                FROM project p
                ORDER BY p.id
                LIMIT :limit OFFSET :offset
            """),
            {"limit": limit, "offset": offset},
        ).fetchall()
        items = [ProjectOut(id=r[0], name=r[1], created_at=_str(r[2]), series_count=r[3]) for r in rows]
        return Paginated(total=total or 0, offset=offset, limit=limit, items=items)
    finally:
        db.close()


@app.get("/projects/{project_id}", tags=["projects"])
def get_project(project_id: int):
    db: Session = next(get_db())
    try:
        row = db.execute(text("SELECT id, name, created_at FROM project WHERE id = :id"), {"id": project_id}).fetchone()
        if not row:
            raise HTTPException(404, "Project not found")
        series_rows = db.execute(
            text("SELECT id, name, generator_type, provenance FROM series WHERE project_id = :pid ORDER BY id"),
            {"pid": project_id},
        ).fetchall()
        return {
            "id": row[0], "name": row[1], "created_at": _str(row[2]),
            "series": [{"id": s[0], "name": s[1], "generator_type": s[2], "provenance": s[3]} for s in series_rows],
        }
    finally:
        db.close()


# --- Series ---

@app.get("/series", tags=["series"])
def list_series(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    generator_type: Optional[str] = None,
    project_id: Optional[int] = None,
    q: Optional[str] = None,
):
    db: Session = next(get_db())
    try:
        where_clauses = []
        params: dict = {"limit": limit, "offset": offset}

        if generator_type:
            where_clauses.append("s.generator_type = :gen_type")
            params["gen_type"] = generator_type
        if project_id is not None:
            where_clauses.append("s.project_id = :project_id")
            params["project_id"] = project_id
        if q:
            where_clauses.append("(s.name LIKE :q OR s.definition LIKE :q)")
            params["q"] = f"%{q}%"

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        total = db.execute(text(f"SELECT COUNT(*) FROM series s {where}"), params).scalar()
        rows = db.execute(
            text(f"""
                SELECT s.id, s.project_id, s.name, s.definition, s.generator_type, s.provenance, s.created_at,
                       (SELECT COUNT(*) FROM representation r WHERE r.series_id = s.id) AS repr_count
                FROM series s {where}
                ORDER BY s.id
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()
        items = [
            SeriesOut(
                id=r[0], project_id=r[1], name=r[2], definition=r[3],
                generator_type=r[4], provenance=r[5], created_at=_str(r[6]),
                representation_count=r[7],
            )
            for r in rows
        ]
        return Paginated(total=total or 0, offset=offset, limit=limit, items=items)
    finally:
        db.close()


@app.get("/series/{series_id}", tags=["series"])
def get_series(series_id: int):
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT id, project_id, name, definition, generator_type, provenance, created_at FROM series WHERE id = :id"),
            {"id": series_id},
        ).fetchone()
        if not row:
            raise HTTPException(404, "Series not found")
        repr_rows = db.execute(
            text("SELECT id, primary_group, canonical_fingerprint FROM representation WHERE series_id = :sid ORDER BY id"),
            {"sid": series_id},
        ).fetchall()
        return {
            "id": row[0], "project_id": row[1], "name": row[2], "definition": row[3],
            "generator_type": row[4], "provenance": row[5], "created_at": _str(row[6]),
            "representations": [{"id": r[0], "primary_group": r[1], "canonical_fingerprint": r[2]} for r in repr_rows],
        }
    finally:
        db.close()


# --- Representations ---

@app.get("/representations", tags=["representations"])
def list_representations(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    primary_group: Optional[str] = None,
    series_id: Optional[int] = None,
    q: Optional[str] = None,
):
    db: Session = next(get_db())
    try:
        where_clauses = []
        params: dict = {"limit": limit, "offset": offset}

        if primary_group:
            where_clauses.append("r.primary_group = :pg")
            params["pg"] = primary_group
        if series_id is not None:
            where_clauses.append("r.series_id = :series_id")
            params["series_id"] = series_id
        if q:
            where_clauses.append("(r.canonical_fingerprint LIKE :q OR r.canonical_payload LIKE :q)")
            params["q"] = f"%{q}%"

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        total = db.execute(text(f"SELECT COUNT(*) FROM representation r {where}"), params).scalar()
        rows = db.execute(
            text(f"""
                SELECT r.id, r.series_id, r.primary_group, r.canonical_fingerprint,
                       r.canonical_payload, r.overlap_groups, r.created_at
                FROM representation r {where}
                ORDER BY r.id
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()
        items = [
            RepresentationOut(
                id=r[0], series_id=r[1], primary_group=r[2],
                canonical_fingerprint=r[3],
                canonical_payload=_safe_json(r[4]),
                overlap_groups=_safe_json(r[5]),
                created_at=_str(r[6]),
            )
            for r in rows
        ]
        return Paginated(total=total or 0, offset=offset, limit=limit, items=items)
    finally:
        db.close()


@app.get("/representations/{repr_id}", tags=["representations"])
def get_representation(repr_id: int):
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT id, series_id, primary_group, canonical_fingerprint, canonical_payload, overlap_groups, created_at FROM representation WHERE id = :id"),
            {"id": repr_id},
        ).fetchone()
        if not row:
            raise HTTPException(404, "Representation not found")

        feat_row = db.execute(
            text("SELECT feature_json, feature_version, computed_at FROM features WHERE representation_id = :rid"),
            {"rid": repr_id},
        ).fetchone()

        cmf_rows = db.execute(
            text("SELECT id, dimension, direction_policy, cmf_payload FROM cmf WHERE representation_id = :rid ORDER BY id"),
            {"rid": repr_id},
        ).fetchall()

        equiv_rows = db.execute(
            text("""
                SELECT ec.id, ec.primary_group, ec.class_fingerprint
                FROM equivalence_class ec
                JOIN representation_equivalence re ON re.equivalence_class_id = ec.id
                WHERE re.representation_id = :rid
            """),
            {"rid": repr_id},
        ).fetchall()

        return {
            "id": row[0], "series_id": row[1], "primary_group": row[2],
            "canonical_fingerprint": row[3],
            "canonical_payload": _safe_json(row[4]),
            "overlap_groups": _safe_json(row[5]),
            "created_at": _str(row[6]),
            "features": {
                "feature_json": _safe_json(feat_row[0]),
                "feature_version": feat_row[1],
                "computed_at": _str(feat_row[2]),
            } if feat_row else None,
            "cmfs": [
                {"id": c[0], "dimension": c[1], "direction_policy": c[2], "cmf_payload": _safe_json(c[3])}
                for c in cmf_rows
            ],
            "equivalence_classes": [
                {"id": e[0], "primary_group": e[1], "class_fingerprint": e[2]}
                for e in equiv_rows
            ],
        }
    finally:
        db.close()


# --- CMFs ---

@app.get("/cmfs", tags=["cmfs"])
def list_cmfs(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    dimension: Optional[int] = None,
    representation_id: Optional[int] = None,
):
    db: Session = next(get_db())
    try:
        where_clauses = []
        params: dict = {"limit": limit, "offset": offset}

        if dimension is not None:
            where_clauses.append("c.dimension = :dim")
            params["dim"] = dimension
        if representation_id is not None:
            where_clauses.append("c.representation_id = :repr_id")
            params["repr_id"] = representation_id

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        total = db.execute(text(f"SELECT COUNT(*) FROM cmf c {where}"), params).scalar()
        rows = db.execute(
            text(f"""
                SELECT c.id, c.representation_id, c.cmf_payload, c.dimension, c.direction_policy, c.created_at
                FROM cmf c {where}
                ORDER BY c.id
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()
        items = [
            CMFOut(
                id=r[0], representation_id=r[1], cmf_payload=_safe_json(r[2]),
                dimension=r[3], direction_policy=r[4], created_at=_str(r[5]),
            )
            for r in rows
        ]
        return Paginated(total=total or 0, offset=offset, limit=limit, items=items)
    finally:
        db.close()


@app.get("/cmfs/browse", tags=["cmfs"])
def browse_cmfs(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    q: Optional[str] = None,
    dimension: Optional[int] = None,
    min_dimension: Optional[int] = Query(2, ge=1, description="Minimum dimension (default 2, excludes 1D)"),
    degree: Optional[int] = None,
    certification: Optional[str] = None,
    source_type: Optional[str] = None,
    has_formula: Optional[bool] = None,
    primary_constant: Optional[str] = None,
):
    """Browse CMFs with filtering and search."""
    db: Session = next(get_db())
    try:
        where_clauses: list = []
        params: dict = {"limit": limit, "offset": offset}

        if dimension is not None:
            where_clauses.append("c.dimension = :dimension")
            params["dimension"] = dimension
        elif min_dimension is not None:
            where_clauses.append("c.dimension >= :min_dimension")
            params["min_dimension"] = min_dimension
        if degree is not None:
            where_clauses.append("CAST(json_extract(c.cmf_payload,'$.degree') AS INTEGER) = :degree")
            params["degree"] = degree
        if certification:
            where_clauses.append("json_extract(c.cmf_payload,'$.certification_level') = :cert")
            params["cert"] = certification
        if source_type:
            where_clauses.append("s.generator_type = :source_type")
            params["source_type"] = source_type
        if has_formula is True:
            where_clauses.append(
                "json_extract(c.cmf_payload,'$.f_poly') IS NOT NULL "
                "AND json_extract(c.cmf_payload,'$.f_poly') != ''"
            )
        elif has_formula is False:
            where_clauses.append(
                "(json_extract(c.cmf_payload,'$.f_poly') IS NULL "
                "OR json_extract(c.cmf_payload,'$.f_poly') = '')"
            )
        if primary_constant:
            where_clauses.append("json_extract(c.cmf_payload,'$.primary_constant') LIKE :pc")
            params["pc"] = f"%{primary_constant}%"
        if q:
            where_clauses.append(
                "(c.cmf_payload LIKE :q OR r.canonical_fingerprint LIKE :q OR s.definition LIKE :q)"
            )
            params["q"] = f"%{q}%"

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        total = db.execute(text(f"""
            SELECT COUNT(*) FROM cmf c
            JOIN representation r ON r.id = c.representation_id
            JOIN series s ON s.id = r.series_id
            {where}
        """), params).scalar() or 0

        rows = db.execute(text(f"""
            SELECT c.id, c.dimension,
                   json_extract(c.cmf_payload,'$.f_poly')              AS f_poly,
                   json_extract(c.cmf_payload,'$.fbar_poly')           AS fbar_poly,
                   json_extract(c.cmf_payload,'$.degree')              AS degree,
                   json_extract(c.cmf_payload,'$.primary_constant')    AS primary_constant,
                   json_extract(c.cmf_payload,'$.certification_level') AS cert,
                   json_extract(c.cmf_payload,'$.source')              AS source,
                   json_extract(c.cmf_payload,'$.flatness_verified')   AS flat,
                   r.canonical_fingerprint, r.primary_group,
                   s.generator_type, s.name AS series_name
            FROM cmf c
            JOIN representation r ON r.id = c.representation_id
            JOIN series s ON s.id = r.series_id
            {where}
            ORDER BY c.id
            LIMIT :limit OFFSET :offset
        """), params).fetchall()

        items = []
        for r in rows:
            items.append({
                "id":                  r[0],
                "dimension":           r[1],
                "f_poly":              r[2] or "",
                "fbar_poly":           r[3] or "",
                "degree":              int(r[4]) if r[4] is not None else 0,
                "primary_constant":    r[5],
                "certification_level": r[6],
                "source":              r[7],
                "flatness_verified":   bool(r[8]) if r[8] is not None else False,
                "canonical_fingerprint": r[9],
                "primary_group":       r[10],
                "generator_type":      r[11],
                "series_name":         r[12],
                "has_formula":         bool(r[2]),
            })

        return {"total": total, "offset": offset, "limit": limit, "items": items}
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}", tags=["cmfs"])
def get_cmf(cmf_id: int):
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT id, representation_id, cmf_payload, dimension, direction_policy, created_at FROM cmf WHERE id = :id"),
            {"id": cmf_id},
        ).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")

        eval_rows = db.execute(
            text("""
                SELECT id, run_type, precision_digits, steps, limit_estimate,
                       error_estimate, convergence_score, stability_score, runtime_ms, notes, created_at
                FROM eval_run WHERE cmf_id = :cid ORDER BY id
            """),
            {"cid": cmf_id},
        ).fetchall()

        return {
            "id": row[0], "representation_id": row[1],
            "cmf_payload": _safe_json(row[2]),
            "dimension": row[3], "direction_policy": row[4], "created_at": _str(row[5]),
            "eval_runs": [
                {
                    "id": e[0], "run_type": e[1], "precision_digits": e[2], "steps": e[3],
                    "limit_estimate": e[4], "error_estimate": e[5],
                    "convergence_score": e[6], "stability_score": e[7],
                    "runtime_ms": e[8], "notes": e[9], "created_at": _str(e[10]),
                }
                for e in eval_rows
            ],
        }
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}/full", tags=["cmfs"])
def get_cmf_full(cmf_id: int):
    """Full CMF details with joined representation, features, and series."""
    db: Session = next(get_db())
    try:
        row = db.execute(text("""
            SELECT c.id, c.dimension, c.direction_policy, c.cmf_payload, c.created_at,
                   r.id, r.primary_group, r.canonical_fingerprint, r.canonical_payload,
                   s.id, s.name, s.generator_type, s.definition, s.provenance,
                   f.feature_json
            FROM cmf c
            JOIN representation r ON r.id = c.representation_id
            JOIN series s ON s.id = r.series_id
            LEFT JOIN features f ON f.representation_id = r.id
            WHERE c.id = :id
        """), {"id": cmf_id}).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")

        payload = _safe_json(row[3]) or {}
        canon   = _safe_json(row[8]) or {}
        feats   = _safe_json(row[14]) or {}

        return {
            "id":                  row[0],
            "dimension":           row[1],
            "direction_policy":    row[2],
            "created_at":          _str(row[4]),
            "f_poly":              payload.get("f_poly", ""),
            "fbar_poly":           payload.get("fbar_poly", ""),
            "degree":              payload.get("degree", 0),
            "primary_constant":    payload.get("primary_constant"),
            "certification_level": payload.get("certification_level"),
            "source":              payload.get("source"),
            "flatness_verified":   payload.get("flatness_verified", False),
            "representation": {
                "id":                  row[5],
                "primary_group":       row[6],
                "canonical_fingerprint": row[7],
                "source_type":         canon.get("source_type"),
                "order":               canon.get("order"),
                "deg_x":               canon.get("deg_x"),
                "deg_y":               canon.get("deg_y"),
                "n_monomials":         canon.get("n_monomials"),
                "conjugacy":           canon.get("conjugacy"),
            },
            "series": {
                "id":             row[9],
                "name":           row[10],
                "generator_type": row[11],
                "definition":     row[12],
                "provenance":     row[13],
            },
            "features":        feats,
            "walk_available":  bool(payload.get("f_poly")),
        }
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}/walk", tags=["cmfs"])
def walk_cmf(
    cmf_id: int,
    depth: int = Query(100, ge=10, le=500),
    m_fixed: int = Query(0, ge=0, le=50),
):
    """
    K1 matrix walk for a CMF with polynomial form.
    Accumulates P = K1(0,m)*K1(1,m)*...*K1(depth-1,m) and reads partial
    convergents P[0,1]/P[1,1] at each step.
    """
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT cmf_payload FROM cmf WHERE id = :id"), {"id": cmf_id}
        ).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")

        payload   = _safe_json(row[0]) or {}
        f_poly    = payload.get("f_poly", "")
        fbar_poly = payload.get("fbar_poly", "")

        if not f_poly or not fbar_poly:
            raise HTTPException(422, "CMF has no polynomial form — walk not available")

        try:
            K1_fn, _ = _build_walk_fns(f_poly, fbar_poly)
        except Exception as exc:
            raise HTTPException(500, f"Polynomial parse error: {exc}")

        mpmath.mp.dps = 30
        P = mpmath.eye(2)
        sequence = []

        for step in range(depth):
            try:
                K = K1_fn(step, m_fixed)
                P = P * K
                # Normalize every 20 steps to prevent overflow
                if (step + 1) % 20 == 0:
                    scale = max(abs(float(mpmath.re(P[0, 0]))),
                                abs(float(mpmath.re(P[0, 1]))), 1e-300)
                    P = P / scale
                denom = P[1, 1]
                numer = P[0, 1]
                if mpmath.fabs(denom) > mpmath.mpf("1e-200"):
                    val = float(mpmath.re(numer / denom))
                    sequence.append({"step": step + 1, "value": val if math.isfinite(val) else None})
                else:
                    sequence.append({"step": step + 1, "value": None})
            except Exception:
                sequence.append({"step": step + 1, "value": None})

        finite = [s["value"] for s in sequence if s["value"] is not None]
        best = finite[-1] if finite else None

        # Convergence rate over last 20 values
        conv_rate = None
        if best is not None and len(finite) >= 20:
            diffs = [abs(v - best) for v in finite[-20:] if abs(v - best) > 0]
            if len(diffs) >= 2:
                try:
                    conv_rate = round(
                        (math.log10(diffs[-1]) - math.log10(diffs[0])) / len(diffs), 3
                    )
                except Exception:
                    pass

        return {
            "cmf_id":          cmf_id,
            "depth":           depth,
            "m_fixed":         m_fixed,
            "f_poly":          f_poly,
            "fbar_poly":       fbar_poly,
            "sequence":        sequence,
            "best_estimate":   best,
            "conv_rate_log10_per_step": conv_rate,
            "constant_matches": _compare_constants(best) if best is not None else [],
            "primary_constant": payload.get("primary_constant"),
            "certification_level": payload.get("certification_level"),
        }
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}/conservative-test", tags=["cmfs"])
def conservative_test(
    cmf_id: int,
    k_max: int = Query(5, ge=1, le=15),
    m_max: int = Query(5, ge=1, le=15),
):
    """
    Empirical flatness (path-independence) test.
    Checks K1(k,m)*K2(k+1,m) = K2(k,m)*K1(k,m+1) at a grid of (k,m) points.
    This is numerical evidence only — not a formal proof.
    """
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT cmf_payload FROM cmf WHERE id = :id"), {"id": cmf_id}
        ).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")

        payload   = _safe_json(row[0]) or {}
        f_poly    = payload.get("f_poly", "")
        fbar_poly = payload.get("fbar_poly", "")

        if not f_poly or not fbar_poly:
            raise HTTPException(422, "CMF has no polynomial form — conservative test not available")

        try:
            K1_fn, K2_fn = _build_walk_fns(f_poly, fbar_poly)
        except Exception as exc:
            raise HTTPException(500, f"Polynomial parse error: {exc}")

        mpmath.mp.dps = 20
        test_points = []
        max_residual = 0.0

        for k in range(1, k_max + 1):
            for m in range(1, m_max + 1):
                try:
                    lhs  = K1_fn(k, m) * K2_fn(k + 1, m)
                    rhs  = K2_fn(k, m) * K1_fn(k, m + 1)
                    diff = lhs - rhs
                    # Frobenius norm manually
                    res = float(mpmath.sqrt(sum(
                        mpmath.fabs(diff[i, j])**2 for i in range(2) for j in range(2)
                    )))
                    max_residual = max(max_residual, res)
                    test_points.append({
                        "k": k, "m": m,
                        "residual": res,
                        "log10_residual": round(math.log10(res), 2) if res > 0 else -20,
                    })
                except Exception as exc:
                    test_points.append({"k": k, "m": m, "residual": None, "error": str(exc)})

        finite_res = [p["residual"] for p in test_points if p.get("residual") is not None]
        if not finite_res:
            verdict = "inconclusive"
        elif max(finite_res) < 1e-10:
            verdict = "consistent_with_path_independence"
        elif max(finite_res) < 1e-3:
            verdict = "approximately_path_independent"
        else:
            verdict = "numerically_inconsistent"

        return {
            "cmf_id":         cmf_id,
            "f_poly":         f_poly,
            "fbar_poly":      fbar_poly,
            "test_points":    test_points,
            "max_residual":   max_residual,
            "verdict":        verdict,
            "k_max":          k_max,
            "m_max":          m_max,
            "note": (
                "Checks K1(k,m)\u00b7K2(k+1,m) = K2(k,m)\u00b7K1(k,m+1) numerically. "
                "Small residual is empirical evidence of path independence, not a formal proof."
            ),
        }
    finally:
        db.close()


# --- Eval runs ---

@app.get("/eval-runs/{run_id}", tags=["eval_runs"])
def get_eval_run(run_id: int):
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("""
                SELECT id, cmf_id, run_type, precision_digits, steps, limit_estimate,
                       error_estimate, convergence_score, stability_score, runtime_ms, notes, created_at
                FROM eval_run WHERE id = :id
            """),
            {"id": run_id},
        ).fetchone()
        if not row:
            raise HTTPException(404, "Eval run not found")

        rec_rows = db.execute(
            text("""
                SELECT id, method, basis_name, basis_payload, success, identified_as,
                       relation_height, residual_log10, attempt_log, created_at
                FROM recognition_attempt WHERE eval_run_id = :eid ORDER BY id
            """),
            {"eid": run_id},
        ).fetchall()

        return {
            "id": row[0], "cmf_id": row[1], "run_type": row[2],
            "precision_digits": row[3], "steps": row[4], "limit_estimate": row[5],
            "error_estimate": row[6], "convergence_score": row[7],
            "stability_score": row[8], "runtime_ms": row[9],
            "notes": row[10], "created_at": _str(row[11]),
            "recognition_attempts": [
                {
                    "id": r[0], "method": r[1], "basis_name": r[2],
                    "basis_payload": _safe_json(r[3]), "success": bool(r[4]),
                    "identified_as": r[5], "relation_height": r[6],
                    "residual_log10": r[7], "attempt_log": r[8],
                    "created_at": _str(r[9]),
                }
                for r in rec_rows
            ],
        }
    finally:
        db.close()


# --- Recognition successes ---

@app.get("/recognitions", tags=["recognition"])
def list_recognitions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    success_only: bool = True,
    method: Optional[str] = None,
):
    """List recognition attempts (defaults to successes only)."""
    db: Session = next(get_db())
    try:
        where_clauses = []
        params: dict = {"limit": limit, "offset": offset}

        if success_only:
            where_clauses.append("ra.success = 1")
        if method:
            where_clauses.append("ra.method = :method")
            params["method"] = method

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        total = db.execute(text(f"SELECT COUNT(*) FROM recognition_attempt ra {where}"), params).scalar()
        rows = db.execute(
            text(f"""
                SELECT ra.id, ra.eval_run_id, ra.method, ra.basis_name, ra.success,
                       ra.identified_as, ra.relation_height, ra.residual_log10, ra.created_at
                FROM recognition_attempt ra {where}
                ORDER BY ra.id
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()
        items = [
            {
                "id": r[0], "eval_run_id": r[1], "method": r[2], "basis_name": r[3],
                "success": bool(r[4]), "identified_as": r[5],
                "relation_height": r[6], "residual_log10": r[7], "created_at": _str(r[8]),
            }
            for r in rows
        ]
        return Paginated(total=total or 0, offset=offset, limit=limit, items=items)
    finally:
        db.close()


# --- Search across all tables ---

@app.get("/search", tags=["search"])
def search(
    q: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(20, ge=1, le=100),
):
    """Full-text search across series definitions, fingerprints, payloads, and recognitions."""
    db: Session = next(get_db())
    try:
        pattern = f"%{q}%"
        results = []

        # Search series definitions
        rows = db.execute(
            text("SELECT id, name, definition, generator_type FROM series WHERE name LIKE :q OR definition LIKE :q LIMIT :limit"),
            {"q": pattern, "limit": limit},
        ).fetchall()
        for r in rows:
            results.append({"type": "series", "id": r[0], "name": r[1], "definition": r[2], "generator_type": r[3]})

        # Search representation fingerprints
        rows = db.execute(
            text("SELECT id, series_id, primary_group, canonical_fingerprint FROM representation WHERE canonical_fingerprint LIKE :q LIMIT :limit"),
            {"q": pattern, "limit": limit},
        ).fetchall()
        for r in rows:
            results.append({"type": "representation", "id": r[0], "series_id": r[1], "primary_group": r[2], "canonical_fingerprint": r[3]})

        # Search recognition results
        rows = db.execute(
            text("SELECT id, eval_run_id, method, identified_as FROM recognition_attempt WHERE identified_as LIKE :q LIMIT :limit"),
            {"q": pattern, "limit": limit},
        ).fetchall()
        for r in rows:
            results.append({"type": "recognition", "id": r[0], "eval_run_id": r[1], "method": r[2], "identified_as": r[3]})

        return {"query": q, "total_results": len(results), "results": results}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Serve frontend static files (must be last — catches all unmatched routes)
# ---------------------------------------------------------------------------
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
