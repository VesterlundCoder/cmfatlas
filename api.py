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

from fastapi import FastAPI, HTTPException, Query, Request
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


def _poly_has_z(f_poly_str: str) -> bool:
    """Return True if f_poly_str contains the variable z."""
    return 'z' in str(_sympify(f_poly_str).free_symbols)


def _build_walk_fns(f_poly_str: str, fbar_poly_str: str):
    """Build Kx, Ky (and Kz for 3D) numerical matrix functions.

    2D (vars x,y):  maps x→k, y→m.  Returns (Kx(k,m,n), Ky(k,m,n), False).
    3D (vars x,y,z): maps x→k, y→m, z→n.  Returns (Kx(k,m,n), Ky(k,m,n), True).

    All returned functions accept (k, m, n) — n is ignored for 2D CMFs.

    Telescope convention:
        b(k)      = g(k,0,[0]) * gbar(k,0,[0])
        a(k,m,[n])= g(k,m,[n]) - gbar(k+1,m,[n])
        Kx = [[0,1],[b(k+1),a]]   (k-step matrix)
        Ky = [[gbar,1],[b,g]]     (m-step matrix)
        Kz = [[gbar,1],[b,g]]     (n-step matrix, 3D only, same functional form as Ky)
    """
    k_s, m_s, x_s, y_s = _sym("k m x y")
    f_expr    = _sympify(f_poly_str)
    fbar_expr = _sympify(fbar_poly_str)
    is_3d     = _poly_has_z(f_poly_str)

    if is_3d:
        # Use free_symbols to identify x,y,z regardless of assumption cache
        _free = f_expr.free_symbols
        x_sym = next((s for s in _free if s.name == 'x'), x_s)
        y_sym = next((s for s in _free if s.name == 'y'), y_s)
        z_sym = next((s for s in _free if s.name == 'z'), _sym("z"))

        g_kmz    = f_expr.subs([(x_sym, k_s), (y_sym, m_s)])
        gbar_kmz = fbar_expr.subs([(x_sym, k_s), (y_sym, m_s)])
        b_kz_expr = _expand(g_kmz.subs(m_s, 0) * gbar_kmz.subs(m_s, 0))
        a_expr    = _expand(g_kmz - gbar_kmz.subs(k_s, k_s + 1))

        # Use subs-based evaluation — lambdify has a symbol-identity issue for z
        def _eval(expr, k_v, m_v, z_v):
            return complex(expr.subs([(k_s, k_v), (m_s, m_v), (z_sym, z_v)]))

        def _eval_b(expr, k_v, z_v):
            return complex(expr.subs([(k_s, k_v), (z_sym, z_v)]))

        def Kx(k, m, n):
            b = _eval_b(b_kz_expr, k + 1, n)
            a = _eval(a_expr, k, m, n)
            return mpmath.matrix([[0, 1], [b, a]])
        def Ky(k, m, n):
            g    = _eval(g_kmz,    k, m, n)
            gbar = _eval(gbar_kmz, k, m, n)
            b    = _eval_b(b_kz_expr, k, n)
            return mpmath.matrix([[gbar, 1], [b, g]])
        def Kz(k, m, n):
            g    = _eval(g_kmz,    k, m, n)
            gbar = _eval(gbar_kmz, k, m, n)
            b    = _eval_b(b_kz_expr, k, n)
            return mpmath.matrix([[gbar, 1], [b, g]])

        return Kx, Ky, Kz, True

    # 2D
    g_km    = f_expr.subs([(x_s, k_s), (y_s, m_s)])
    gbar_km = fbar_expr.subs([(x_s, k_s), (y_s, m_s)])
    b_expr  = _expand(g_km.subs(m_s, 0) * gbar_km.subs(m_s, 0))
    a_expr  = _expand(g_km - gbar_km.subs(k_s, k_s + 1))

    g_fn    = _lambdify([k_s, m_s], g_km,    modules="mpmath")
    gbar_fn = _lambdify([k_s, m_s], gbar_km, modules="mpmath")
    b_fn    = _lambdify([k_s],       b_expr,  modules="mpmath")
    a_fn    = _lambdify([k_s, m_s], a_expr,  modules="mpmath")

    def Kx2(k, m, n=0):
        return mpmath.matrix([[0, 1], [b_fn(k + 1), a_fn(k, m)]])
    def Ky2(k, m, n=0):
        return mpmath.matrix([[gbar_fn(k, m), 1], [b_fn(k), g_fn(k, m)]])

    return Kx2, Ky2, None, False


def _build_matrix_walk_from_stored(matrices_dict: dict, axes: list, direction: str):
    """
    Build a K(step, fixed_vals) callable from the stored matrix expression strings
    used by RamanujanTools / known_family CMFs.

    matrices_dict: {axis_name: "[[expr, expr], [expr, expr]]"}
    axes: list of axis variable names, e.g. ['x','y'] or ['a','b','c']
    direction: 'x', 'y', 'z'
    Returns (K_fn, step_axis, matrix_size) or None on failure.
    """
    import ast as _ast
    dir_idx = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
    if dir_idx >= len(axes):
        return None
    step_axis = axes[dir_idx]
    mat_str = matrices_dict.get(step_axis)
    if not mat_str:
        return None
    try:
        parsed_rows = _ast.literal_eval(mat_str)
    except Exception:
        return None
    size = len(parsed_rows)
    # Pre-compile each cell expression for speed
    compiled = []
    for row in parsed_rows:
        crow = []
        for cell in row:
            try:
                crow.append(compile(str(cell), "<cmf_expr>", "eval"))
            except Exception:
                crow.append(None)
        compiled.append(crow)

    def K_fn(step_val, fixed_vals: dict):
        ns = {ax: float(v) for ax, v in fixed_vals.items()}
        ns[step_axis] = float(step_val)
        mat = mpmath.zeros(size)
        for i in range(size):
            for j in range(len(compiled[i])):
                code = compiled[i][j]
                if code is None:
                    mat[i, j] = mpmath.mpf(0)
                    continue
                try:
                    raw = eval(code, {"__builtins__": None}, ns)  # noqa: S307
                    mat[i, j] = mpmath.mpf(str(raw))
                except Exception:
                    mat[i, j] = mpmath.mpf(0)
        return mat

    return K_fn, step_axis, size


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}")
    # Ensure category column exists (idempotent migration for Railway)
    import sqlite3 as _sq, json as _js, re as _re
    _TRANS_RE = _re.compile(
        r"zeta|pi|log|ln\(|digamma|psi\(|catalan|euler|\be\b|harmonic|gamma|\bG\b"
        r"|eta\(|hurwitz|dirichlet|3f2|hypergeometric|2f1|pfq|identified",
        _re.IGNORECASE)
    _RAT_RE = _re.compile(r"^-?\s*\d+\s*/\s*\d+\s*$")
    _DISC_SRC = ["ramanujantools", "pfq", "3f2", "hypergeometric", "discovery"]
    def _classify(payload, dim):
        cert = (payload.get("certification_level") or "").lower()
        src  = (payload.get("source_category") or payload.get("source") or "").lower()
        pc   = (payload.get("primary_constant") or "").strip()
        deg  = int(payload.get("degree") or 0)
        if dim >= 3 or any(s in src for s in _DISC_SRC) or cert == "a_plus":
            return "discovery"
        if deg <= 1: return "reference"
        if (not pc or pc.lower() in ("none", "", "null")) and deg <= 2: return "reference"
        if _RAT_RE.match(pc): return "reference"
        if pc and not _TRANS_RE.search(pc): return "reference"
        return "interesting"
    _conn = _sq.connect(DB_PATH)
    _cur  = _conn.cursor()
    _cols = [r[1] for r in _cur.execute("PRAGMA table_info(cmf)").fetchall()]
    if "category" not in _cols:
        _cur.execute("ALTER TABLE cmf ADD COLUMN category TEXT")
    _cur.execute("CREATE INDEX IF NOT EXISTS idx_cmf_category ON cmf(category)")
    _rows = _cur.execute("SELECT id, dimension, cmf_payload FROM cmf WHERE category IS NULL").fetchall()
    for _rid, _dim, _pay in _rows:
        try: _p = _js.loads(_pay) if _pay else {}
        except: _p = {}
        _cur.execute("UPDATE cmf SET category=? WHERE id=?", (_classify(_p, _dim or 2), _rid))
    _conn.commit(); _conn.close()

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


@app.get("/export/cmfs.json", tags=["meta"])
def export_cmfs():
    """Bulk dataset download — all CMFs as a single JSON array (CC BY 4.0)."""
    db: Session = next(get_db())
    try:
        rows = db.execute(text("""
            SELECT c.id, c.dimension, c.cmf_payload, c.created_at,
                   r.canonical_fingerprint, r.canonical_payload,
                   s.name, s.provenance
            FROM cmf c
            JOIN representation r ON r.id = c.representation_id
            JOIN series s ON s.id = r.series_id
            ORDER BY c.id
        """)).fetchall()
        out = []
        for row in rows:
            payload  = _safe_json(row[2]) or {}
            canon    = _safe_json(row[5]) or {}
            out.append({
                "id":                   row[0],
                "dimension":            row[1],
                "f_poly":               payload.get("f_poly", ""),
                "fbar_poly":            payload.get("fbar_poly", ""),
                "degree":               payload.get("degree", 0),
                "primary_constant":     payload.get("primary_constant"),
                "certification_level":  payload.get("certification_level"),
                "source_category":      payload.get("source_category"),
                "flatness_verified":    payload.get("flatness_verified", False),
                "canonical_fingerprint": row[4],
                "series_name":          row[6],
                "provenance":           row[7],
                "created_at":           _str(row[3]),
                "entry_uri":            payload.get("entry_uri"),
                "proof_status":         payload.get("proof_status"),
                "identification_status": payload.get("identification_status"),
                "source_family":        payload.get("source_family"),
                "construction_type":    payload.get("construction_type"),
            })
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={
                "version": "2.5",
                "schema_version": "2.5",
                "license": "CC BY 4.0",
                "url": "https://davidvesterlund.com/cmf-atlas/",
                "total": len(out),
                "cmfs": out,
            },
            headers={"Content-Disposition": "attachment; filename=cmfs.json"},
        )
    finally:
        db.close()


# Known constants referenced by CMF Atlas entries
_CONSTANTS_REGISTRY = [
    {"id": "zeta3",      "label": "\u03b6(3)",         "latex": "\\zeta(3)",
     "aliases": ["Apery's constant", "zeta(3)"],
     "value_30d": "1.202056903159594285399738161511",
     "formula": "\\sum_{n=1}^\\infty n^{-3}",
     "irrational": True,  "proven_irrational": True,  "transcendental": None,
     "irrationality_status": "proven_irrational",  "transcendence_status": "open_question",
     "reference": "https://mathworld.wolfram.com/AperysConstant.html"},
    {"id": "pi",         "label": "\u03c0",             "latex": "\\pi",
     "aliases": ["pi", "Archimedes constant"],
     "value_30d": "3.141592653589793238462643383280",
     "formula": "4 \\sum_{n=0}^\\infty \\frac{(-1)^n}{2n+1}",
     "irrational": True,  "proven_irrational": True,  "transcendental": True,
     "irrationality_status": "proven_transcendental",  "transcendence_status": "proven_transcendental",
     "reference": "https://mathworld.wolfram.com/Pi.html"},
    {"id": "e",          "label": "e",             "latex": "e",
     "aliases": ["Euler's number", "exp(1)"],
     "value_30d": "2.718281828459045235360287471353",
     "formula": "\\sum_{n=0}^\\infty \\frac{1}{n!}",
     "irrational": True,  "proven_irrational": True,  "transcendental": True,
     "irrationality_status": "proven_transcendental",  "transcendence_status": "proven_transcendental",
     "reference": "https://mathworld.wolfram.com/e.html"},
    {"id": "ln2",        "label": "ln 2",          "latex": "\\ln 2",
     "aliases": ["log(2)", "ln(2)"],
     "value_30d": "0.693147180559945309417232121458",
     "formula": "\\sum_{n=1}^\\infty \\frac{(-1)^{n+1}}{n}",
     "irrational": True,  "proven_irrational": True,  "transcendental": True,
     "irrationality_status": "proven_transcendental",  "transcendence_status": "proven_transcendental",
     "reference": "https://mathworld.wolfram.com/NaturalLogarithmof2.html"},
    {"id": "zeta2",      "label": "\u03b6(2) = \u03c0\u00b2/6",  "latex": "\\zeta(2) = \\pi^2/6",
     "aliases": ["pi^2/6", "zeta(2)"],
     "value_30d": "1.644934066848226436472415166646",
     "formula": "\\sum_{n=1}^\\infty n^{-2} = \\pi^2/6",
     "irrational": True,  "proven_irrational": True,  "transcendental": True,
     "irrationality_status": "proven_transcendental",  "transcendence_status": "proven_transcendental",
     "reference": "https://mathworld.wolfram.com/RiemannZetaFunction.html"},
    {"id": "zeta5",      "label": "\u03b6(5)",          "latex": "\\zeta(5)",
     "aliases": ["zeta(5)"],
     "value_30d": "1.036927755143369926331365486458",
     "formula": "\\sum_{n=1}^\\infty n^{-5}",
     "irrational": None,  "proven_irrational": False,  "transcendental": None,
     "irrationality_status": "open_question",  "transcendence_status": "open_question",
     "reference": "https://mathworld.wolfram.com/RiemannZetaFunction.html"},
    {"id": "catalan",    "label": "G (Catalan)",   "latex": "G",
     "aliases": ["Catalan's constant", "Catalan"],
     "value_30d": "0.915965594177219015054603514932",
     "formula": "\\sum_{n=0}^\\infty \\frac{(-1)^n}{(2n+1)^2}",
     "irrational": None,  "proven_irrational": False,  "transcendental": None,
     "irrationality_status": "open_question",  "transcendence_status": "open_question",
     "reference": "https://mathworld.wolfram.com/CatalansConstant.html"},
    {"id": "sqrt2",      "label": "\u221a2",            "latex": "\\sqrt{2}",
     "aliases": ["sqrt(2)", "Pythagoras constant"],
     "value_30d": "1.414213562373095048801688724210",
     "formula": "\\sqrt{2}",
     "irrational": True,  "proven_irrational": True,  "transcendental": False,
     "irrationality_status": "proven_irrational",  "transcendence_status": "algebraic",
     "reference": "https://mathworld.wolfram.com/PythagorassConstant.html"},
    {"id": "pi_half",    "label": "\u03c0/2",           "latex": "\\pi/2",
     "aliases": ["pi/2"],
     "value_30d": "1.570796326794896619231321691640",
     "formula": "\\pi/2",
     "irrational": True,  "proven_irrational": True,  "transcendental": True,
     "irrationality_status": "proven_transcendental",  "transcendence_status": "proven_transcendental",
     "reference": "https://mathworld.wolfram.com/Pi.html"},
    {"id": "pi_quarter", "label": "\u03c0/4",           "latex": "\\pi/4",
     "aliases": ["pi/4", "Leibniz formula limit"],
     "value_30d": "0.785398163397448309615660845820",
     "formula": "\\pi/4 = 1 - 1/3 + 1/5 - \\cdots",
     "irrational": True,  "proven_irrational": True,  "transcendental": True,
     "irrationality_status": "proven_transcendental",  "transcendence_status": "proven_transcendental",
     "reference": "https://mathworld.wolfram.com/Pi.html"},
]


@app.get("/constants", tags=["meta"])
def get_constants():
    """Registry of known mathematical constants referenced by CMF Atlas entries."""
    db: Session = next(get_db())
    try:
        counts_rows = db.execute(text(
            "SELECT json_extract(cmf_payload,'$.primary_constant'), COUNT(*) "
            "FROM cmf WHERE json_extract(cmf_payload,'$.primary_constant') IS NOT NULL "
            "GROUP BY json_extract(cmf_payload,'$.primary_constant')"
        )).fetchall()
        counts_by_label = {r[0]: r[1] for r in counts_rows}
        enriched = []
        for c in _CONSTANTS_REGISTRY:
            entry = dict(c)
            all_labels = [c["label"]] + c.get("aliases", [])
            cnt = max((counts_by_label.get(lbl, 0) for lbl in all_labels), default=0)
            entry["entry_count"] = cnt
            enriched.append(entry)
        return {
            "version": "2.4",
            "schema_version": "2.4",
            "description": "Constants appearing as identified limits of CMFs in the atlas",
            "count": len(enriched),
            "constants": enriched,
        }
    finally:
        db.close()


@app.get("/release", tags=["meta"])
def release_manifest():
    """Release manifest — version, schema, counts, and URLs for the current Atlas release."""
    db: Session = next(get_db())
    try:
        total = db.execute(text("SELECT COUNT(*) FROM cmf WHERE dimension >= 2")).scalar() or 0
        dims = db.execute(text(
            "SELECT dimension, COUNT(*) FROM cmf WHERE dimension >= 2 GROUP BY dimension"
        )).fetchall()
        certs = db.execute(text(
            "SELECT json_extract(cmf_payload,'$.certification_level'), COUNT(*) "
            "FROM cmf WHERE dimension >= 2 "
            "GROUP BY json_extract(cmf_payload,'$.certification_level')"
        )).fetchall()
        walkable = db.execute(text(
            "SELECT COUNT(*) FROM cmf WHERE dimension >= 2 AND "
            "json_extract(cmf_payload,'$.f_poly') IS NOT NULL AND "
            "json_extract(cmf_payload,'$.f_poly') != ''"
        )).scalar() or 0
        return {
            "version": "2.4",
            "schema_version": "2.4",
            "release_date": "2026-04-05",
            "license": "CC BY 4.0",
            "url": "https://davidvesterlund.com/cmf-atlas/",
            "api_base": "https://cmfatlas-production.up.railway.app",
            "export_url": "https://cmfatlas-production.up.railway.app/export/cmfs.json",
            "total_cmfs": total,
            "walkable_cmfs": walkable,
            "by_dimension": {str(d): c for d, c in dims},
            "by_certification": {c: n for c, n in certs if c},
        }
    finally:
        db.close()


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
            src = d.get("source_category") or d.get("source") or "unknown"
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

        const_sorted = dict(sorted(constants.items(), key=lambda x: -x[1])[:20])
        return {
            "total_cmfs": sum(v for _, v in dims),
            "by_dimension": {str(d): c for d, c in dims},
            "by_certification": certs,
            "by_source": dict(sorted(sources.items(), key=lambda x: -x[1])[:15]),
            "by_degree": dict(sorted(degrees.items(), key=lambda x: int(x[0]) if x[0] not in (None, "None") else -1)),
            "known_constants": const_sorted,
            "by_constant": const_sorted,
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
    source_category: Optional[str] = None,
    has_formula: Optional[bool] = None,
    primary_constant: Optional[str] = None,
    category: Optional[str] = None,
    matrix_size: Optional[int] = None,
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
        if source_category:
            where_clauses.append("json_extract(c.cmf_payload,'$.source_category') = :source_category")
            params["source_category"] = source_category
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
        if category:
            where_clauses.append("c.category = :category")
            params["category"] = category
        if matrix_size is not None:
            where_clauses.append("CAST(json_extract(c.cmf_payload,'$.matrix_size') AS INTEGER) = :matrix_size")
            params["matrix_size"] = matrix_size
        if q:
            where_clauses.append(
                "(c.cmf_payload LIKE :q OR r.canonical_fingerprint LIKE :q OR s.definition LIKE :q)"
            )
            params["q"] = f"%{q}%"

        # Always exclude hidden CMFs from public browse
        where_clauses.append(
            "(json_extract(c.cmf_payload,'$.hidden') IS NULL "
            "OR json_extract(c.cmf_payload,'$.hidden') = 0)"
        )
        where = "WHERE " + " AND ".join(where_clauses)

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
                   json_extract(c.cmf_payload,'$.identified_constant')  AS identified_constant,
                   json_extract(c.cmf_payload,'$.certification_level') AS cert,
                   json_extract(c.cmf_payload,'$.source')              AS source,
                   json_extract(c.cmf_payload,'$.source_category')     AS source_category,
                   json_extract(c.cmf_payload,'$.flatness_verified')   AS flat,
                   r.canonical_fingerprint, r.primary_group,
                   s.generator_type, s.name AS series_name,
                   c.category,
                   json_extract(r.canonical_payload,'$.matrices')      AS has_stored_matrices,
                   json_extract(c.cmf_payload,'$.matrix_size')          AS matrix_size
            FROM cmf c
            JOIN representation r ON r.id = c.representation_id
            JOIN series s ON s.id = r.series_id
            {where}
            ORDER BY c.id
            LIMIT :limit OFFSET :offset
        """), params).fetchall()

        _CERT_PROOF_STATUS = {
            "A_plus": "symbolically_certified",
            "A_certified": "verified",
            "B_verified_numeric": "numeric_only",
            "C_scouting": "unverified",
        }
        items = []
        for r in rows:
            cert = r[7]
            f_poly = r[2] or ""
            gen_type = r[13] or ""
            src_cat = r[9] or r[8] or ""
            primary_const = r[5]
            identified_const = r[6]
            proof_status = _CERT_PROOF_STATUS.get(cert, "unverified")
            if identified_const:
                identification_status = "pslq_identified"
            elif primary_const:
                identification_status = "matched"
            else:
                identification_status = "unidentified"
            if f_poly:
                construction_type = "telescope_polynomial"
            elif "pfq" in gen_type.lower() or "hypergeometric" in gen_type.lower():
                construction_type = "hypergeometric_pfq"
            else:
                construction_type = "matrix_explicit"
            items.append({
                "id":                  r[0],
                "dimension":           r[1],
                "f_poly":              f_poly,
                "fbar_poly":           r[3] or "",
                "degree":              int(r[4]) if r[4] is not None else 0,
                "primary_constant":    primary_const,
                "identified_constant": identified_const,
                "certification_level": cert,
                "source":              r[8],
                "source_category":     src_cat,
                "source_family":       src_cat,
                "flatness_verified":   bool(r[10]) if r[10] is not None else False,
                "canonical_fingerprint": r[11],
                "primary_group":       r[12],
                "matrix_size":         int(r[17]) if r[17] is not None else None,
                "generator_type":      gen_type,
                "series_name":         r[14],
                "has_formula":         bool(f_poly) or bool(r[16]),
                "category":            r[15] or "reference",
                "proof_status":        proof_status,
                "identification_status": identification_status,
                "construction_type":   construction_type,
                "entry_uri":           f"https://davidvesterlund.com/cmf-atlas/entry.html?id={r[0]}",
                "release_version":     "2.3",
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


def _compute_matrices(f_poly: str, fbar_poly: str, canonical_payload: dict) -> list:
    """Return list of K-matrix dicts with LaTeX entries."""
    explicit = canonical_payload.get("matrices")
    if explicit and isinstance(explicit, dict):
        result = []
        subs_k = ["₁","₂","₃","₄","₅","₆","₇","₈","₉","₁₀","₁₁"]
        for i, (axis, mat_str) in enumerate(explicit.items()):
            try:
                _sym_ns = {s: _sp.Symbol(s) for s in "abcdefghijklmnopqrstuvwxyz"}
                _sym_ns.update({f"x{j}": _sp.Symbol(f"x{j}") for j in range(5)})
                _sym_ns.update({f"y{j}": _sp.Symbol(f"y{j}") for j in range(3)})
                _sym_ns.update({f"c{j}": _sp.Symbol(f"c{j}") for j in range(5)})
                rows_raw = eval(mat_str, {"__builtins__": {}}, _sym_ns)  # noqa: S307
                rows_latex = []
                for row in rows_raw:
                    row_l = []
                    for e in row:
                        try:
                            row_l.append(_sp.latex(_sp.simplify(e)))
                        except Exception:
                            row_l.append(str(e))
                    rows_latex.append(row_l)
                _axis_labels = {"x": "K₁", "y": "K₂", "z": "K₃", "G": "G", "D": "D₀"}
                label = _axis_labels.get(axis, f"K{subs_k[i] if i < len(subs_k) else str(i+1)}")
                result.append({
                    "index": i + 1,
                    "label": label,
                    "axis": axis,
                    "source": "explicit",
                    "rows": rows_latex,
                })
            except Exception:
                continue
        return result

    if not f_poly or not fbar_poly:
        return []

    try:
        if _poly_has_z(f_poly):
            k, m = _sp.symbols("k m", integer=True)
            x, y, z_s = _sp.symbols("x y z")
            g    = _sympify(f_poly).subs([(x, k), (y, m)])
            gbar = _sympify(fbar_poly).subs([(x, k), (y, m)])
            # b(k,z) = g(k,0,z)*gbar(k,0,z) — keep z live
            b_kz  = _expand(g.subs(m, 0) * gbar.subs(m, 0))
            b_kz1 = _expand(b_kz.subs(k, k + 1))
            a_kmn = _expand(g - gbar.subs(k, k + 1))
            return [
                {
                    "index": 1, "label": "K_x", "axis": "k", "source": "computed",
                    "rows": [["0", "1"], [_sp.latex(b_kz1), _sp.latex(a_kmn)]],
                },
                {
                    "index": 2, "label": "K_y", "axis": "m", "source": "computed",
                    "rows": [[_sp.latex(gbar), "1"], [_sp.latex(b_kz), _sp.latex(g)]],
                },
                {
                    "index": 3, "label": "K_z", "axis": "n", "source": "computed",
                    "rows": [[_sp.latex(gbar), "1"], [_sp.latex(b_kz), _sp.latex(g)]],
                },
            ]
        k, m = _sp.symbols("k m", integer=True)
        x, y = _sp.symbols("x y")
        g    = _sympify(f_poly).subs([(x, k), (y, m)])
        gbar = _sympify(fbar_poly).subs([(x, k), (y, m)])
        b_k  = _expand(g.subs(m, 0) * gbar.subs(m, 0))
        b_k1 = _expand(b_k.subs(k, k + 1))
        a_km = _expand(g - gbar.subs(k, k + 1))
        return [
            {
                "index": 1, "label": "K₁", "axis": "k", "source": "computed",
                "rows": [["0", "1"], [_sp.latex(b_k1), _sp.latex(a_km)]],
            },
            {
                "index": 2, "label": "K₂", "axis": "m", "source": "computed",
                "rows": [[_sp.latex(gbar), "1"], [_sp.latex(b_k), _sp.latex(g)]],
            },
        ]
    except Exception:
        return []


@app.get("/cmfs/{cmf_id}/matrices", tags=["cmfs"])
def get_cmf_matrices(cmf_id: int):
    """Return computed K-matrix entries (LaTeX) for a CMF."""
    db: Session = next(get_db())
    try:
        row = db.execute(text("""
            SELECT c.cmf_payload, r.canonical_payload
            FROM cmf c JOIN representation r ON r.id = c.representation_id
            WHERE c.id = :id
        """), {"id": cmf_id}).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")
        payload = _safe_json(row[0]) or {}
        canon   = _safe_json(row[1]) or {}
        matrices = _compute_matrices(
            payload.get("f_poly", ""),
            payload.get("fbar_poly", ""),
            canon,
        )
        if not matrices:
            raise HTTPException(404, "No matrix representation available for this CMF")
        return {"cmf_id": cmf_id, "matrices": matrices}
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
                   f.feature_json, c.category
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
            "primary_constant":        payload.get("primary_constant"),
            "identified_constant":     payload.get("identified_constant"),
            "identification_method":   payload.get("identification_method"),
            "identification_digits":   payload.get("identification_digits"),
            "identification_updated_at": payload.get("identification_updated_at"),
            "certification_level": payload.get("certification_level"),
            "source":              payload.get("source"),
            "source_category":     payload.get("source_category"),
            "matrix_size":         payload.get("matrix_size"),
            "agent_type":          payload.get("agent_type"),
            "total_score":         payload.get("total_score"),
            "best_delta":          payload.get("best_delta"),
            "flatness_verified":   payload.get("flatness_verified", False),
            "representation": {
                "id":                  row[5],
                "primary_group":       row[6],
                "canonical_fingerprint": row[7],
                "source_type":         canon.get("source_type"),
                "axes":                canon.get("axes"),
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
            "is_3d":           payload.get("dimension", row[1]) == 3 or _poly_has_z(payload.get("f_poly", "")),
            "symbolic_verification": payload.get("symbolic_verification"),
            "category":        row[15] or "reference",
            "walk_available":  bool(payload.get("f_poly")) or bool(canon.get("matrices")),
            "matrices": _compute_matrices(
                payload.get("f_poly", ""),
                payload.get("fbar_poly", ""),
                canon,
            ),
        }
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}/verify-steps", tags=["cmfs"])
def verify_steps(cmf_id: int):
    """
    Return symbolic flatness verification steps as LaTeX for a polynomial CMF.
    Computes K_x, K_y, the commutator K_x·K_y(k+1) − K_y·K_x(m+1), and checks = 0.
    No SageMath required — uses sympy only.
    """
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT cmf_payload FROM cmf WHERE id = :id"), {"id": cmf_id}
        ).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")

        payload  = _safe_json(row[0]) or {}
        f_poly   = payload.get("f_poly", "").strip()
        fbar_poly = payload.get("fbar_poly", "").strip()

        if not f_poly or not fbar_poly:
            raise HTTPException(400, "CMF has no polynomial form — verify-steps only works for polynomial CMFs")

        # Detect parametric (c0, c1, ...)
        import re as _re
        _extra = set(_re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', f_poly + ' ' + fbar_poly)) \
                 - set('xyz') - {'True','False','None','and','or','not'}
        if _extra:
            raise HTTPException(400, f"Parametric polynomial — free parameters {sorted(_extra)} must be fixed before verifying")

        is_3d = _poly_has_z(f_poly)
        k, m = _sp.symbols("k m", integer=True, positive=True)
        x, y = _sp.symbols("x y")

        steps = []

        if not is_3d:
            g    = _sympify(f_poly).subs([(x, k), (y, m)])
            gbar = _sympify(fbar_poly).subs([(x, k), (y, m)])
            b    = _expand(g.subs(m, 0) * gbar.subs(m, 0))
            b1   = _expand(b.subs(k, k + 1))
            a    = _expand(g - gbar.subs(k, k + 1))

            steps.append({"label": "Step 1 — Define g and ḡ",
                "latex": rf"g(k,m) = {_sp.latex(g)}, \quad \bar{{g}}(k,m) = {_sp.latex(gbar)}"})
            steps.append({"label": "Step 2 — Compute b(k) = g(k,0)·ḡ(k,0)",
                "latex": rf"b(k) = {_sp.latex(b)}"})
            steps.append({"label": "Step 3 — Compute a(k,m) = g(k,m) − ḡ(k+1,m)",
                "latex": rf"a(k,m) = {_sp.latex(a)}"})
            steps.append({"label": "Step 4 — K matrices",
                "latex": (
                    r"K_x(k,m) = \begin{pmatrix}0 & 1\\ b(k+1) & a(k,m)\end{pmatrix}, \quad "
                    r"K_y(k,m) = \begin{pmatrix}\bar{g}(k,m) & 1\\ b(k) & g(k,m)\end{pmatrix}"
                )})

            # Compute commutator symbolically
            K1 = _sp.Matrix([[0, 1], [b1, a]])
            K2 = _sp.Matrix([[gbar, 1], [b, g]])
            K2_k1 = K2.subs(k, k + 1)
            K1_m1 = _sp.Matrix([[0, 1], [b1, a.subs(m, m + 1)]])
            comm = _sp.simplify(K1 * K2_k1 - K2 * K1_m1)

            steps.append({"label": "Step 5 — Flatness condition to check",
                "latex": r"K_x(k,m)\cdot K_y(k{+}1,m) - K_y(k,m)\cdot K_x(k,m{+}1) \stackrel{?}{=} 0"})

            is_zero = all(e == 0 for e in comm)
            if is_zero:
                steps.append({"label": "Step 6 — Result",
                    "latex": r"= \mathbf{0} \quad \checkmark \text{ proved flat in } \mathbb{Q}[k,m]",
                    "result": "PASS"})
            else:
                nz = {(i,j): _sp.latex(_sp.expand(comm[i,j]))
                      for i in range(2) for j in range(2) if comm[i,j] != 0}
                steps.append({"label": "Step 6 — Result",
                    "latex": r"\neq \mathbf{0} \text{ — non-zero residual}",
                    "residual": {f"[{i},{j}]": v for (i,j),v in nz.items()},
                    "result": "FAIL"})

            existing_cert = payload.get("symbolic_verification")
            return {
                "cmf_id":    cmf_id,
                "is_3d":     False,
                "steps":     steps,
                "result":    "PASS" if is_zero else "FAIL",
                "certified": existing_cert is not None,
                "certified_at": existing_cert.get("verified_at") if existing_cert else None,
            }
        else:
            # 3D — return the theoretical steps but note that no formula is stored
            n = _sp.symbols("n", integer=True, positive=True)
            z = _sp.symbols("z")
            g    = _sympify(f_poly).subs([(x, k), (y, m), (z, n)])
            gbar = _sympify(fbar_poly).subs([(x, k), (y, m), (z, n)])
            b    = _expand(g.subs([(m, 0), (n, 0)]) * gbar.subs([(m, 0), (n, 0)]))
            b1   = _expand(b.subs(k, k + 1))
            a    = _expand(g - gbar.subs(k, k + 1))

            steps.append({"label": "Step 1 — Define g and ḡ (3D)",
                "latex": rf"g(k,m,n) = {_sp.latex(g)}, \quad \bar{{g}}(k,m,n) = {_sp.latex(gbar)}"})
            steps.append({"label": "Step 2 — b(k) = g(k,0,0)·ḡ(k,0,0)",
                "latex": rf"b(k) = {_sp.latex(b)}"})
            steps.append({"label": "Step 3 — Three flatness pairs to verify",
                "latex": (
                    r"K_x\cdot K_y(k{+}1,m,n) = K_y\cdot K_x(k,m{+}1,n) \quad \text{(Kx/Ky)}\\"
                    r"K_x\cdot K_z(k{+}1,m,n) = K_z\cdot K_x(k,m,n{+}1) \quad \text{(Kx/Kz)}\\"
                    r"K_y\cdot K_z(k,m{+}1,n) = K_z\cdot K_y(k,m,n{+}1) \quad \text{(Ky/Kz)}"
                )})
            steps.append({"label": "Note",
                "latex": r"\text{3D CMF Hunter entries use internal matrices not stored in the Atlas — symbolic certificate not yet available}",
                "result": "PENDING"})

            return {
                "cmf_id": cmf_id,
                "is_3d":  True,
                "steps":  steps,
                "result": "PENDING",
                "certified": False,
                "certified_at": None,
            }
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}/walk", tags=["cmfs"])
def walk_cmf(
    cmf_id: int,
    depth: int = Query(100, ge=10, le=2000),
    m_fixed: int = Query(0, ge=0, le=50),
    n_fixed: int = Query(1, ge=1, le=50),
    k_fixed: int = Query(1, ge=1, le=50),
    direction: str = Query("x"),
    k_start_override: int = Query(-1, ge=-1, le=200),
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

        dir_clean = direction.lower().strip() if direction else "x"
        if dir_clean not in ("x", "y", "z"):
            dir_clean = "x"

        # ── Fallback: RamanujanTools / known_family stored-expression walk ──
        stored_walk = None
        stored_axes = None
        stored_size = 2
        if not f_poly or not fbar_poly:
            rep_row = db.execute(
                text("SELECT r.canonical_payload FROM cmf c JOIN representation r ON r.id=c.representation_id WHERE c.id=:id"),
                {"id": cmf_id},
            ).fetchone()
            if rep_row:
                canon = _safe_json(rep_row[0]) or {}
                mats  = canon.get("matrices", {})
                axes  = canon.get("axes", [])
                if mats and axes:
                    stored_axes = axes
                    result = _build_matrix_walk_from_stored(mats, axes, dir_clean)
                    if result:
                        stored_walk, _, stored_size = result
            if not stored_walk:
                raise HTTPException(422, "CMF has no polynomial form — walk not available")

        is_3d = False
        if stored_walk:
            # RamanujanTools stored-expression walk
            is_3d = len(stored_axes or []) >= 3
            dim_dirs = {"x": 0, "y": 1, "z": 2}
            step_idx = dim_dirs.get(dir_clean, 0)
            fixed_axes = [ax for i, ax in enumerate(stored_axes) if i != step_idx]
            fixed_vals_list = [m_fixed, n_fixed]  # default fixed values

            def _K(step):
                fv = {ax: fixed_vals_list[i] for i, ax in enumerate(fixed_axes[:2])}
                return stored_walk(step, fv)

            P = mpmath.eye(stored_size)
            k_start = 1
            # Find first non-degenerate step (use last col for NxN generality)
            _nc_s = stored_size - 1
            _nr_s = stored_size - 2
            for _ks in range(1, 20):
                try:
                    _m = _K(_ks)
                    if abs(float(mpmath.re(_m[_nr_s, _nc_s]))) > 1e-15 or abs(float(mpmath.re(_m[_nc_s, _nc_s]))) > 1e-15:
                        k_start = _ks
                        break
                except Exception:
                    pass
        else:
            try:
                walk_fns = _build_walk_fns(f_poly, fbar_poly)
            except Exception as exc:
                raise HTTPException(500, f"Polynomial parse error: {exc}")

            Kx_fn, Ky_fn, Kz_fn, is_3d = walk_fns
            P = mpmath.eye(2)

            if dir_clean == "z" and not is_3d:
                raise HTTPException(422, "Kz walk requires a 3D CMF")
            if dir_clean == "y" and Ky_fn is None:
                raise HTTPException(422, "Ky walk not available for this CMF")

            k_start = 0
            if dir_clean == "y":
                def _K(step): return Ky_fn(k_fixed, step, n_fixed)
                k_start = 1
            elif dir_clean == "z":
                def _K(step): return Kz_fn(k_fixed, m_fixed, step)
                for _s in range(1, 30):
                    try:
                        _b = abs(float(mpmath.re(Kz_fn(k_fixed, m_fixed, _s)[1, 0])))
                        if _b > 1e-10:
                            k_start = _s
                            break
                    except Exception:
                        pass
                if k_start == 0:
                    k_start = 1
            else:
                def _K(step): return Kx_fn(step, m_fixed, n_fixed)
                if is_3d:
                    for _k in range(30):
                        try:
                            _b = abs(float(mpmath.re(Kx_fn(_k, 0, n_fixed)[1, 0])))
                            if _b > 1e-10:
                                k_start = max(0, _k - 1)
                                break
                        except Exception:
                            pass

        mpmath.mp.dps = 30
        sequence = []

        if k_start_override >= 0:
            k_start = k_start_override
        for _iter, step in enumerate(range(k_start, k_start + depth)):
            try:
                K = _K(step)
                P = P * K
                # Normalize every 20 iterations to prevent overflow
                _sz = P.rows
                _nc = _sz - 1          # last col index
                _nr = _sz - 2          # second-to-last row index
                if (_iter + 1) % 20 == 0:
                    scale = max(abs(float(mpmath.re(P[_nr, _nc]))),
                                abs(float(mpmath.re(P[_nc, _nc]))), 1e-300)
                    P = P / scale
                denom = P[_nc, _nc]
                numer = P[_nr, _nc]
                if mpmath.fabs(denom) > mpmath.mpf("1e-200"):
                    val = float(mpmath.re(numer / denom))
                    sequence.append({"step": step + 1, "value": val if math.isfinite(val) else None})
                else:
                    sequence.append({"step": step + 1, "value": None})
            except Exception:
                sequence.append({"step": step + 1, "value": None})

        finite = [s["value"] for s in sequence if s["value"] is not None]
        best = finite[-1] if finite else None

        # Convergence rate — self-delta (bits per step)
        conv_rate = None
        self_delta = None
        if best is not None and len(finite) >= 20:
            diffs = [abs(v - best) for v in finite[-20:] if abs(v - best) > 0]
            if len(diffs) >= 2:
                try:
                    conv_rate = round(
                        (math.log10(diffs[-1]) - math.log10(diffs[0])) / len(diffs), 3
                    )
                except Exception:
                    pass
        # self_delta: bits of new precision gained per step (milestone method)
        if len(finite) >= 6:
            n = len(finite)
            e1 = abs(finite[n//3] - finite[-1]) if abs(finite[n//3] - finite[-1]) > 0 else None
            e2 = abs(finite[2*n//3] - finite[-1]) if abs(finite[2*n//3] - finite[-1]) > 0 else None
            if e1 and e2 and e2 < e1:
                try:
                    import math as _m
                    self_delta = round(_m.log2(e1 / e2) / (n // 3), 4)
                except Exception:
                    pass

        walk_note = None
        if is_3d:
            walk_note = (
                "3D CMF walk uses the telescope formula with b(k,n)=g(k,0,n)·ḡ(k,0,n). "
                "CMF Hunter 3D entries use a different internal matrix construction, so "
                "the walk estimate may not converge to the certified constant. "
                "The certified value is stored in primary_constant."
            )

        return {
            "cmf_id":          cmf_id,
            "depth":           depth,
            "direction":        dir_clean,
            "k_start":         k_start,
            "k_fixed":         k_fixed if dir_clean != "x" else None,
            "m_fixed":         m_fixed,
            "n_fixed":         n_fixed,
            "is_3d":           is_3d,
            "f_poly":          f_poly,
            "fbar_poly":       fbar_poly,
            "sequence":        sequence,
            "best_estimate":   best,
            "conv_rate_log10_per_step": conv_rate,
            "self_delta_bits_per_step": self_delta,
            "constant_matches": _compare_constants(best) if best is not None else [],
            "primary_constant": payload.get("primary_constant"),
            "certification_level": payload.get("certification_level"),
            "walk_note":       walk_note,
        }
    finally:
        db.close()


@app.get("/cmfs/{cmf_id}/conservative-test", tags=["cmfs"])
def conservative_test(
    cmf_id: int,
    k_max: int = Query(5, ge=1, le=15),
    m_max: int = Query(5, ge=1, le=15),
    n_fixed: int = Query(1, ge=0, le=50),
):
    """
    Empirical flatness (path-independence) test.
    2D: checks Kx(k,m)*Ky(k+1,m) = Ky(k,m)*Kx(k,m+1) on a (k,m) grid.
    3D: additionally checks Kx/Kz and Ky/Kz flatness with n=n_fixed.
    Numerical evidence only — not a formal proof.
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
            walk_fns = _build_walk_fns(f_poly, fbar_poly)
        except Exception as exc:
            raise HTTPException(500, f"Polynomial parse error: {exc}")

        Kx_fn, Ky_fn, Kz_fn, is_3d = walk_fns

        mpmath.mp.dps = 20
        test_points = []
        max_residual = 0.0

        def _frob(M):
            sz = M.rows
            return float(mpmath.sqrt(sum(
                mpmath.fabs(M[i, j])**2 for i in range(sz) for j in range(sz)
            )))

        def _test_pair(Ma_fn, Mb_fn, pt, label):
            """Check Ma(pt)*Mb(pt+e_a) == Mb(pt)*Ma(pt+e_b) numerically."""
            nonlocal max_residual
            k_, m_, n_ = pt
            if label == 'km':
                lhs = Ma_fn(k_, m_, n_) * Mb_fn(k_+1, m_, n_)
                rhs = Mb_fn(k_, m_, n_) * Ma_fn(k_, m_+1, n_)
            elif label == 'kn':
                lhs = Ma_fn(k_, m_, n_) * Mb_fn(k_+1, m_, n_)
                rhs = Mb_fn(k_, m_, n_) * Ma_fn(k_, m_, n_+1)
            else:  # 'mn'
                lhs = Ma_fn(k_, m_, n_) * Mb_fn(k_, m_+1, n_)
                rhs = Mb_fn(k_, m_, n_) * Ma_fn(k_, m_, n_+1)
            diff = lhs - rhs
            res = _frob(diff)
            max_residual = max(max_residual, res)
            return res

        # Kx/Ky flatness — test on (k,m) grid with n=n_fixed
        for k in range(1, k_max + 1):
            for m in range(1, m_max + 1):
                try:
                    res = _test_pair(Kx_fn, Ky_fn, (k, m, n_fixed), 'km')
                    test_points.append({
                        "pair": "Kx/Ky", "k": k, "m": m, "n": n_fixed,
                        "residual": res,
                        "log10_residual": round(math.log10(res), 2) if res > 0 else -20,
                    })
                except Exception as exc:
                    test_points.append({"pair": "Kx/Ky", "k": k, "m": m,
                                        "n": n_fixed, "residual": None, "error": str(exc)})

        if is_3d and Kz_fn is not None:
            # Kx/Kz flatness — test on (k,n) grid with m=1
            for k in range(1, k_max + 1):
                for n in range(1, m_max + 1):
                    try:
                        res = _test_pair(Kx_fn, Kz_fn, (k, 1, n), 'kn')
                        test_points.append({
                            "pair": "Kx/Kz", "k": k, "m": 1, "n": n,
                            "residual": res,
                            "log10_residual": round(math.log10(res), 2) if res > 0 else -20,
                        })
                    except Exception as exc:
                        test_points.append({"pair": "Kx/Kz", "k": k, "m": 1,
                                            "n": n, "residual": None, "error": str(exc)})
            # Ky/Kz flatness — test on (m,n) grid with k=2
            for m in range(1, m_max + 1):
                for n in range(1, m_max + 1):
                    try:
                        res = _test_pair(Ky_fn, Kz_fn, (2, m, n), 'mn')
                        test_points.append({
                            "pair": "Ky/Kz", "k": 2, "m": m, "n": n,
                            "residual": res,
                            "log10_residual": round(math.log10(res), 2) if res > 0 else -20,
                        })
                    except Exception as exc:
                        test_points.append({"pair": "Ky/Kz", "k": 2, "m": m,
                                            "n": n, "residual": None, "error": str(exc)})

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
            "is_3d":          is_3d,
            "f_poly":         f_poly,
            "fbar_poly":      fbar_poly,
            "test_points":    test_points,
            "max_residual":   max_residual,
            "verdict":        verdict,
            "k_max":          k_max,
            "m_max":          m_max,
            "n_fixed":        n_fixed if is_3d else None,
            "note": (
                "3D CMF: checks Kx/Ky, Kx/Kz, Ky/Kz using the telescope formula extended to 3 variables. "
                "CMF Hunter 3D entries use a different internal matrix construction (not stored), "
                "so large residuals here do not invalidate the CMF's own B-level numerical certification. "
                if is_3d else
                "Checks Kx(k,m)\u00b7Ky(k+1,m) = Ky(k,m)\u00b7Kx(k,m+1) numerically. "
            ) + "Small residual is empirical evidence of path independence, not a formal proof.",
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


# --- CMF Relations (Sprint 3) ---

@app.get("/cmfs/{cmf_id}/relations", tags=["cmfs"])
def cmf_relations(cmf_id: int):
    """Relational context for a CMF: linked constant, source family, related entries, proof artifact."""
    db: Session = next(get_db())
    try:
        row = db.execute(
            text("SELECT id, dimension, cmf_payload FROM cmf WHERE id = :id"),
            {"id": cmf_id},
        ).fetchone()
        if not row:
            raise HTTPException(404, "CMF not found")

        payload = json.loads(row[2]) if row[2] else {}
        primary_const = payload.get("primary_constant")
        identified_const = payload.get("identified_constant")
        source_cat = payload.get("source_category") or payload.get("source", "")
        cert = payload.get("certification_level")

        _proof_map = {
            "A_plus": "symbolically_certified",
            "A_certified": "verified",
            "B_verified_numeric": "numeric_only",
            "C_scouting": "unverified",
        }

        linked_const = None
        for c in _CONSTANTS_REGISTRY:
            all_labels = [c["label"]] + c.get("aliases", [])
            if primary_const and (primary_const in all_labels or primary_const == c["id"]):
                linked_const = {
                    "id": c["id"], "label": c["label"], "latex": c["latex"],
                    "irrationality_status": c.get("irrationality_status"),
                    "transcendence_status": c.get("transcendence_status"),
                    "constants_page_uri": f"https://davidvesterlund.com/cmf-atlas/constants.html#{c['id']}",
                }
                break

        related: list = []
        if primary_const:
            rel_rows = db.execute(text(
                "SELECT id, dimension, json_extract(cmf_payload,'$.certification_level') "
                "FROM cmf WHERE json_extract(cmf_payload,'$.primary_constant') = :pc "
                "AND id != :cid AND dimension >= 2 ORDER BY id LIMIT 5"
            ), {"pc": primary_const, "cid": cmf_id}).fetchall()
            related = [
                {"id": r[0], "dimension": r[1], "certification_level": r[2],
                 "entry_uri": f"https://davidvesterlund.com/cmf-atlas/entry.html?id={r[0]}"}
                for r in rel_rows
            ]

        return {
            "cmf_id": cmf_id,
            "entry_uri": f"https://davidvesterlund.com/cmf-atlas/entry.html?id={cmf_id}",
            "schema_version": "2.4",
            "primary_constant": primary_const,
            "identified_constant": identified_const,
            "linked_constant": linked_const,
            "source_family": source_cat,
            "proof_status": _proof_map.get(cert, "unverified"),
            "identification_status": (
                "pslq_identified" if identified_const else
                "matched" if primary_const else "unidentified"
            ),
            "proof_artifact": payload.get("symbolic_verification"),
            "related_entries": related,
        }
    finally:
        db.close()


# --- API v1 versioned aliases (Sprint 3) ---

@app.api_route("/api/v1/{path:path}", methods=["GET"], tags=["versioned"],
               summary="Versioned API alias",
               description="/api/v1/X is a stable alias for /X. Redirects 307 for forward compatibility.")
async def v1_proxy(path: str, request: Request):
    qs = str(request.url.query)
    target = f"/{path}?{qs}" if qs else f"/{path}"
    return RedirectResponse(url=target, status_code=307)


# ---------------------------------------------------------------------------
# Serve frontend static files (must be last — catches all unmatched routes)
# ---------------------------------------------------------------------------
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
