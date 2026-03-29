"""
import_ramanujantools.py
========================
Pull every CMF from ramanujantools.cmf.known_cmfs, convert to the atlas
schema, and upsert into atlas_2d.db.

New entries get:
  source            = "ramanujantools"
  source_category   = "RamanujanTools"
  certification_level = "A_plus"
  flatness_verified = True

Existing entries (matched by fingerprint) are updated in-place.
"""

import hashlib, json, sqlite3, sys
from pathlib import Path
from datetime import datetime, timezone

import sympy as sp
from ramanujantools import Matrix
from ramanujantools.cmf import CMF
from ramanujantools.cmf.ffbar import FFbar
import ramanujantools.cmf.known_cmfs as kc

DB = Path("data/atlas_2d.db")

# ── helpers ──────────────────────────────────────────────────────────────────

def mat_to_str(mat) -> str:
    """Convert a ramanujantools Matrix (or list-of-lists) to a storable string."""
    if hasattr(mat, "tolist"):
        rows = mat.tolist()
    else:
        rows = list(mat)
    return str([[str(e) for e in r] for r in rows])


def fingerprint(name: str, axes_str: str) -> str:
    h = hashlib.sha256(f"{name}|{axes_str}".encode()).hexdigest()
    return h[:16]


def poly_str(expr) -> str:
    """Sympy expr → Python-syntax string (using x, y as variables)."""
    return str(sp.expand(expr)) if expr is not None else ""


# ── CMF definitions ──────────────────────────────────────────────────────────
# Each tuple: (name, primary_constant, certification, flatness, call_fn)
# For FFbar CMFs with parameter symbols, we store the symbolic f/fbar polys.

CMFS = [
    # ── Concrete (explicit matrices, no free params) ──────────────────────
    ("e",                    "e (Euler)",       "A_plus", True,  kc.e),
    ("pi",                   "pi",              "A_plus", True,  kc.pi),
    ("symmetric_pi",         "pi",              "A_plus", True,  kc.symmetric_pi),
    ("zeta3",                "zeta(3)",         "A_plus", True,  kc.zeta3),
    ("hypergeometric_2F1",   "zeta(2) / pi^2",  "A_plus", True,  kc.hypergeometric_derived_2F1),
    # ── Parametric FFbar families ─────────────────────────────────────────
    ("cmf1_linear",          None,              "A_plus", True,  kc.cmf1),
    ("cmf2_quadratic",       None,              "A_plus", True,  kc.cmf2),
    ("cmf3_1",               None,              "A_plus", True,  kc.cmf3_1),
    ("cmf3_2",               None,              "A_plus", True,  kc.cmf3_2),
    ("cmf3_3",               None,              "A_plus", True,  kc.cmf3_3),
    ("var_root_cmf",         None,              "A_plus", True,  kc.var_root_cmf),
]

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_or_create_project(cur, name="RamanujanTools"):
    r = cur.execute("SELECT id FROM project WHERE name=?", (name,)).fetchone()
    if r:
        return r[0]
    cur.execute("INSERT INTO project (name, created_at) VALUES (?,?)",
                (name, datetime.now(timezone.utc).isoformat()))
    return cur.lastrowid


def get_or_create_series(cur, project_id, series_name):
    r = cur.execute("SELECT id FROM series WHERE name=?", (series_name,)).fetchone()
    if r:
        return r[0]
    cur.execute("""INSERT INTO series (project_id, name, definition, generator_type, provenance, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (project_id, series_name, series_name, "known_family",
                 "https://github.com/RamanujanMachine/RamanujanTools",
                 datetime.now(timezone.utc).isoformat()))
    return cur.lastrowid


def upsert_repr(cur, series_id, fp16, canon_payload_dict):
    canon_str = json.dumps(canon_payload_dict)
    r = cur.execute("SELECT id FROM representation WHERE canonical_fingerprint=?", (fp16,)).fetchone()
    if r:
        cur.execute("UPDATE representation SET canonical_payload=? WHERE id=?", (canon_str, r[0]))
        return r[0]
    cur.execute("""INSERT INTO representation (series_id, primary_group, canonical_fingerprint,
                   canonical_payload, created_at) VALUES (?,?,?,?,?)""",
                (series_id, "known_family", fp16, canon_str,
                 datetime.now(timezone.utc).isoformat()))
    return cur.lastrowid


def upsert_cmf(cur, repr_id, cmf_payload_dict, dimension):
    fp16 = cmf_payload_dict.get("fingerprint", "")
    # Try to match existing by fingerprint stored in cmf_payload
    rows = cur.execute("SELECT id FROM cmf WHERE representation_id=?", (repr_id,)).fetchall()
    cmf_payload_str = json.dumps(cmf_payload_dict)
    if rows:
        cid = rows[0][0]
        cur.execute("UPDATE cmf SET cmf_payload=?, dimension=? WHERE id=?",
                    (cmf_payload_str, dimension, cid))
        return cid, "updated"
    cur.execute("""INSERT INTO cmf (representation_id, dimension, cmf_payload,
                   direction_policy, created_at) VALUES (?,?,?,?,?)""",
                (repr_id, dimension, cmf_payload_str, "diagonal",
                 datetime.now(timezone.utc).isoformat()))
    return cur.lastrowid, "inserted"


# ── Main ─────────────────────────────────────────────────────────────────────

con = sqlite3.connect(DB)
cur = con.cursor()

project_id = get_or_create_project(cur)
series_id  = get_or_create_series(cur, project_id, "ramanujantools_known")

results = []

for (cmf_name, primary_const, cert, flatness, fn) in CMFS:
    try:
        cmf_obj = fn()
    except Exception as ex:
        print(f"SKIP {cmf_name}: could not instantiate — {ex}")
        continue

    mats = cmf_obj.matrices  # dict: axis_symbol → Matrix
    axes = list(mats.keys())
    dim  = len(axes)

    # Build the matrices dict for storage: axis_str → list-of-lists-of-strings
    mats_dict = {}
    for ax, mat in mats.items():
        rows_raw = mat.tolist() if hasattr(mat, "tolist") else list(mat)
        mats_dict[str(ax)] = str([[str(sp.simplify(e)) for e in r] for r in rows_raw])

    # For FFbar CMFs: extract f_poly and fbar_poly
    f_poly_str    = ""
    fbar_poly_str = ""
    if isinstance(cmf_obj, FFbar):
        try:
            f_poly_str    = poly_str(cmf_obj.f)
            fbar_poly_str = poly_str(cmf_obj.fbar)
        except Exception:
            pass

    # Compute degree from f_poly if available
    degree = 0
    if f_poly_str:
        try:
            x, y = sp.symbols("x y")
            expr = sp.sympify(f_poly_str)
            degree = sp.total_degree(expr, [x, y])
        except Exception:
            pass

    fp16 = fingerprint(cmf_name, "|".join(str(a) for a in axes))

    canon_payload = {
        "source_type":  "known_family",
        "axes":         [str(a) for a in axes],
        "dimension":    dim,
        "matrices":     mats_dict,
        "f_poly":       f_poly_str,
        "fbar_poly":    fbar_poly_str,
        "fingerprint":  fp16,
    }

    cmf_payload = {
        "certification_level": cert,
        "degree":              degree,
        "dimension":           dim,
        "f_poly":              f_poly_str,
        "fbar_poly":           fbar_poly_str,
        "flatness_verified":   flatness,
        "primary_constant":    primary_const,
        "source":              "ramanujantools",
        "source_category":     "RamanujanTools",
        "description":         f"CMF from RamanujanTools known_cmfs.{cmf_name}()",
        "fingerprint":         fp16,
        "cmf_name":            cmf_name,
    }

    repr_id      = upsert_repr(cur, series_id, fp16, canon_payload)
    cmf_id, act  = upsert_cmf(cur, repr_id, cmf_payload, dim)
    results.append((cmf_id, act, cmf_name, dim, primary_const))
    print(f"  [{act}] id={cmf_id}  {cmf_name}  dim={dim}  const={primary_const}")

con.commit()
con.execute("VACUUM")
con.close()

print(f"\nDone. {len(results)} CMFs processed.")
total = sqlite3.connect(DB).execute("SELECT COUNT(*) FROM cmf").fetchone()[0]
print(f"Total CMFs in atlas: {total}")
