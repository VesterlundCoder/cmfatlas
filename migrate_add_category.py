"""
migrate_add_category.py
=======================
Adds `category` column to the `cmf` table and auto-classifies every entry.

Categories
----------
  reference   — Trivial / small 2D CMFs with rational or uninteresting limits.
                (linear polynomials, rational constant, degree ≤ 2 with no
                 named transcendental constant)
  interesting — 2D CMFs that converge to a known transcendental constant
                (ζ(n), π, ln(2), Catalan G, digamma ψ(p/q), e, etc.) or
                to an unknown/unidentified irrational.
  discovery   — Higher-dimensional CMFs (dim ≥ 3), Ramanujan/pFq families,
                and any A+ manually verified entries.

Run once:
  python migrate_add_category.py
"""

import re
import sqlite3
import json
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "atlas_2d.db"

# ── Patterns that signal a non-trivial transcendental constant ──────────────
TRANSCENDENTAL_PATTERNS = [
    r"zeta", r"pi", r"\bpi\b", r"log", r"ln\(", r"digamma", r"psi\(",
    r"catalan", r"euler", r"\be\b", r"li_", r"polylog", r"harmonic",
    r"gamma", r"G\b", r"eta\(", r"hurwitz", r"dirichlet", r"3f2",
    r"hypergeometric", r"2f1", r"pFq", r"identified",
]
_TRANS_RE = re.compile("|".join(TRANSCENDENTAL_PATTERNS), re.IGNORECASE)

# ── Patterns that are clearly rational (e.g. "-4/2", "1/1", "3/5") ─────────
_RATIONAL_RE = re.compile(r"^-?\s*\d+\s*/\s*\d+\s*$")

# ── Source strings that indicate discovery-tier ──────────────────────────────
DISCOVERY_SOURCES = ["ramanujantools", "pfq", "3f2", "hypergeometric", "discovery"]


def classify(payload: dict, dimension: int) -> str:
    """Return 'reference', 'interesting', or 'discovery' for one CMF."""
    cert = (payload.get("certification_level") or "").lower()
    src  = (payload.get("source_category") or payload.get("source") or "").lower()
    pc   = (payload.get("primary_constant") or "").strip()
    deg  = int(payload.get("degree") or 0)

    # ── discovery ────────────────────────────────────────────────────────────
    if dimension >= 3:
        return "discovery"
    if any(s in src for s in DISCOVERY_SOURCES):
        return "discovery"
    if cert == "a_plus":
        return "discovery"

    # ── reference ────────────────────────────────────────────────────────────
    if deg <= 1:
        return "reference"
    if not pc or pc.lower() in ("none", "", "null", "—", "-"):
        if deg <= 2:
            return "reference"
    if _RATIONAL_RE.match(pc):
        return "reference"
    # Catch patterns like "1/ln(2) (CF limit)" — contains transcendental?
    if pc and not _TRANS_RE.search(pc):
        # No transcendental signal → reference
        return "reference"

    # ── interesting ──────────────────────────────────────────────────────────
    return "interesting"


def run():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Add column if missing
    cols = [r[1] for r in cur.execute("PRAGMA table_info(cmf)").fetchall()]
    if "category" not in cols:
        cur.execute("ALTER TABLE cmf ADD COLUMN category TEXT")
        print("Added column `category` to cmf table.")
    else:
        print("Column `category` already exists.")

    # Create index for fast filtering
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_cmf_category
        ON cmf(category)
    """)

    # Classify all rows
    rows = cur.execute(
        "SELECT id, dimension, cmf_payload FROM cmf"
    ).fetchall()

    counts = {"reference": 0, "interesting": 0, "discovery": 0}
    for cmf_id, dim, pay_str in rows:
        try:
            payload = json.loads(pay_str) if pay_str else {}
        except Exception:
            payload = {}
        cat = classify(payload, dim or 2)
        cur.execute("UPDATE cmf SET category = ? WHERE id = ?", (cat, cmf_id))
        counts[cat] += 1

    conn.commit()
    conn.close()

    print(f"\nClassified {sum(counts.values())} CMFs:")
    for cat, n in counts.items():
        print(f"  {cat:12s}: {n}")


if __name__ == "__main__":
    run()
