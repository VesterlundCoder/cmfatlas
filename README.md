# CMF Atlas

**A public mathematical database of Conservative Matrix Fields (CMFs).**

> Live site: [davidvesterlund.com/cmfatlas](https://davidvesterlund.com/cmfatlas)

CMF Atlas is an open research database and interactive web platform for exploring
Conservative Matrix Fields — 2D recurrences with path-independent matrix products
that encode connections between continued fractions, hypergeometric series, and
transcendental constants such as ζ(3), π, ln(2), and Catalan's constant.

[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Contents

- **5,585 CMF entries** across dimensions 1–4
- **4,108 walkable entries** with explicit polynomial form f(x,y)
- **1,448 A-certified** formally verified entries
- Sources: Telescope search, Euler2AI, CMF Hunter, Targeted 2D, Known families (Apéry, ln(2), …)

---

## What is a Conservative Matrix Field?

A CMF consists of matrix-valued functions K₁(k,m) and K₂(k,m) satisfying the **flatness condition**:

```
K₁(k,m) · K₂(k+1, m) = K₂(k,m) · K₁(k, m+1)
```

This means products along any admissible lattice path from (0,0) to (K,M) agree —
the CMF is *path-independent*. For the telescope polynomial family with f(x,y) and conjugate f̄(x,y):

```
g(k,m) = f(k,m),     ḡ(k,m) = f̄(k,m)
b(k)   = g(k,0)·ḡ(k,0)
a(k,m) = g(k,m) - ḡ(k+1, m)

K₁(k,m) = [[0,       1   ],      K₂(k,m) = [[ḡ(k,m), 1  ],
            [b(k+1),  a(k,m)]]               [b(k),   g(k,m)]]
```

**Example — Apéry's ζ(3):** `f(x,y) = x³ + 2x²y + 2xy² + y³ = (x+y)³`

The matrix walk P_N = K₁(0,0)·K₁(1,0)·…·K₁(N-1,0) converges:
`P_N[0,1] / P_N[1,1] → ζ(3) ≈ 1.2020569…`

---

## Web Platform

The CMF Atlas web platform provides six interactive pages:

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/app/index.html` | Summary statistics, dimension breakdown, featured entries |
| Browse | `/app/browse.html` | Searchable, filterable index of all CMFs |
| Entry | `/app/entry.html?id=N` | Full CMF page with matrix form, metadata, BibTeX |
| Explorer | `/app/explorer.html` | Interactive K₁ matrix walk (depth 100/200/500) |
| Conservative Test | `/app/conservative-test.html` | Empirical path-independence test |
| About | `/app/about.html` | Project info, API docs, citation, license |

---

## Running Locally

### Requirements

- Python 3.11+
- The database file `data/atlas.db` (SQLite)

### Setup

```bash
cd cmf_atlas
python -m venv venv
source venv/bin/activate
pip install -r requirements-api.txt
```

### Start the server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The web platform is at **http://localhost:8000/app/**  
The REST API is at **http://localhost:8000/**  
Swagger docs at **http://localhost:8000/docs**

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats` | Basic aggregate stats |
| GET | `/stats/detailed` | Dashboard stats (dimension/cert/source breakdown) |
| GET | `/cmfs` | Paginated CMF list |
| GET | `/cmfs/browse` | Search & filter CMFs (q, dimension, degree, certification, …) |
| GET | `/cmfs/{id}` | Basic CMF entry |
| GET | `/cmfs/{id}/full` | Full entry with representation, features, series |
| GET | `/cmfs/{id}/walk` | K₁ matrix walk (`depth`, `m_fixed` params) |
| GET | `/cmfs/{id}/conservative-test` | Empirical flatness test (`k_max`, `m_max` params) |
| GET | `/representations/{id}` | Representation detail |
| GET | `/series/{id}` | Series detail |
| GET | `/search` | Full-text search |

**Walk endpoint example:**
```bash
curl "http://localhost:8000/cmfs/1/walk?depth=200&m_fixed=0"
# Returns partial convergents converging to ζ(3), with constant proximity matches
```

**Conservative test example:**
```bash
curl "http://localhost:8000/cmfs/1/conservative-test?k_max=5&m_max=5"
# Returns residuals for K₁(k,m)·K₂(k+1,m) - K₂(k,m)·K₁(k,m+1) at each grid point
```

---

## Database Schema

The SQLite database `data/atlas.db` has these key tables:

```
cmf            — Core entries: cmf_payload (JSON), dimension, representation_id
representation — Canonical forms: canonical_payload (JSON), fingerprint, primary_group
series         — Provenance: generator_type, definition, name
features       — Computed: feature_json (complexity, d_finite_rank, stability, …)
eval_run       — Numerical evaluation results (limit_estimate, convergence_score, …)
recognition_attempt — PSLQ/ISC attempts with identified_as, residual_log10
```

Key fields in `cmf_payload` JSON:
- `f_poly`, `fbar_poly` — SymPy-syntax polynomial strings
- `degree` — total polynomial degree
- `primary_constant` — identified limit (e.g., `"zeta(3)"`, `"ln(2)"`)
- `certification_level` — `"A_certified"` / `"B_verified_numeric"` / `"C_scouting"`
- `flatness_verified` — boolean

---

## Ramanujan Dreams Integration

CMF Atlas polynomial entries are compatible with the
[Ramanujan Dreams](https://github.com/RamanujanMachine/RamanujanTools) pipeline.

```python
# Fetch walkable CMFs from the API
import requests
r = requests.get("http://localhost:8000/cmfs/browse", params={"has_formula": "true", "limit": 50})
entries = r.json()["items"]

# Each entry has f_poly / fbar_poly ready for Dreams-style evaluation
for cmf in entries:
    print(cmf["id"], cmf["f_poly"], "->", cmf["primary_constant"])
```

---

## Project Structure

```
cmf_atlas/
  api.py               — FastAPI backend (all endpoints + math helpers)
  requirements-api.txt — Dependencies: fastapi, uvicorn, sqlalchemy, sympy, mpmath
  frontend/
    index.html         — Dashboard with stats + featured entries
    browse.html        — Search/filter all CMFs
    entry.html         — Single CMF page (matrix form, BibTeX, links)
    explorer.html      — Interactive matrix walk explorer
    conservative-test.html — Empirical path-independence tester
    about.html         — Project info, API reference, citation
  data/
    atlas.db           — SQLite database (5,585 CMF entries)
  verify_truly_2d.py   — Symbolic + numerical flatness verification (SymPy/mpmath)
```

---

## Citation

If you use CMF Atlas in academic work, please cite:

```bibtex
@misc{vesterlund2026cmfatlas,
  author  = {Vesterlund, David},
  title   = {{CMF Atlas}: A Database of Conservative Matrix Fields},
  year    = {2026},
  url     = {https://davidvesterlund.com/cmfatlas},
  note    = {Version 2.2, open research dataset},
  license = {CC BY 4.0}
}
```

Each individual CMF entry page provides a pre-filled BibTeX citation block.

---

## License

**Dataset:** [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
**Code:** MIT

---

## Tech Stack

Python 3.11, FastAPI, SQLAlchemy, SymPy, mpmath · Vanilla JS, Tailwind CSS, Chart.js, MathJax 3
