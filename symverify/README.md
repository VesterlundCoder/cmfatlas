# CMF Atlas — Symbolic Flatness Verifier

Checks the flatness condition

```
K₁(k,m) · K₂(k+1, m)  =  K₂(k,m) · K₁(k, m+1)
```

for every CMF in `data/atlas_2d.db` using **exact symbolic arithmetic in SageMath**.

## Requirements

- SageMath ≥ 10.0  (`which sage`)
- The `data/atlas_2d.db` database (relative to project root)

## Quick start

```bash
cd symverify/
chmod +x run_all.sh

# Smoke test — 5 entries
sage verify_cmfs.sage --limit 5

# All polynomial CMFs (fast, ~2–5 min for 355 entries)
sage verify_cmfs.sage --mode poly

# RamanujanTools entries (explicit matrices, SR arithmetic)
sage verify_cmfs.sage --source RamanujanTools

# Single entry by id
sage verify_cmfs.sage --id 1

# Full run with logging
./run_all.sh

# Full run, polynomial only
./run_all.sh --mode poly
```

## Verification modes

### Mode 1 — Polynomial CMFs (`--mode poly`)

For CMFs with `f_poly` and `fbar_poly`, builds the canonical 2×2 matrices:

```
K₁(k,m) = ⎡    0       1   ⎤
           ⎣  b(k+1)  a(k,m)⎦

K₂(k,m) = ⎡ ḡ(k,m)    1   ⎤
           ⎣  b(k)   g(k,m) ⎦

where  g = f_poly,  ḡ = fbar_poly
       b(k) = g(k,0) · ḡ(k,0)
       a(k,m) = g(k,m) − ḡ(k+1, m)
```

Computes `K₁·K₂(k+1,m) − K₂·K₁(k,m+1)` in `QQ[k,m]` and checks every entry
is the zero polynomial.  This is **100% exact** — no floating point involved.

### Mode 2 — Explicit matrix CMFs (`--mode explicit`)

For CMFs with stored matrix dictionaries (RamanujanTools, hypergeometric families),
uses SageMath's Symbolic Ring.  For each pair of axes (i, j):

```
K_i(n) · K_j(n + eᵢ)  =  K_j(n) · K_i(n + eⱼ)
```

Uses `simplify_rational()` to reduce each residual entry to zero.

## Output

Reports are written to `symverify/reports/`:

- `verification_YYYYMMDD_HHMMSS.json` — full machine-readable results
- `verification_YYYYMMDD_HHMMSS.txt`  — human-readable summary

Each entry records:

| Field | Description |
|-------|-------------|
| `id` | CMF id |
| `result` | `PASS`, `FAIL`, `ERROR`, `TIMEOUT`, `SKIP` |
| `mode` | `poly` or `explicit` |
| `detail` | Residual description or error message |
| `time_s` | Wall time for this entry |

## Interpreting results

- **PASS** — flatness condition verified exactly. The CMF is mathematically confirmed flat.
- **FAIL** — non-zero residual found. Either a data entry error or the CMF is not actually flat (shouldn't happen for certified entries).
- **SKIP** — no polynomial form and no explicit matrices stored; cannot verify symbolically.
- **ERROR** — SageMath parse or evaluation error (e.g., malformed polynomial string).
- **TIMEOUT** — exceeded per-entry time limit (`--timeout N`, default 30s). Increase limit or check complexity.

## Timing estimates

| Category | Count | Typical time |
|----------|-------|-------------|
| Gauge Transformed (poly) | 81 | < 0.1s each |
| CMF Hunter (poly) | 263 | 0.1–2s each |
| RamanujanTools (explicit, 2D) | 10 | 1–5s each |
| RamanujanTools (explicit, 3D+) | 2 | 5–30s each |

Full run ≈ 5–15 minutes on a modern Mac.

## Updating certification levels

After a successful PASS, you can promote a CMF's certification level in the DB:

```python
# promote_certified.py — run after verifying
import sqlite3, json
con = sqlite3.connect('../data/atlas_2d.db')
# Set A_certified for symbolically verified CMF Hunter entries
con.execute("""
    UPDATE cmf SET cmf_payload = json_set(cmf_payload, '$.certification_level', 'A_certified')
    WHERE id = ?
""", (CMF_ID,))
con.commit()
```

Or use the bulk promotion helper (see `promote_certified.py`).
