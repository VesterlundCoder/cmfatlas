# CMF Atlas — Complete Algorithm & Generation Pipeline
## Reference Document for Research Paper

**Last updated:** April 2026  
**Author:** David Vesterlund

---

## Overview

The CMF Atlas generation pipeline consists of five sequential stages, each implemented as a standalone Python module. The diagram below shows the full data flow:

```
STAGE 0: Candidate Sampling
    run_all_agents.py  →  sample_params()  →  build_eval_fns()

STAGE 1: Fast Filtering (pre-convergence)
    asymptotic_filter.py  →  T1 pole check  →  T2 fast numpy convergence

STAGE 2: High-Precision Convergence Verification
    T3 thorough walk (mpmath, dps=50)  →  T4 flatness / path-independence

STAGE 3: Irrationality Classification (4-Gate Filter)
    irrational_scout.py  →  get_limit_value()  →  classify_limit_strict()

STAGE 4: Database Ingestion & Distillation
    ingest_gauge_agents.py  →  backfill_irrational.py  →  cmf_distiller.py
```

---

## Stage 0 — Candidate Sampling

**File:** `gauge_agents/run_all_agents.py`

The pipeline uses 10 parallel *gauge-bootstrap agents* (A–J), each with a different parameterisation region:

| Agent | Dims    | n_vars | Strategy |
|-------|---------|--------|----------|
| A     | 3–5     | 3      | Classic LDU, extended off-diagonal values ∈ {±0.5, ±1, ±2} |
| B     | 3–5     | 3      | Sparse LDU, small fractions ∈ {±0.25, ±0.5} |
| C     | 6–8     | 3      | Large-dim climbing |
| D     | 4–6     | 4      | 4-variable holonomic |
| E     | 3–5     | 3      | Extended rationals (thirds / halves) |
| F     | 3–5     | 3      | Symmetric G: L_off = U_off^T |
| G     | 3–6     | 3      | Minimal structure (1 off-diagonal) |
| H     | 4–6     | 3      | Dense off-diagonals |
| I     | 3–4     | 3      | Integer-only parameters, wide range |
| J     | 3–6     | 5      | 5-variable holonomic |

### The LDU Gauge Construction

Each candidate CMF is parameterised by a **gauge matrix**:

```
G(n) = L(n) · D_diag(n) · U(n)
```

where:
- `L` is lower unitriangular with off-diagonal entries drawn from agent-specific pools
- `D_diag(n)` = diag(a₀·n + b₀, a₁·n + b₁, …) — linear diagonal
- `U` is upper unitriangular (same structure as L)

The **k-th generator matrix** is defined as:

```
X_k(n) = G(n + e_k) · Δ_k(n_k) · G(n)⁻¹
```

where `Δ_k(n_k) = diag(n_k, n_k+1, …, n_k+dim-1)` is the canonical shift.

**Path independence is guaranteed by construction**: since diagonal matrices commute, `X_j(n+e_i) · X_i(n) = X_i(n+e_j) · X_j(n)` holds for any choice of L, D, U.

**Key function:** `sample_params(dim, agent_cfg, rng)` → `build_eval_fns(params)`

---

## Stage 1 — Asymptotic Fast-Fail Filter

**File:** `gauge_agents/asymptotic_filter.py`

Before any expensive mpmath walk, every candidate passes a 3-step millisecond filter:

### Step 1: Degree Balance (Determinant Asymptotics)
Sample `det(M_0(n))` at n = 200, 400, 800. Fit `log|det| = k·log(n) + c`.
- Reject if `k < -1` (matrix collapses to zero)
- Reject if `k > 3·dim + 2` (explosive growth)

### Step 2: Spectral Condition of Limit Matrix
Evaluate `M_0` at n=500, normalise by Frobenius norm, compute SVD.
- Reject if condition number `> 10^(dim+3)` — one direction dominates → walk diverges

### Step 3: Early-Stopping Norm Ratio
Walk 20 steps along axis 0 with running renormalisation. Compute `R = norm(step 20) / norm(step 10)`.
- Reject if `R > 1e6` (diverging) or `R < 1e-6` (collapsing)

**Typical rejection rate:** ~95% of candidates are eliminated here in < 1ms.

---

## Stage 2 — High-Precision Convergence Verification

**File:** `gauge_agents/run_all_agents.py` — functions `t1_pole_check`, `t2_fast`, `t3_thorough`, `t4_flatness`

### T1: Pole Check
Random-axis walk at n=2. Checks that no matrix entry becomes NaN/Inf.

### T2: Fast Numpy Convergence
Walk 200 steps with float64. Track ratio `v[0]/v[-1]` at steps 50/100/200. Require convergence (delta > threshold).

### T3: Thorough High-Precision Walk
Walk 500 steps with `mpmath, dps=50`. Verify convergence delta > `DELTA_FULL_MIN = 2.5`.

### T4: Path-Independence (Flatness) Check
Walk along 2 different orderings of the same endpoint. Compare limits; require relative error < 10⁻⁸.

**Output:** `deltas` (per-axis convergence rates), `pi_err` (flatness error), `bidir_ratio` (coupling metric).

---

## Stage 3 — Irrationality Classification (4-Gate Filter)

**File:** `gauge_agents/reward_engine.py` — `classify_limit_strict(val, hp_val=None)`  
**File:** `gauge_agents/irrational_scout.py` — `get_limit_value()`, `identify_limit()`

### High-Precision Limit Walk

For each converging CMF, walk the matrix product to depth 1200 at `mp.dps = 60`:

```python
v[0] / v[dim-1]   # readout ratio
```

This gives the limit value `L` to ~60 decimal places.

### The 4-Gate Strict Irrationality Filter

Applied to every limit value `L`. Gates are evaluated in order; any FATAL terminates evaluation.

#### Gate 1 — Triviality & Zero Trapdoor
Rejects obviously degenerate limits:
- `|L| < 10⁻⁴` → **FATAL_ZERO_TRAP** (limit collapsed to zero)
- `|L| > 10⁴` → **FATAL_DIVERGENCE_TRAP** (limit diverged)
- `|L ± 1| < 10⁻⁶` → **FATAL_NEAR_ONE / FATAL_NEAR_MINUS_ONE** (trivial fixed points)
- Farey/Stern-Brocot rationality check with max_denominator=10,000 → **FATAL_TRIVIAL_RATIONAL**

#### Gate 2 — Algebraic Purge
Runs `mpmath.identify(L)` at `dps=50`. If the returned formula string contains **no** transcendental keyword (`pi`, `zeta`, `log`, `catalan`, `euler`, `gamma`, `Li`, `exp`) — meaning it is a pure algebraic expression — the candidate is rejected:
- → **FATAL_ALGEBRAIC_ESCAPE** (e.g. `3**(208/81)` — an algebraic irrational)

#### Gate 3 — Strict PSLQ Coefficient Cap
Runs PSLQ at `dps=100` against the strict 7-element transcendental basis:
```
[L, 1, π, π², log(2), ζ(2), ζ(3), Catalan]
```
If a relation `Σ cᵢ·bᵢ = 0` is found with `max|cᵢ| > 50`, the candidate is rejected:
- → **FATAL_PSLQ_OVERFIT** (agent exploited large-coefficient fitting to manufacture a false "transcendental")

**Note:** coefficients[0] (the coefficient of L) must be nonzero, otherwise the relation is a trivial basis identity, not an identification.

#### Gate 4 — True Transcendental Reward
If Gates 1–3 all pass and a PSLQ relation **is** found (with `max|cᵢ| ≤ 50`) with residual `< 10⁻⁵⁰`:
- → **TRUE_TRANSCENDENTAL** — score multiplier ×5.0

If Gates 1–3 pass but no PSLQ relation is found:
- → **IRRATIONAL_UNKNOWN** — score multiplier ×3.0 (genuine candidate, unidentified)

### Scoring

```python
score = base_score × gate_result["score_mult"]
```

where:
```python
base_score = 0.25 * conv_rate + 0.15 * ray_stability
           + 0.15 * identifiability + 0.10 * simplicity
           + 0.05 * proofability
```

Score multipliers:
| Label | Multiplier |
|-------|------------|
| TRUE_TRANSCENDENTAL | ×5.0 |
| IRRATIONAL_UNKNOWN | ×3.0 |
| FATAL_* | ×0.02 |

---

## Stage 4 — Ingestion, Backfill & Distillation

### Ingestion

**File:** `ingest_gauge_agents.py`

Reads all `gauge_agents/store_*.jsonl` files and inserts NEW records into `data/atlas_2d.db` (SQLite). For each record:
- Builds `cmf_payload` with all metadata including `looks_irrational`, `limit_label`, `limit_value`
- Builds `canonical_payload` with symbolic matrices `X0`, `X1`, `X2`, ...
- Assigns certification level: A_plus → A_certified → B_verified_numeric → C_scouting
- Deduplicates by 16-character fingerprint (`canonical_fingerprint`)

### Backfill

**File:** `backfill_irrational.py`

Patches **existing** DB records (already ingested in prior runs) with irrational metadata from updated store files. Matches by fingerprint[:16]. Also exports `data/irrational_candidates.json`.

### CMF Distiller — Three-Stage Rigorous Pipeline

**File:** `cmf_distiller.py`

Takes `data/irrational_candidates.json` and applies extreme mathematical rigor:

#### Distiller Stage 1 — Extreme-Precision Verification
Re-walks each candidate at `mp.dps = 1000` for 10,000 steps. Applies the same 4-gate classifier with the 1000-digit value as `hp_val`, enabling Gate 4 (PSLQ residual < 10⁻⁵⁰) at full precision.

#### Distiller Stage 2 — Gauge-Equivalence Clustering
Computes a gauge-invariant structural fingerprint from eigenvalues and trace products. Groups CMFs into *families* of gauge-equivalent structures.

#### Distiller Stage 3 — Auto-Theorem Export
For each family: generates structured JSON and LaTeX theorem snippets suitable for direct inclusion in a research paper.

---

## Key Data Files

| File | Description |
|------|-------------|
| `gauge_agents/store_[A-J]_[dim]x[dim].jsonl` | Raw discovery log — one JSON record per converging CMF |
| `data/atlas_2d.db` | SQLite database — full CMF atlas |
| `data/irrational_candidates.json` | All CMFs with `looks_irrational=True` (all runs) |
| `data/irrational_candidates_v2.json` | CMFs from v2 anti-hack scout run |
| `data/distiller_stage1.jsonl` | Stage 1 output — per-CMF gate results at 1000 dps |
| `data/distiller_stage2.json` | Stage 2 output — gauge-equivalence clusters |
| `data/distiller_paper_data.json` | Stage 3 output — publication-ready structured JSON |
| `data/distiller_theorems.tex` | Stage 3 output — LaTeX theorem snippets |
| `data/research_paper_draft.md` | Research paper introduction draft |

---

## Key Algorithms Summary (for Methods section)

### 1. LDU Gauge Bootstrap
*"Path independence by construction"* — any L, D, U choice gives a valid CMF family. The gauge parameterisation ensures the flatness condition `X_j(n+eᵢ)·X_i(n) = X_i(n+eⱼ)·X_j(n)` holds identically.

### 2. Asymptotic Pre-Filter
3-step millisecond filter (degree balance, spectral condition, norm ratio) eliminates ~95% of candidates before the expensive mpmath walk.

### 3. 4-Gate Irrationality Classifier
Prevents reward hacking by enforcing: no near-zero/near-integer limits (G1), no pure algebraic expressions (G2), no large-coefficient PSLQ fits (G3). Rewards only limits matching known transcendentals with small integer coefficients (G4).

### 4. Gauge-Invariant Clustering
Structural fingerprint computed from eigenvalues and trace products of the product matrix `X₀(n) · X₁(n)`. Invariant under gauge transformations `Xᵢ → S · Xᵢ · S⁻¹`.

### 5. PSLQ Identification
Extended integer-relation algorithm at up to 1000 decimal digits. Basis: `[1, π, π², log(2), ζ(2), ζ(3), Catalan]`. Relations found with max coefficient ≤ 50 and residual < 10⁻⁵⁰ are reported as proven identities.

---

## Running the Full Pipeline

```bash
# Step 1: Run scouts (all 10 agents, 20 irrational CMFs each)
nohup venv/bin/python3 -u gauge_agents/irrational_scout.py \
    --agents A B C D E F G H I J --target 20 \
    > /tmp/scout_v2.log 2>&1 &

# Step 2: Export v2 candidates to separate JSON
python3 export_v2_candidates.py

# Step 3: Ingest new CMFs into DB + backfill irrational metadata
python3 ingest_gauge_agents.py
python3 backfill_irrational.py

# Step 4: Run CMF Distiller (3 stages)
python3 cmf_distiller.py

# Step 5: Push to Railway (live server)
railway up --detach
```
