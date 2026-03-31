# CMF Atlas v2.3 → v3.0 Roadmap

*Operational checklist for elevating CMF Atlas from strong beta to top-level mathematical research database.*

---

## Rollout Order

### Sprint 1 — Credibility blockers (current)
- [ ] Fix Home: remove "transcendental constants" language, fix depth wording, fix citation URL
- [ ] Remove `Loading…` from Home notable entries, Browse table, Constants, entry pages
- [ ] Fix Browse filters: remove non-existent 4D/7D, add degree 0/1, generate from live backend
- [ ] Fix Explorer label typo: K₀ along m → Ky along m
- [ ] Fix Conservative Test residual wording: K₁K₂−K₂K₁ → correct shifted flatness identity
- [ ] Unify certification enum (A+/A/B/C) and wording identically across all pages

### Sprint 2 — Research-grade data contract
- [ ] Expand per-entry schema (see §3)
- [ ] Rebuild Constants as a real registry page (constants vs. expressions split)
- [ ] Add release manifest endpoint; drive all page counts from it
- [ ] Add reverse links: entries ↔ constants ↔ sources
- [ ] Align Guide, About, Sources, Conservative Test terminology exactly

### Sprint 3 — Publication-grade infrastructure
- [ ] DOI-backed release snapshots (changelog, checksum, frozen export)
- [ ] Proof artifact registry (generalize symbolic_verification block)
- [ ] Equivalence classes and gauge relations
- [ ] Relation graph: entry ↔ constant ↔ source family ↔ proof artifact
- [ ] Tombstone policy for removed/merged entries
- [ ] API versioning under /api/v1/

---

## 1. Must Fix — Global Layer

### 1.1 Database & Metadata
- [ ] Create one canonical release manifest; generate all public counts from it
- [ ] Stop hard-coding dimension counts, source-family counts, certification summaries
- [ ] Canonical vocabulary (use exactly one name everywhere):
  - `dimension`: integer
  - `source_family`: `cmf_hunter | gauge_transformed | ramanujantools | euler2ai`
  - `category`: `telescope | accumulator | companion | ore_algebra | bank_cmf | recurrence_cmf | pcf | hypergeometric`
  - `certification_level`: `A_plus | A | B | C`  (never mix with A+ / A_certified / B_verified_numeric)
  - `representation_type`: `conjugate_poly | ore_algebra_operator | k1k2_explicit | pcf | hypergeom`
  - `identification_status`: `identified | unidentified | partial`
  - `proof_status`: `symbolic | numeric | unverified`
- [ ] One canonical citation URL everywhere
- [ ] Visible `release_version` and `schema_version` on all pages and API responses

### 1.2 Rendering & Crawlability
- [ ] Remove all `Loading…` placeholders from core pages
- [ ] Server-render or statically inject: release stats, entry metadata, constants table, default Browse page
- [ ] Every public entry page must contain real content in initial HTML (no JS-only shells)

### 1.3 Mathematical Language
- [ ] Remove/revise all uses of "transcendental constants" → "named constants" or "special values"
- [ ] Replace "transcendental values" on Sources with neutral arithmetic language
- [ ] Add explicit arithmetic-status per constant:
  - `irrationality_status`: `proved | open | false`
  - `transcendence_status`: `proved | open | false`
  - `identity_status`: `exact | symbolic_certified | numerical_match | unknown`
- [ ] Specific notes: ζ(3) = irrational (proved), transcendence unknown; ζ(5) unknown both; Catalan unknown both

### 1.4 Certification System
- [ ] One enum: `A_plus | A | B | C` — used identically on Browse, About, Sources, Guide, entry pages
- [ ] One fixed public wording set (e.g. A+ = Symbolically certified, A = Verified, B = Numeric, C = Scouting)
- [ ] `proof_status` field separate from `certification_level`
- [ ] Update schema docs so A+ is documented

### 1.5 Release & Citation
- [ ] Pick one canonical citation URL (decide: davidvesterlund.com/cmfatlas vs /cmf-atlas/)
- [ ] Add release note page for v2.3
- [ ] Add checksums for exports
- [ ] Add DOI-ready citation block structure
- [ ] Freeze one immutable export per release

---

## 2. Page-by-Page Checklist

### Home
**Must fix:**
- [ ] Remove "transcendental constants such as ζ(3), π, ln(2) and ζ(5)"
- [ ] Replace "Compare against 20+ known constants" with live registry count
- [ ] Fix "depth 500" → Explorer supports 500/1000/2000
- [ ] Remove `Loading…` from notable entries block
- [ ] Fix citation URL to canonical form

**Should fix:**
- [ ] Add: release number, schema version, total entry count, cert distribution, source-family distribution, dimension distribution
- [ ] Make those numbers link to release manifest

**Nice to have:**
- [ ] 3–5 featured entries with proof status, identification status, source family, citation shortcut

---

### Browse
**Must fix:**
- [ ] Generate all filter options from live backend metadata
- [ ] Remove 4D and 7D dimension filters (no records in current release)
- [ ] Add degree 0 and 1 filters
- [ ] Remove `Loading…` from table
- [ ] Make default page server-side render

**Should fix:**
- [ ] Replace "Formula = Has f(x,y) / No formula" with: explicit polynomial | projected polynomial | operator-backed | not public
- [ ] Add filters: `identification_status`, `proof_status`, `walk_available`, `representation_type`

**Nice to have:**
- [ ] Sortable columns: last updated, release, proof status, source family

---

### Entry Pages
**Must fix:**
- [ ] Render real entry metadata in HTML (no JS-only shell)
- [ ] Add machine-readable metadata in page source: title, formula, constant, certification, release version, citation
- [ ] Remove `Loading entry…` shell behavior

**Should fix:**
- [ ] Add fields: `entry_uri`, `release_version`, `schema_version`, `updated_at`, `proof_status`, `identification_status`, `recognition_method`, `recognition_digits`, `constant_id`, `source_family`, `construction_type`, `representation_type`, `equivalence_class_id`, `related_entries`, `related_publications`, `proof_artifact_url`

**Nice to have:**
- [ ] Downloadable BibTeX / CSL JSON citation files
- [ ] Downloadable proof artifact JSON

---

### Explorer
**Must fix:**
- [ ] Fix label typo: K₀ along m → Ky along m
- [ ] Add export of trajectory runs (JSON, CSV)
- [ ] Store full run metadata: entry ID, depth, walk direction, fixed coords, precision, normalization cadence, final estimate, local rate, self-delta

**Should fix:**
- [ ] Separate canonical constants from comparison expressions in comparison menu
- [ ] Record pairwise spread across queued trajectories
- [ ] Add downloadable run report

**Nice to have:**
- [ ] Depth-vs-spread plot for multi-trajectory comparisons
- [ ] Run permalinks

---

### Conservative Test
**Must fix:**
- [ ] Change page title/subtitle: "2D Flatness Residual Test" (3D = exploratory only)
- [ ] Correct residual description: currently says K₁K₂−K₂K₁; must reflect shifted identity K₁(k,m)K₂(k+1,m) = K₂(k,m)K₁(k,m+1)
- [ ] Remove template leakage in results block

**Should fix:**
- [ ] Split UI: 2D exact residual test | 3D slice consistency test (exploratory)
- [ ] Add explicit warning when 3D slices use projected rather than full operator data

**Nice to have:**
- [ ] Downloadable residual grids and summary certificates

---

### Sources
**Must fix:**
- [ ] Regenerate source-family counts from backend data
- [ ] Separate "what exists in theory" from "what is present in current release"
- [ ] Remove/qualify claims about dimensions not present in live release
- [ ] Replace "transcendental values" with neutral arithmetic language

**Should fix:**
- [ ] Add structured block per family: entry count, dimension dist., cert. dist., upstream source, import date, transformation status, novelty status

**Nice to have:**
- [ ] Links to upstream repositories and papers per family
- [ ] Per-family downloadable sub-exports

---

### Constants
**Must fix:**
- [ ] Remove `Loading…`; show real constant rows in initial HTML
- [ ] Distinguish constants from derived expressions (no raw -4/2 or 1/1 in canonical list)
- [ ] Replace boolean/null arithmetic metadata with explicit status enums
- [ ] Add reverse links from constants to entry pages

**Should fix:**
- [ ] Canonical constant schema: `constant_id`, `label`, `latex`, `aliases`, `value_display`, `value_precision_digits`, `formula`, `irrationality_status`, `transcendence_status`, `identity_notes`, `references`, `entry_count`, `entry_ids`
- [ ] Separate derived expression objects: `expression_id`, `expression_latex`, `relation_type`, `canonical_constant_links`

**Nice to have:**
- [ ] Relation graph: constant ↔ expression ↔ CMF entries ↔ source family ↔ proof artifacts

---

### Guide
**Must fix:**
- [ ] Ensure Guide uses exactly the same terminology as About, Browse, Conservative Test, entry pages
- [ ] Ensure category names and certification names match actual schema enums

**Should fix:**
- [ ] Add explicit note on matrix convention used by the Atlas
- [ ] Add explicit note on whether 3D formulas are full representations or projections

**Nice to have:**
- [ ] Short glossary: flatness, cocycle condition, trajectory, certification, identification status

---

### About
**Must fix:**
- [ ] Rewrite opening: Atlas is not only "2D lattice recurrences"
- [ ] Replace hard-coded counts with release-derived counts
- [ ] Update schema documentation to include actual public classification system
- [ ] Document API version strategy
- [ ] Add release-governance section: release process, deprecation policy, identifier policy, citation policy

**Should fix:**
- [ ] Add sections: schema version, enum dictionary, identifier semantics, proof artifact semantics, identification semantics, export formats, backward compatibility policy

**Nice to have:**
- [ ] Tombstone policy for deprecated/merged entries
- [ ] Editorial policy for certification level promotion/demotion

---

## 3. Schema Checklist

### Must add now
- [ ] `release_version`
- [ ] `schema_version`
- [ ] `entry_uri`
- [ ] `updated_at`
- [ ] `proof_status`
- [ ] `identification_status`
- [ ] `constant_id`
- [ ] `source_family`
- [ ] `construction_type`
- [ ] `representation_type`

### Should add next
- [ ] `recognition_method`
- [ ] `recognition_digits`
- [ ] `proof_artifact_url`
- [ ] `equivalence_class_id`
- [ ] `related_entries`
- [ ] `related_publications`
- [ ] `not_walkable_reason`

### Nice to have later
- [ ] `gauge_relation`
- [ ] `symbolic_certificate_hash`
- [ ] `numeric_experiment_ids`
- [ ] `operator_data_uri`

---

## 4. Five First-Class Entities (Target Schema)

**A. cmf_entry** — one mathematical object record
**B. constant** — canonical constant (ζ(3), π, ln(2), …)
**C. expression_identity** — derived form (ζ(2)/π², π/4, scalar multiples)
**D. proof_artifact** — machine-readable symbolic/formal verification
**E. release** — immutable snapshot with DOI, changelog, checksum, export manifest, schema version

---

## 5. Definition of Done for v2.3

- [ ] No core public page shows `Loading…`
- [ ] All page counts match one release manifest
- [ ] No page overstates arithmetic status of constants
- [ ] Browse filters reflect real current data
- [ ] Entry pages are citable and machine-readable
- [ ] Certification language is identical across the site
- [ ] Constants layer distinguishes canonical constants from derived expressions
- [ ] About documents the real schema and release policy

---

## 6. Definition of Done for v3.0

1. Every public claim matches the live release data
2. Every public entry page is directly citable and machine-readable without JS
3. Every identified limit has explicit identification status and arithmetic-status semantics
4. Every certified entry links to an auditable proof artifact
5. Every release has DOI, checksum, changelog, and frozen export
6. Database distinguishes exact mathematics from numerical evidence everywhere

---

*Last updated: 2026-03-31*
