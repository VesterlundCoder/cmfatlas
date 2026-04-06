"""
deep_dreams_pslq.py

For every passing CMF, runs N_TRAJ deterministic trajectories spread evenly
over starting coordinates and matrix axes, then applies PSLQ to identify the
convergent limit against a library of fundamental constants.

Pipeline per CMF
────────────────
1. Generate 10 000 starting points via a low-discrepancy grid over [2, MAX_START]^n_vars.
2. Screen each trajectory with a fast float64 walk; keep convergent ones.
3. Cluster the convergent float values; sample ≤ N_DEEP representative values.
4. Re-compute each representative at DPS_DEEP decimal places using mpmath.
5. Run PSLQ against the fundamental constants basis; store every hit.

Results
───────
  data/pslq_hits.db          — SQLite with tables pslq_hits + pslq_cmf_summary
  /tmp/pslq_results.jsonl    — one JSON line per CMF (overridable with --out)

Usage
─────
    python deep_dreams_pslq.py \\
        --audit /tmp/dreams_full_audit_v3.jsonl \\
        --jobs 0 \\
        --out /tmp/pslq_results.jsonl \\
        [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import mpmath
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from verify_convergence_dreams import make_step_fns  # noqa: E402

DB_PATH  = HERE.parent / "data" / "atlas_2d.db"
PSLQ_DB  = HERE.parent / "data" / "pslq_hits.db"

# ── Tuning constants ────────────────────────────────────────────────────────
N_TRAJ        = 10_000   # trajectories per CMF
MAX_START     = 60       # max starting coordinate (min is always 2)
SCREEN_DEPTH  = 300      # float64 screen walk steps
DPS_DEEP      = 35       # mpmath precision for confirmed trajectories (35 is enough for PSLQ)
DEEP_DEPTH    = 400      # mpmath confirmation walk depth
N_DEEP        = 50       # max unique cluster representatives per CMF
N_CONFIRM     = 5        # max mpmath confirmation walks per CMF (only for candidate hits)
PSLQ_TOL_EXP  = 12       # require PSLQ residual < 10^-PSLQ_TOL_EXP
PSLQ_MAXCOEFF = 300      # max integer coefficient in PSLQ search


# ── Fundamental constants basis ─────────────────────────────────────────────

def _build_basis() -> tuple[list[str], list]:
    mpmath.mp.dps = 70
    names: list[str] = []
    vals:  list      = []

    def add(name: str, val) -> None:
        names.append(name)
        vals.append(val)

    add("1",           mpmath.mpf(1))
    add("pi",          mpmath.pi)
    add("pi2",         mpmath.pi ** 2)
    add("pi3",         mpmath.pi ** 3)
    add("pi4",         mpmath.pi ** 4)
    add("e",           mpmath.e)
    add("e2",          mpmath.e ** 2)
    add("log2",        mpmath.log(2))
    add("log3",        mpmath.log(3))
    add("log5",        mpmath.log(5))
    add("log7",        mpmath.log(7))
    add("sqrt2",       mpmath.sqrt(2))
    add("sqrt3",       mpmath.sqrt(3))
    add("sqrt5",       mpmath.sqrt(5))
    add("sqrt6",       mpmath.sqrt(6))
    add("sqrt7",       mpmath.sqrt(7))
    add("sqrt10",      mpmath.sqrt(10))
    add("phi",         (1 + mpmath.sqrt(5)) / 2)
    add("zeta2",       mpmath.zeta(2))
    add("zeta3",       mpmath.zeta(3))
    add("zeta4",       mpmath.zeta(4))
    add("zeta5",       mpmath.zeta(5))
    add("zeta6",       mpmath.zeta(6))
    add("catalan",     mpmath.catalan)
    add("euler_gamma", mpmath.euler)
    add("pi_sqrt2",    mpmath.pi * mpmath.sqrt(2))
    add("pi_sqrt3",    mpmath.pi * mpmath.sqrt(3))
    add("pi_e",        mpmath.pi * mpmath.e)
    add("log2_pi",     mpmath.log(2) * mpmath.pi)
    add("log2_sqrt2",  mpmath.log(2) * mpmath.sqrt(2))
    add("log3_pi",     mpmath.log(3) * mpmath.pi)
    add("pi2_6",       mpmath.pi ** 2 / 6)   # = zeta(2), but listed separately
    add("log2_2",      mpmath.log(2) ** 2)
    add("pi_over_4",   mpmath.pi / 4)
    add("pi_over_3",   mpmath.pi / 3)
    add("pi_over_6",   mpmath.pi / 6)

    return names, vals


_BASIS_NAMES, _BASIS_VALS = _build_basis()

# Subsets for PSLQ probing (smaller = faster; try them in order)
_PSLQ_SUBSETS: list[list[str]] = [
    ["1", "pi", "log2", "sqrt2", "zeta3"],
    ["1", "pi", "pi2", "log2", "e"],
    ["1", "zeta2", "zeta3", "zeta4", "zeta5"],
    ["1", "pi", "log2", "log3", "sqrt2", "sqrt3"],
    ["1", "pi", "catalan", "euler_gamma", "log2"],
    ["1", "pi2", "pi3", "log2", "zeta3"],
    ["1", "phi", "sqrt5", "log2", "pi"],
    ["1", "pi", "e", "sqrt2", "log2", "zeta3", "catalan"],
    _BASIS_NAMES[:10],
    _BASIS_NAMES[:15],
    _BASIS_NAMES[:20],
    _BASIS_NAMES,
]


# ── Float64 screening walk ────────────────────────────────────────────────

def _screen_walk(fn, dim: int, n_vars: int,
                 start_coords: list, depth: int = SCREEN_DEPTH) -> Optional[float]:
    """
    Fast float64 walk along pos[0] starting from start_coords.
    Returns the final ratio v[i]/v[j] if convergent (delta >= 2), else None.
    """
    pos    = list(start_coords)
    warmup = max(10, depth // 4)
    v      = np.ones(dim, dtype=float) / math.sqrt(dim)

    for _ in range(warmup):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
            if not np.all(np.isfinite(M)):
                continue
            v_new = M @ v
        except Exception:
            continue  # skip singular lattice point
        n = np.max(np.abs(v_new))
        if n > 1e25:
            v = v_new / n
        elif n < 1e-25:
            return None
        else:
            v = v_new

    norms = np.abs(v)
    if np.max(norms) < 1e-300:
        return None
    sc = np.argsort(norms)[::-1]
    j, i = int(sc[0]), int(sc[1])  # top-2 by absolute norm, no relative threshold

    ratio_prev: Optional[float] = None
    delta = 0.0

    for _ in range(depth - warmup):
        pos[0] += 1
        try:
            M = np.asarray(fn(*pos), dtype=float)
            if not np.all(np.isfinite(M)):
                continue
            v_new = M @ v
        except Exception:
            continue  # skip singular lattice point
        n = np.max(np.abs(v_new))
        if n > 1e25:
            v = v_new / n
        elif n < 1e-25:
            return None
        else:
            v = v_new
        if abs(v[j]) < 1e-18:
            return None
        ratio = float(v[i] / v[j])
        if ratio_prev is not None:
            diff = abs(ratio - ratio_prev)
            if 0 < diff < 1e50:
                delta = max(delta, -math.log10(diff + 1e-300))
        ratio_prev = ratio

    if delta < 2.0 or ratio_prev is None:
        return None
    return ratio_prev


# ── High-precision mpmath walk ─────────────────────────────────────────────

def _deep_walk(fn, dim: int, n_vars: int,
               start_coords: list, depth: int = DEEP_DEPTH,
               dps: int = DPS_DEEP) -> Optional[mpmath.mpf]:
    """
    High-precision mpmath walk along pos[0].
    Returns top-2-component ratio as mpf, or None if it collapses.
    """
    mpmath.mp.dps = dps + 10
    pos    = list(start_coords)
    warmup = max(10, depth // 4)
    v      = mpmath.ones(dim, 1)
    v      = v / mpmath.sqrt(dim)

    j_idx: Optional[int] = None
    i_idx: Optional[int] = None

    for step in range(depth):
        pos[0] += 1
        try:
            raw = np.asarray(fn(*pos), dtype=float)
            M   = mpmath.matrix([[mpmath.mpf(str(raw[r][c]))
                                  for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            return None
        scale = max(abs(v[k]) for k in range(dim))
        if scale > mpmath.power(10, 25):
            v /= scale
        elif scale < mpmath.power(10, -25):
            return None

        if step + 1 == warmup:
            norms      = [abs(v[k]) for k in range(dim)]
            sorted_idx = sorted(range(dim), key=lambda x: -norms[x])
            j_idx, i_idx = sorted_idx[0], sorted_idx[1]
            if norms[j_idx] < mpmath.power(10, -200):
                return None  # genuine collapse

    if j_idx is None or i_idx is None:
        return None
    if abs(v[j_idx]) < mpmath.power(10, -(dps - 5)):
        return None
    return v[i_idx] / v[j_idx]


# ── PSLQ identification ────────────────────────────────────────────────────

def _pslq_identify(x: mpmath.mpf, dps: int = DPS_DEEP) -> list[dict]:
    """
    Try to express x as a rational-linear combination of fundamental constants.
    Returns a list of hit dicts (possibly empty).
    """
    mpmath.mp.dps = dps + 5
    hits: list[dict] = []
    tol = mpmath.power(10, -PSLQ_TOL_EXP)

    # ── Method 1: mpmath.identify ─────────────────────────────────────────
    try:
        s = mpmath.identify(x, tol=tol)
        if s and len(s) < 120:
            hits.append({
                "method":    "mpmath_identify",
                "formula":   s,
                "residual":  0.0,
                "basis":     [],
                "coefficients": [],
            })
    except Exception:
        pass

    # ── Method 2: custom PSLQ against basis subsets ───────────────────────
    basis_map = dict(zip(_BASIS_NAMES, _BASIS_VALS))

    for subset in _PSLQ_SUBSETS:
        if len(hits) >= 4:
            break
        try:
            sub_vals = [basis_map[n] for n in subset]
            vec      = [x] + sub_vals
            result   = mpmath.pslq(vec, tol=tol, maxcoeff=PSLQ_MAXCOEFF)
            if result is None:
                continue
            if abs(result[0]) < 1e-9:
                continue  # degenerate: coefficient of x is zero

            residual = abs(sum(result[k] * vec[k] for k in range(len(result))))
            if residual > tol * 100:
                continue

            # Build human-readable formula: x = -(r[1]*b0 + r[2]*b1 + ...) / r[0]
            terms = []
            for k in range(1, len(result)):
                c = int(result[k])
                if c == 0:
                    continue
                sign = "+" if c > 0 else ""
                terms.append(f"{sign}{c}*{subset[k-1]}")
            if not terms:
                formula = "0"
            else:
                formula = f"(-({''.join(terms)}))/{int(result[0])}"

            hits.append({
                "method":       "pslq",
                "formula":      formula,
                "residual":     float(residual),
                "basis":        subset,
                "coefficients": [int(r) for r in result],
            })
        except Exception:
            continue

    return hits


# ── Per-CMF worker ─────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict:
    rec, traj_starts = args
    fp    = rec.get("fingerprint", "?")
    p     = rec.get("params", {})
    dim   = int(p.get("dim", rec.get("dim", 3)))
    agent = rec.get("agent", "?")

    try:
        fns = make_step_fns(rec)
        if not fns:
            return {"fingerprint": fp, "agent": agent, "dim": dim,
                    "status": "no_step_fns", "hits": [],
                    "n_trajectories": 0, "n_convergent": 0}

        n_vars = len(fns)

        # ── Phase 1: screen all trajectories ────────────────────────────
        # Map ratio_bucket -> (start_coords, ax, float_r)
        bucket_map: dict[str, tuple] = {}
        for start_coords, ax in traj_starts:
            if ax >= n_vars:
                ax = ax % n_vars
            fn = fns[ax]
            r  = _screen_walk(fn, dim, n_vars, list(start_coords), SCREEN_DEPTH)
            if r is None:
                continue
            bucket = f"{ax}_{r:.5f}"
            bucket_map[bucket] = (start_coords, ax, r)

        n_convergent = len(bucket_map)
        if n_convergent == 0:
            return {"fingerprint": fp, "agent": agent, "dim": dim,
                    "status": "no_convergent", "hits": [],
                    "n_trajectories": len(traj_starts),
                    "n_convergent": 0}

        # ── Phase 2: cluster → sample ≤ N_DEEP representatives ──────────
        items = sorted(bucket_map.values(), key=lambda x: (x[1], float(x[2])))
        step  = max(1, len(items) // N_DEEP)
        sampled = items[::step][:N_DEEP]

        # ── Phase 3: PSLQ on float64 ratios directly (fast path) ────────
        # We use mpmath.mpf(float_r) directly — no expensive deep mpmath walk.
        # For confirmed candidate hits, we do ONE mpmath confirmation walk.
        all_hits: list[dict] = []
        confirmed_deep = 0
        seen_formulas: set[str] = set()

        for start_coords, ax, float_r in sampled:
            # Use float64 value via mpmath for identify/PSLQ
            x_approx = mpmath.mpf(float_r)
            hits = _pslq_identify(x_approx, dps=DPS_DEEP)
            if not hits:
                continue

            # Confirmation: one mpmath walk per genuine candidate formula
            fn = fns[ax % n_vars]
            r_confirmed: Optional[mpmath.mpf] = None
            for h in hits:
                formula = h.get("formula", "")
                if formula in seen_formulas:
                    continue
                seen_formulas.add(formula)
                # Do the expensive deep walk only once per unique formula
                if r_confirmed is None and confirmed_deep < N_CONFIRM:
                    r_confirmed = _deep_walk(fn, dim, n_vars, list(start_coords),
                                             DEEP_DEPTH, DPS_DEEP)
                    confirmed_deep += 1
                if r_confirmed is not None:
                    confirm_hits = _pslq_identify(r_confirmed, dps=DPS_DEEP)
                    for ch in confirm_hits:
                        ch["start_coords"] = list(start_coords)
                        ch["axis"]         = ax
                        ch["ratio_value"]  = str(r_confirmed)
                    all_hits.extend(confirm_hits)
                else:
                    # Accept float64-based hit with lower confidence
                    h["start_coords"] = list(start_coords)
                    h["axis"]         = ax
                    h["ratio_value"]  = str(float_r)
                    h["method"]      += "_float64"
                    all_hits.append(h)

        best_formula  = None
        best_residual = 1.0
        if all_hits:
            best          = min(all_hits, key=lambda h: h.get("residual", 1.0))
            best_formula  = best.get("formula", "")
            best_residual = best.get("residual", 1.0)

        return {
            "fingerprint":    fp,
            "agent":          agent,
            "dim":            dim,
            "n_vars":         n_vars,
            "n_trajectories": len(traj_starts),
            "n_convergent":   n_convergent,
            "n_sampled":      len(sampled),
            "n_hits":         len(all_hits),
            "hits":           all_hits,
            "best_formula":   best_formula,
            "best_residual":  best_residual,
            "status":         "ok" if all_hits else "no_hits",
            "_source_rec":    rec,  # kept for golden-hits writer; stripped before DB save
        }

    except Exception as exc:
        return {"fingerprint": fp, "agent": agent, "dim": dim,
                "status": f"error:{exc}", "hits": [],
                "n_trajectories": 0, "n_convergent": 0}


# ── DB helpers ─────────────────────────────────────────────────────────────

def _init_pslq_db() -> None:
    con = sqlite3.connect(PSLQ_DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS pslq_hits (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint    TEXT    NOT NULL,
            agent          TEXT,
            dim            INTEGER,
            n_vars         INTEGER,
            start_coords   TEXT,
            axis           INTEGER,
            ratio_value    TEXT,
            formula        TEXT,
            method         TEXT,
            residual       REAL,
            coefficients   TEXT,
            basis          TEXT,
            dps            INTEGER,
            created_at     TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS pslq_cmf_summary (
            fingerprint    TEXT PRIMARY KEY,
            agent          TEXT,
            dim            INTEGER,
            n_vars         INTEGER,
            n_trajectories INTEGER,
            n_convergent   INTEGER,
            n_sampled      INTEGER,
            n_hits         INTEGER,
            best_formula   TEXT,
            best_residual  REAL,
            status         TEXT,
            processed_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ph_fp  ON pslq_hits(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_ph_res ON pslq_hits(residual);
        CREATE INDEX IF NOT EXISTS idx_ph_method ON pslq_hits(method);
    """)
    con.commit()
    con.close()


def _save_results(results: list[dict]) -> None:
    con = sqlite3.connect(PSLQ_DB)
    for res in results:
        fp    = res["fingerprint"]
        agent = res.get("agent", "?")
        dim   = res.get("dim", 0)
        n_vars = res.get("n_vars", 0)

        for h in res.get("hits", []):
            con.execute("""
                INSERT INTO pslq_hits
                  (fingerprint, agent, dim, n_vars, start_coords, axis,
                   ratio_value, formula, method, residual,
                   coefficients, basis, dps)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (fp, agent, dim, n_vars,
                  json.dumps(h.get("start_coords", [])),
                  h.get("axis", 0),
                  h.get("ratio_value", ""),
                  h.get("formula", ""),
                  h.get("method", ""),
                  h.get("residual", 0.0),
                  json.dumps(h.get("coefficients", [])),
                  json.dumps(h.get("basis", [])),
                  DPS_DEEP))

        hits       = res.get("hits", [])
        best       = min(hits, key=lambda h: h.get("residual", 1.0)) if hits else None
        con.execute("""
            INSERT OR REPLACE INTO pslq_cmf_summary
              (fingerprint, agent, dim, n_vars,
               n_trajectories, n_convergent, n_sampled, n_hits,
               best_formula, best_residual, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (fp, agent, dim, n_vars,
              res.get("n_trajectories", 0),
              res.get("n_convergent", 0),
              res.get("n_sampled", 0),
              len(hits),
              best["formula"] if best else None,
              best["residual"] if best else None,
              res.get("status", "ok")))

    con.commit()
    con.close()


# ── Trajectory generator ───────────────────────────────────────────────────

def _gen_trajectories(n_vars: int, n_traj: int, fp: str) -> list[tuple]:
    """
    Generate n_traj (start_coords, axis) pairs.
    Uses a low-discrepancy grid: axis cycles 0..n_vars-1, coordinates
    drawn from a regular grid over [2, MAX_START] stratified by axis.
    """
    rng    = random.Random(hash(fp) & 0xFFFFFFFF)
    starts = []
    # Build a flat grid of starting coordinates
    grid_side = max(2, int(round((n_traj ** (1 / n_vars)))))
    coords_pool = list(range(2, MAX_START + 1))

    for t in range(n_traj):
        ax    = t % n_vars
        start = [rng.choice(coords_pool) for _ in range(n_vars)]
        starts.append((start, ax))
    return starts


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Dreams PSLQ analysis")
    parser.add_argument("--audit",  default="/tmp/dreams_full_audit_v3.jsonl",
                        help="Path to Dreams audit JSONL (pass/fail per CMF)")
    parser.add_argument("--jobs",   type=int, default=0,
                        help="Worker processes (0 = cpu_count-1)")
    parser.add_argument("--n-traj", type=int, default=N_TRAJ,
                        help="Trajectories per CMF")
    parser.add_argument("--out",    default="/tmp/pslq_results.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--resume", action="store_true",
                        help="Skip CMFs already in pslq_cmf_summary")
    parser.add_argument("--golden-out", default=str(HERE.parent / "data" / "pslq_golden_hits.jsonl"),
                        help="Separate JSONL for confirmed PSLQ hits (ML training data)")
    args = parser.parse_args()

    n_jobs = args.jobs if args.jobs > 0 else max(1, mp.cpu_count() - 1)

    _init_pslq_db()

    # ── Load passing fingerprints ─────────────────────────────────────────
    passing_fps: set[str] = set()
    audit_path = Path(args.audit)
    if audit_path.exists():
        with open(audit_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("pass"):
                        passing_fps.add(rec["fp"])
                except Exception:
                    pass
        print(f"Loaded {len(passing_fps):,} passing CMFs from audit")
    else:
        print(f"Audit not found at {audit_path} — will process ALL store records")

    # ── Load already-processed if resuming ───────────────────────────────
    processed_fps: set[str] = set()
    if args.resume:
        con = sqlite3.connect(PSLQ_DB)
        for row in con.execute("SELECT fingerprint FROM pslq_cmf_summary"):
            processed_fps.add(row[0])
        con.close()
        print(f"Resume: {len(processed_fps):,} CMFs already processed — skipping")

    # ── Load records from store files ────────────────────────────────────
    store_files = sorted(HERE.glob("store_*.jsonl"))
    records: list[dict] = []
    for sf in store_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                fp = rec.get("fingerprint", "")
                if passing_fps and fp not in passing_fps:
                    continue
                if fp in processed_fps:
                    continue
                records.append(rec)

    print(f"Processing {len(records):,} CMFs  |  {args.n_traj:,} trajectories each  |  {n_jobs} workers")
    print(f"Screen depth={SCREEN_DEPTH}  Deep depth={DEEP_DEPTH}  dps={DPS_DEEP}")
    print(f"PSLQ basis size={len(_BASIS_NAMES)}  tol=1e-{PSLQ_TOL_EXP}")
    print(f"Output: {args.out}  |  DB: {PSLQ_DB}\n")

    # ── Build work items ──────────────────────────────────────────────────
    work_items = []
    for rec in records:
        p      = rec.get("params", {})
        n_vars = int(p.get("n_vars") or p.get("nvars") or
                     rec.get("n_matrices") or len(p.get("D_params", [])) or 3)
        traj_starts = _gen_trajectories(n_vars, args.n_traj, rec.get("fingerprint", ""))
        work_items.append((rec, traj_starts))

    # ── Process ───────────────────────────────────────────────────────────
    t0             = time.time()
    n_done         = 0
    n_total        = len(work_items)
    n_hits_total   = 0
    out_path       = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path    = Path(args.golden_out)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    batch: list[dict] = []
    BATCH_SIZE = 20

    def _write_golden(result: dict) -> None:
        """Write a detailed ML training record for each confirmed PSLQ hit."""
        rec = result.get("_source_rec", {})
        params = rec.get("params", {})
        for h in result.get("hits", []):
            golden_record = {
                "fingerprint":    result["fingerprint"],
                "agent":          result["agent"],
                "dim":            result["dim"],
                "n_vars":         result.get("n_vars", 2),
                "formula":        h.get("formula"),
                "method":         h.get("method"),
                "residual":       h.get("residual"),
                "ratio_value":    h.get("ratio_value"),
                "start_coords":   h.get("start_coords"),
                "axis":           h.get("axis"),
                "n_trajectories": result.get("n_trajectories"),
                "n_convergent":   result.get("n_convergent"),
                # Full parameter payload for ML feature extraction
                "D_params":       params.get("D_params"),
                "L_off":          params.get("L_off"),
                "U_off":          params.get("U_off"),
                "X0":             rec.get("X0"),
                "X1":             rec.get("X1"),
                "X2":             rec.get("X2"),
                "f_poly":         rec.get("f_poly"),
                "fbar_poly":      rec.get("fbar_poly"),
                "source":         rec.get("source"),
                "certification_level": rec.get("certification_level"),
                "pslq_basis":     _BASIS_NAMES,
            }
            with open(golden_path, "a") as fg:
                fg.write(json.dumps(golden_record, default=str) + "\n")

    def _handle(result: dict) -> None:
        nonlocal n_done, n_hits_total
        n_done += 1
        n_h     = len(result.get("hits", []))
        n_hits_total += n_h

        batch.append(result)
        if len(batch) >= BATCH_SIZE:
            _save_results(batch)
            batch.clear()

        if n_h > 0:
            _write_golden(result)

        result.pop("_source_rec", None)  # strip before main output
        with open(out_path, "a") as fout:
            fout.write(json.dumps(result, default=str) + "\n")

        elapsed = time.time() - t0
        rate    = n_done / max(elapsed, 1)
        eta     = (n_total - n_done) / max(rate, 1e-9)

        if n_h > 0:
            print(f"\n  ✦ [{n_done}/{n_total}] {result['agent']}/{result['dim']}x{result['dim']}"
                  f"  fp={result['fingerprint'][:14]}"
                  f"  conv={result.get('n_convergent',0)}/{result.get('n_trajectories',0)}"
                  f"  hits={n_h}",
                  flush=True)
            for h in result.get("hits", []):
                print(f"      {h.get('formula','')}  [{h.get('method','')}]"
                      f"  residual={h.get('residual',0):.2e}",
                      flush=True)
        elif n_done % 500 == 0:
            print(f"  [{n_done:>6}/{n_total}]  hits={n_hits_total}  "
                  f"{rate:.1f} cmf/s  ETA {eta/60:.1f}min",
                  flush=True)

    if n_jobs == 1:
        for item in work_items:
            _handle(_worker(item))
    else:
        with mp.Pool(n_jobs) as pool:
            for result in pool.imap_unordered(_worker, work_items, chunksize=1):
                _handle(result)

    if batch:
        _save_results(batch)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE  {n_done:,} CMFs  in {elapsed/60:.1f} min")
    print(f"  PSLQ hits : {n_hits_total:,}")
    print(f"  Output    : {out_path}")
    print(f"  DB        : {PSLQ_DB}")

    # ── Quick summary of best hits ────────────────────────────────────────
    try:
        con = sqlite3.connect(PSLQ_DB)
        rows = con.execute("""
            SELECT fingerprint, agent, dim, best_formula, best_residual
            FROM pslq_cmf_summary
            WHERE n_hits > 0
            ORDER BY best_residual ASC
            LIMIT 30
        """).fetchall()
        con.close()
        if rows:
            print(f"\nTop {len(rows)} PSLQ hits by residual:")
            print(f"  {'FP':<16}  {'Agt':<3}  {'dim':<4}  {'Formula':<50}  residual")
            for fp, agt, d, formula, res in rows:
                print(f"  {fp:<16}  {agt:<3}  {d:<4}  {str(formula):<50}  {res:.2e}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
