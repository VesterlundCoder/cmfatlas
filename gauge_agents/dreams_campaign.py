#!/usr/bin/env python3
"""
dreams_campaign.py — 50-seed Ramanujan Dreams exploration campaign
====================================================================
10x-scaled trajectory budget, diversity-aware seed selection,
5-checkpoint PSLQ + self-delta logging, full metrics schema.

Usage:
  python3 dreams_campaign.py [--seeds-file dreams_seeds.json]
                             [--focused-basis] [--dps 80]
                             [--resume] [--seed-filter PATTERN]
                             [--dry-run]
"""
from __future__ import annotations
import argparse, json, logging, math, os, random, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import mpmath as mp

HERE    = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dreams_runner import (
    build_fns, _make_focused_basis, _make_basis, _pslq_identify,
)

OUT_DIR = HERE / "campaign_out"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "campaign.log"),
    ],
)
log = logging.getLogger("campaign")

# ── Global config (overridden by CLI) ─────────────────────────────────────────
DPS      = 80
PSLQ_TOL = 1e-15

TIER_BUDGETS: dict = {
    "S": {"n_traj": 100, "depth_deep": 2000, "depth_shallow": 500},
    "A": {"n_traj": 80,  "depth_deep": 1500, "depth_shallow": 400},
    "B": {"n_traj": 60,  "depth_deep": 1200, "depth_shallow": 300},
    "C": {"n_traj": 50,  "depth_deep": 1000, "depth_shallow": 200},
}

CP_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.00]


# ══════════════════════════════════════════════════════════════════════════════
# Walk engine with 5-checkpoint recording
# ══════════════════════════════════════════════════════════════════════════════

def _walk_checkpointed(
    fns, dim: int, n_ax: int,
    ray: tuple, start: list,
    total_depth: int,
) -> list[dict]:
    """
    Walk to total_depth, recording limit estimate at 5 checkpoints.
    Returns list of 5 dicts: {fraction, step, limit, converged}.
    """
    mp.mp.dps = DPS + 10
    pos = list(start)
    v   = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)

    cp_steps = [max(1, int(total_depth * f)) for f in CP_FRACTIONS]
    cp_steps[0] = 0

    def _ratio():
        denom = abs(v[dim - 1])
        if denom < mp.power(10, -(DPS - 8)):
            return None
        return v[0] / v[dim - 1]

    checkpoints: list[dict] = []
    cp_idx = 0
    failed = False

    def _record_cp(step_actual: int, fraction: float):
        r = _ratio()
        checkpoints.append({
            "fraction":  fraction,
            "step":      step_actual,
            "limit":     float(mp.re(r)) if r is not None else None,
            "converged": r is not None,
        })

    # checkpoint 0 (start)
    _record_cp(0, 0.0)
    cp_idx = 1

    for step in range(1, total_depth + 1):
        ax = step % n_ax
        pos[ax] += ray[ax] if ax < len(ray) else 1
        try:
            Mr = fns[ax](*pos)
            M = mp.matrix([
                [mp.mpf(str(float(
                    Mr[r][c] if hasattr(Mr[r], "__len__") else Mr[r, c]
                ))) for c in range(dim)]
                for r in range(dim)
            ])
            v = M * v
        except Exception:
            failed = True
            break

        sc = max(abs(v[i]) for i in range(dim))
        if sc > mp.power(10, 30):
            v /= sc
        elif sc < mp.power(10, -30):
            failed = True
            break

        while cp_idx < 5 and step >= cp_steps[cp_idx]:
            _record_cp(step, CP_FRACTIONS[cp_idx])
            cp_idx += 1

    while cp_idx < 5:
        _record_cp(total_depth, CP_FRACTIONS[cp_idx])
        cp_idx += 1

    return checkpoints


# ══════════════════════════════════════════════════════════════════════════════
# PSLQ at a single value
# ══════════════════════════════════════════════════════════════════════════════

def _pslq_at_checkpoint(
    limit_float: Optional[float],
    basis_vals: list,
    basis_names: list,
) -> Optional[dict]:
    if limit_float is None or not math.isfinite(limit_float):
        return None
    try:
        lv_mp = mp.mpf(str(limit_float))
        match = _pslq_identify(lv_mp, basis_vals, basis_names, DPS, PSLQ_TOL)
        if match:
            return {
                "formula":    match["formula"],
                "residual":   match["residual"],
                "confidence": "numerically_plausible",
                "precision_dps": DPS,
            }
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Self-delta on linear combinations
# ══════════════════════════════════════════════════════════════════════════════

def _self_delta(
    fns, dim: int, n_ax: int,
    ray: tuple, start: list,
    depth: int,
) -> Optional[float]:
    """
    Self-delta: |L(start) - L(start+1)| / max(|L(start)|, 1).
    Measures limit stability across two adjacent starting positions.
    Evaluated at checkpoint fraction 0.25 of depth for speed.
    """
    probe_depth = max(20, depth // 4)
    results = []
    for offset in (0, 1):
        s2 = [s + offset for s in start]
        cps = _walk_checkpointed(fns, dim, n_ax, ray, s2, probe_depth)
        final = cps[-1]["limit"] if cps else None
        results.append(final)
    if None in results:
        return None
    diff  = abs(results[0] - results[1])
    scale = max(abs(results[0]), abs(results[1]), 1.0)
    return float(diff / scale)


# Self-delta on all axis linear combinations (e_i ± e_j rays)
def _self_delta_lincombs(
    fns, dim: int, n_ax: int,
    start: list,
    depth: int,
) -> dict:
    """
    Compute self-delta for standard rays and selected ±-combos of axes.
    Returns dict: {ray_label: self_delta_value}.
    """
    probe = max(20, depth // 4)
    rays: list[tuple] = []
    labels: list[str] = []

    for ax in range(min(n_ax, 4)):
        r = tuple(1 if k == ax else 0 for k in range(n_ax))
        rays.append(r);  labels.append(f"e{ax}")

    for i in range(min(n_ax, 3)):
        for j in range(i + 1, min(n_ax, 3)):
            r = tuple(1 if k in (i, j) else 0 for k in range(n_ax))
            rays.append(r);  labels.append(f"e{i}+e{j}")

    results: dict = {}
    for ray, lbl in zip(rays, labels):
        try:
            sd = _self_delta(fns, dim, n_ax, ray, start, probe)
            results[lbl] = sd
        except Exception:
            results[lbl] = None
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory mode → (start, ray)
# ══════════════════════════════════════════════════════════════════════════════

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _std_rays(n_ax: int) -> list[tuple]:
    rays = [tuple(1 if k == ax else 0 for k in range(n_ax)) for ax in range(min(n_ax, 3))]
    if n_ax >= 2:
        rays += [tuple(1 if k in (i, j) else 0 for k in range(n_ax))
                 for i in range(min(n_ax, 3)) for j in range(i+1, min(n_ax, 3))]
    return rays or [(1,) + (0,) * (n_ax - 1)]

def _traj_params(mode: str, n_ax: int, rng: random.Random) -> tuple[list, tuple]:
    rays = _std_rays(n_ax)
    pad  = lambda lst: (lst + [3] * n_ax)[:n_ax]

    if mode == "exploit":
        starts = [pad([3,3,3]), pad([4,2,5]), pad([2,5,3]), pad([5,3,4])]
        return rng.choice(starts), rng.choice(rays[:3] or rays)

    if mode == "padic":
        p_start = [_PRIMES[i % len(_PRIMES)] for i in range(n_ax)]
        p_start2 = [_PRIMES[(i+3) % len(_PRIMES)] for i in range(n_ax)]
        return rng.choice([p_start, p_start2]), rng.choice(rays[:3] or rays)

    if mode == "dio":
        base = rng.choice([2, 3, 4, 5])
        return [base] * n_ax, rng.choice(rays)

    if mode == "congruence":
        base = rng.choice([6, 7, 8, 9, 12])
        return [base] * n_ax, rng.choice(rays[:3] or rays)

    if mode == "ore":
        return pad([3, 4, 3]), rng.choice(rays)

    if mode == "novelty":
        return [rng.randint(2, 12) for _ in range(n_ax)], rng.choice(rays)

    if mode == "crossdim":
        return [3 + (i * 2) % 7 for i in range(n_ax)], rng.choice(rays)

    if mode == "anti_score":
        return [rng.randint(1, 4) for _ in range(n_ax)], rng.choice(rays)

    if mode == "long_random":
        return [rng.randint(3, 15) for _ in range(n_ax)], rng.choice(rays)

    if mode == "pslq_chase":
        starts = [pad([3,3,3]), pad([4,2,5]), pad([2,5,3])]
        return rng.choice(starts), rng.choice(rays[:3] or rays)

    # fallback
    return pad([3, 3, 3]), rays[0]


# ══════════════════════════════════════════════════════════════════════════════
# Run one trajectory
# ══════════════════════════════════════════════════════════════════════════════

def run_trajectory(
    rec: dict,
    seed_meta: dict,
    traj_idx: int,
    mode: str,
    basis_vals: list,
    basis_names: list,
    rng: random.Random,
) -> dict:
    tier   = seed_meta.get("tier", "C")
    budget = TIER_BUDGETS[tier]
    # 2 out of 3 trajectories are deep, 1 is shallow (screening)
    depth  = budget["depth_deep"] if traj_idx % 3 != 0 else budget["depth_shallow"]
    dim    = rec.get("dim", 3)
    t0     = time.time()

    try:
        fns, n_ax = build_fns(rec)
    except Exception as e:
        return {
            "seed_id": seed_meta["id"], "traj_idx": traj_idx, "mode": mode,
            "error": str(e), "elapsed": 0.0,
        }

    start, ray = _traj_params(mode, n_ax, rng)
    ray = tuple(ray)

    # ── 5-checkpoint walk ────────────────────────────────────────────
    checkpoints = _walk_checkpointed(fns, dim, n_ax, ray, start, depth)

    # ── PSLQ at each checkpoint ──────────────────────────────────────
    for cp in checkpoints:
        cp["pslq"] = _pslq_at_checkpoint(
            cp["limit"], basis_vals, basis_names
        )

    # ── Self-delta on all linear combinations ────────────────────────
    sd_main   = _self_delta(fns, dim, n_ax, ray, start, depth)
    sd_lincombs = _self_delta_lincombs(fns, dim, n_ax, start, depth)

    # ── Stability: relative drift from 50%→100% limit ───────────────
    lims = [cp["limit"] for cp in checkpoints if cp["limit"] is not None]
    stability = None
    if len(lims) >= 2:
        stability = abs(lims[-1] - lims[-2]) / max(abs(lims[-1]), 1.0)

    # ── Novelty distance (rough proxy: abs difference from best known) ─
    best_known = seed_meta.get("known_limit_float")
    novelty_dist = None
    if best_known is not None and checkpoints[-1]["limit"] is not None:
        novelty_dist = abs(checkpoints[-1]["limit"] - best_known)

    final_limit = checkpoints[-1]["limit"]
    all_pslq    = [cp["pslq"] for cp in checkpoints if cp["pslq"]]
    top_formula = all_pslq[0]["formula"] if all_pslq else None

    elapsed = time.time() - t0

    return {
        # ── Identity ──────────────────────────────────────────────────
        "seed_id":  seed_meta["id"],
        "seed_fp":  rec.get("fingerprint", "")[:16],
        "agent":    rec.get("agent", "?"),
        "dim":      dim,
        "bucket":   rec.get("coupling_bucket", "?"),
        "tier":     tier,
        "style":    seed_meta.get("style", ""),
        # ── Trajectory ────────────────────────────────────────────────
        "traj_idx": traj_idx,
        "mode":     mode,
        "start":    start,
        "ray":      list(ray),
        "depth":    depth,
        "n_ax":     n_ax,
        # ── Scores from seed ──────────────────────────────────────────
        "seed_total_score":   seed_meta.get("total_score"),
        "seed_A_score":       seed_meta.get("A_score"),
        "seed_B_score":       seed_meta.get("B_score"),
        "seed_C_score":       seed_meta.get("C_score"),
        "seed_P_score":       seed_meta.get("P_score"),
        "seed_DIO_score":     seed_meta.get("DIO_score"),
        "seed_congruence_depth": seed_meta.get("congruence_depth"),
        "seed_ore":           seed_meta.get("ore"),
        "seed_delta":         seed_meta.get("best_delta"),
        # ── 5 checkpoints with PSLQ ───────────────────────────────────
        "checkpoints": checkpoints,
        # ── Self-delta ────────────────────────────────────────────────
        "self_delta_main":    sd_main,
        "self_delta_lincombs": sd_lincombs,
        # ── Final limit ───────────────────────────────────────────────
        "final_limit":   final_limit,
        "stability":     stability,
        "novelty_dist":  novelty_dist,
        "top_formula":   top_formula,
        # ── Promotion signals ─────────────────────────────────────────
        "promote_pslq":      bool(all_pslq),
        "promote_stability": stability is not None and stability < 1e-6,
        "promote_self_delta": sd_main is not None and sd_main < 1e-4,
        "promote_formula_match": top_formula is not None,
        # ── Bonus flags from seed ─────────────────────────────────────
        "seed_bonus_flags":  seed_meta.get("bonus_flags", []),
        "seed_symbolic_flat": seed_meta.get("symbolic_flat", False),
        "seed_identified_limit": seed_meta.get("identified_limit"),
        # ── Timing ────────────────────────────────────────────────────
        "elapsed":  elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Process one seed (all trajectories)
# ══════════════════════════════════════════════════════════════════════════════

def process_seed(
    seed_meta: dict,
    records:   dict,
    basis_vals: list,
    basis_names: list,
    out_fh,
    skip_done: set,
) -> list[dict]:
    seed_id = seed_meta["id"]
    tier    = seed_meta.get("tier", "C")
    budget  = TIER_BUDGETS[tier]
    n_traj  = budget["n_traj"]
    modes   = seed_meta.get("mode_mix", ["exploit"] * n_traj)

    rec = records.get(seed_id)
    if rec is None:
        log.warning("Seed %s not found in store files — skipping", seed_id)
        return []

    rng = random.Random(hash(seed_id) & 0xFFFFFFFF)
    results: list[dict] = []

    for ti in range(n_traj):
        tkey = f"{seed_id}:{ti}"
        if tkey in skip_done:
            continue
        mode   = modes[ti % len(modes)]
        result = run_trajectory(rec, seed_meta, ti, mode, basis_vals, basis_names, rng)
        results.append(result)

        out_fh.write(json.dumps(result, default=str) + "\n")
        out_fh.flush()

        # ── Live per-trajectory summary ──────────────────────────────
        lstr  = f"{result.get('final_limit'):.8f}" if result.get("final_limit") is not None else "        —"
        pslq  = "✓ " + result.get("top_formula", "?")[:24] if result.get("promote_pslq") else "·"
        sd    = f"{result['self_delta_main']:.2e}" if result.get("self_delta_main") is not None else "     —"
        stab  = f"{result['stability']:.2e}" if result.get("stability") is not None else "     —"
        log.info("  [%4d/%d] %-14s  lim=%-20s  pslq=%-26s  sd=%s  stab=%s  (%.1fs)",
                 ti + 1, n_traj, mode, lstr, pslq, sd, stab, result.get("elapsed", 0))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Promotion rules
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_promotions(seed_meta: dict, traj_results: list[dict]) -> dict:
    sid    = seed_meta["id"]
    n      = len(traj_results)
    if n == 0:
        return {"seed_id": sid, "n_traj": 0, "promote_level": "NONE"}

    n_pslq   = sum(1 for r in traj_results if r.get("promote_pslq"))
    n_stable = sum(1 for r in traj_results if r.get("promote_stability"))
    n_sdelta = sum(1 for r in traj_results if r.get("promote_self_delta"))

    # Formula reproducibility
    formulas: dict = {}
    for r in traj_results:
        f = r.get("top_formula")
        if f:
            formulas[f] = formulas.get(f, 0) + 1
    top_formula   = max(formulas, key=formulas.get) if formulas else None
    formula_count = formulas.get(top_formula, 0) if top_formula else 0

    # Multi-trajectory convergence: fraction of runs that converge to same limit ±1e-6
    limits = [r.get("final_limit") for r in traj_results if r.get("final_limit") is not None]
    convergent_runs = 0
    if limits:
        median_lim = sorted(limits)[len(limits) // 2]
        convergent_runs = sum(1 for l in limits if abs(l - median_lim) < 1e-6)

    # Promotion level
    if n_pslq >= 10 or formula_count >= 5 or convergent_runs > n * 0.5:
        level = "PAPER"
    elif n_pslq >= 3 or n_stable >= 5 or formula_count >= 2:
        level = "DEEP"
    elif n_pslq >= 1 or n_stable >= 2 or n_sdelta >= 5:
        level = "WATCH"
    else:
        level = "NONE"

    return {
        "seed_id":        sid,
        "tier":           seed_meta.get("tier", "?"),
        "style":          seed_meta.get("style", ""),
        "dim":            seed_meta.get("dim"),
        "n_traj":         n,
        "n_pslq_hits":    n_pslq,
        "n_stable":       n_stable,
        "n_self_delta":   n_sdelta,
        "top_formula":    top_formula,
        "formula_count":  formula_count,
        "convergent_runs": convergent_runs,
        "promote_level":  level,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Load seeds and CMF records
# ══════════════════════════════════════════════════════════════════════════════

def load_seeds(seeds_file: Path) -> list[dict]:
    raw    = json.loads(seeds_file.read_text())
    seeds  = raw["seeds"]
    # Merge per-seed scores from global_summary.csv if available
    csv_p  = HERE.parent / "cmf_analysis_results" / "global_summary.csv"
    if csv_p.exists():
        import csv as _csv
        score_map: dict = {}
        with open(csv_p) as f:
            for row in _csv.DictReader(f):
                score_map[row["cmf_id"]] = row
        # Load identified_limits from summary.json files
        limit_map: dict = {}
        results_dir = HERE.parent / "cmf_analysis_results"
        for d in results_dir.iterdir():
            sj = d / "summary.json"
            if not sj.exists(): continue
            try:
                data = json.loads(sj.read_text())
                lv   = data.get("identified_limit")
                cid  = data.get("cmf_id") or data.get("id")
                if cid and lv is not None:
                    limit_map[cid] = lv
            except Exception:
                pass
        for s in seeds:
            sid = s["id"]
            if sid in score_map:
                row = score_map[sid]
                def _f(k): 
                    try: return float(row.get(k) or 0)
                    except: return None
                s.update({
                    "total_score":      _f("total_score"),
                    "A_score":          _f("A_score"),
                    "B_score":          _f("B_score"),
                    "C_score":          _f("C_score"),
                    "P_score":          _f("P_score"),
                    "DIO_score":        _f("DIO_score"),
                    "congruence_depth": _f("congruence_depth"),
                    "ore":              _f("ore_compression"),
                    "best_delta":       _f("best_delta"),
                    "bonus_flags":      row.get("bonus_flags", []),
                    "symbolic_flat":    row.get("symbolic_flat", "False") == "True",
                })
            if sid in limit_map:
                s["identified_limit"] = limit_map[sid]
                try:
                    s["known_limit_float"] = float(eval(str(limit_map[sid]).replace(" ", "")))
                except Exception:
                    pass
    return seeds


def load_all_records() -> dict:
    records: dict = {}
    for store_file in sorted(HERE.glob("store_*.jsonl")):
        for line in store_file.read_text().splitlines():
            if not line.strip(): continue
            try:
                rec  = json.loads(line)
                fp16 = rec.get("fingerprint", "")[:16]
                ag   = rec.get("agent", "?")
                rid  = f"gauge_{ag}_{fp16}"
                records[rid] = rec
                records[fp16] = rec
            except Exception:
                pass
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="50-seed Ramanujan Dreams campaign")
    ap.add_argument("--seeds-file", type=Path,
                    default=HERE / "dreams_seeds.json")
    ap.add_argument("--focused-basis", action="store_true",
                    help="Use 8-element focused basis (reduces PSLQ overfitting)")
    ap.add_argument("--dps", type=int, default=80,
                    help="mpmath decimal places (default 80)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-completed trajectories")
    ap.add_argument("--seed-filter", type=str, default="",
                    help="Only run seeds whose ID contains FILTER")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit without running")
    args = ap.parse_args()

    global DPS
    DPS = args.dps
    mp.mp.dps = DPS + 20

    basis_vals, basis_names = (
        _make_focused_basis(DPS) if args.focused_basis else _make_basis(DPS)
    )

    seeds   = load_seeds(args.seeds_file)
    records = load_all_records()

    total_traj = sum(TIER_BUDGETS[s.get("tier","C")]["n_traj"] for s in seeds)

    log.info("════════════════════════════════════════════════════════════════════")
    log.info("  Dreams Campaign — %d seeds  %d total trajectories  DPS=%d",
             len(seeds), total_traj, DPS)
    log.info("  Seeds file: %s", args.seeds_file)
    log.info("  Output dir: %s", OUT_DIR)
    log.info("════════════════════════════════════════════════════════════════════")

    if args.dry_run:
        for s in seeds:
            b = TIER_BUDGETS[s.get("tier","C")]
            log.info("  [%-1s] %-34s  dim=%2d  n_traj=%3d  depth_deep=%d",
                     s["tier"], s["id"][:34], s.get("dim",0), b["n_traj"], b["depth_deep"])
        log.info("Total trajectories: %d", total_traj)
        return

    out_file   = OUT_DIR / "campaign_results.jsonl"
    promo_file = OUT_DIR / "promotions.json"
    plan_file  = OUT_DIR / "campaign_plan.json"

    # Save the plan
    plan_file.write_text(json.dumps({
        "seeds":        seeds,
        "total_traj":   total_traj,
        "tier_budgets": TIER_BUDGETS,
        "dps":          DPS,
        "focused_basis": args.focused_basis,
    }, indent=2))

    # Resume: collect already-done keys
    skip_done: set = set()
    if args.resume and out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                r = json.loads(line)
                skip_done.add(f"{r['seed_id']}:{r['traj_idx']}")
            except Exception:
                pass
        log.info("  Resume: %d trajectories already completed", len(skip_done))

    all_promotions: list[dict] = []
    campaign_t0 = time.time()

    with open(out_file, "a" if args.resume else "w") as out_fh:
        for si, seed_meta in enumerate(seeds):
            sid = seed_meta["id"]
            if args.seed_filter and args.seed_filter not in sid:
                continue

            tier   = seed_meta.get("tier", "C")
            style  = seed_meta.get("style", "—")
            budget = TIER_BUDGETS[tier]

            log.info("")
            log.info("[%3d/%d] %s", si + 1, len(seeds), sid)
            log.info("         tier=%-2s  style=%-30s  n_traj=%d  "
                     "depth_deep=%d  depth_shallow=%d",
                     tier, style, budget["n_traj"],
                     budget["depth_deep"], budget["depth_shallow"])

            traj_results = process_seed(
                seed_meta, records, basis_vals, basis_names, out_fh, skip_done
            )
            promo = evaluate_promotions(seed_meta, traj_results)
            all_promotions.append(promo)

            lvl = promo["promote_level"]
            if lvl != "NONE":
                log.info("         ★ PROMOTE [%s]  pslq=%d  formula=%s  convergent=%d",
                         lvl, promo["n_pslq_hits"],
                         promo["top_formula"] or "—",
                         promo["convergent_runs"])

    # Write promotions
    promo_file.write_text(json.dumps(all_promotions, indent=2))

    elapsed_total = time.time() - campaign_t0
    promoted = [p for p in all_promotions if p["promote_level"] != "NONE"]

    log.info("")
    log.info("════════════════════════════════════════════════════════════════════")
    log.info("  Campaign complete in %.1f min", elapsed_total / 60)
    log.info("  Seeds promoted: %d / %d", len(promoted), len(all_promotions))
    for p in sorted(promoted, key=lambda x: -x["n_pslq_hits"])[:15]:
        log.info("    %-36s [%s]  pslq=%d  %-20s  conv=%d",
                 p["seed_id"][:36], p["promote_level"],
                 p["n_pslq_hits"], p.get("top_formula") or "—",
                 p["convergent_runs"])
    log.info("  Results: %s", out_file)
    log.info("  Promotions: %s", promo_file)
    log.info("════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
