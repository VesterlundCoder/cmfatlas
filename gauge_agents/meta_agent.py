#!/usr/bin/env python3
"""
meta_agent.py — 3-Layer CMF Meta-Optimization Loop (10 rounds)
===============================================================
Architecture (no OpenAI — statistical critic only):

  Layer 1 – Inner loop   : CMF generation + feature extraction + reward scoring
  Layer 2 – Middle loop  : Frozen benchmark evaluation + partition into TP/FP/FN
  Layer 3 – Outer loop   : Statistical critic proposes challenger weights;
                           champion-challenger gating prevents reward hacking.

Round cycle
  1.  Generate BATCH_SIZE candidates using current champion weights.
  2.  Evaluate irrationality (truth label) for every converging candidate.
  3.  Partition: TP (high score + irrational), FP (high score + rational),
                 FN (low score + irrational), TN (low score + rational).
  4.  Statistical critic:
        - Spearman correlation of each feature with irrationality label.
        - Features over-represented in FP → reduce weight.
        - Features correlated with irrationality → increase weight.
        - Propose challenger weights (unit-norm in L1).
  5.  Validate challenger vs champion on a held-out 30% holdout.
  6.  Promote challenger if precision@K on holdout improves; version the weights.
  7.  Log round summary to meta_agent_log.jsonl.
  8.  Repeat for N_ROUNDS.

Usage
  python3 gauge_agents/meta_agent.py
  python3 gauge_agents/meta_agent.py --rounds 10 --batch 300 --agents A B E K L
  python3 gauge_agents/meta_agent.py --rounds 5 --batch 150 --no-promote
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import mpmath as mp
import numpy as np
from scipy import stats as scipy_stats

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_all_agents import (
    AGENT_CONFIGS,
    sample_params, build_eval_fns,
    t1_pole_check, t2_fast, t3_thorough, t4_flatness,
    fingerprint, bidir_ratio, load_seen,
    DELTA_FAST_MIN, DELTA_FULL_MIN, store_path, SENTINEL,
)
from asymptotic_filter import asymptotic_filter
from reward_engine import (
    classify_limit_strict, fast_rationality_check,
    score_conv_rate, score_ray_stability, score_positive_structure,
    W as DEFAULT_W,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
WEIGHTS_FILE = HERE / "reward_weights.json"
LOG_FILE     = HERE / "meta_agent_log.jsonl"

# ── Defaults ──────────────────────────────────────────────────────────────────
N_ROUNDS     = 10
BATCH_SIZE   = 200    # candidates to evaluate per round
HOLDOUT_FRAC = 0.30   # fraction reserved as held-out validation set
TOP_K_FRAC   = 0.25   # precision@K: top 25% by score
WALK_DPS     = 60
WALK_DEPTH   = 800
STEP_SIZE    = 0.15   # max weight perturbation per round
MIN_WEIGHT   = 0.01   # no weight ever goes below this
PROMOTE_MIN_DELTA = 0.02  # challenger must beat champion by at least 2 pp

# Gate 5
GATE5_REL_TOL = 1e-3  # cross-ray relative agreement threshold

# Between-round deep verifier
VERIFY_DPS   = 80
VERIFY_DEPTH = 2000

# ── KLM focused agents (half-integer shifts + positive slopes) ──────────────
KLM_AGENTS = ["K", "L", "M"]

# ── Feature names (must match _extract_features keys) ────────────────────────
FEATURE_KEYS = [
    "best_delta",
    "ray_stability",
    "n_converging_frac",
    "positive_structure",
    "bidir_ratio",
    "delta_spread_inv",
]

# ── Reward keys (subset of DEFAULT_W that we tune) ────────────────────────────
TUNABLE_KEYS = ["conv_rate", "ray_stability", "identifiability",
                 "simplicity", "proofability", "positive_structure"]


# ══════════════════════════════════════════════════════════════════════════════
# Weight management — champion / challenger
# ══════════════════════════════════════════════════════════════════════════════

def _default_weights() -> dict:
    return {k: DEFAULT_W[k] for k in TUNABLE_KEYS}


def load_champion() -> dict:
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            data = json.load(f)
        w = data.get("weights", {})
        for k in TUNABLE_KEYS:
            if k not in w:
                w[k] = DEFAULT_W[k]
        return w
    return _default_weights()


def save_champion(weights: dict, version: int, round_idx: int,
                  metrics: dict) -> None:
    payload = {
        "version":   version,
        "round":     round_idx,
        "weights":   weights,
        "metrics":   metrics,
        "timestamp": time.time(),
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ Champion promoted → v{version}  "
          f"precision@K={metrics.get('precision_k', 0):.3f}", flush=True)


def _normalise(w: dict) -> dict:
    """L1-normalise tunable weights so they sum to 1."""
    total = sum(max(MIN_WEIGHT, v) for v in w.values())
    return {k: max(MIN_WEIGHT, v) / total for k, v in w.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction from a converging CMF
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features(deltas: list[float], params: dict, br: float) -> dict:
    """
    Return a dict of numerical features for one CMF candidate.
    All values are in [0, 1] or [0, ∞) — normalised for correlation analysis.
    """
    best   = max(deltas) if deltas else 0.0
    valid  = [d for d in deltas if d > 0.5]
    spread = float(np.std(deltas)) if len(deltas) > 1 else 0.0

    return {
        "best_delta":        min(1.0, best / 30.0),
        "ray_stability":     score_ray_stability(deltas),
        "n_converging_frac": len(valid) / max(1, len(deltas)),
        "positive_structure":score_positive_structure(params),
        "bidir_ratio":       min(1.0, br),
        "delta_spread_inv":  1.0 / (1.0 + spread),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Scoring with arbitrary weights
# ══════════════════════════════════════════════════════════════════════════════

def _score(features: dict, weights: dict) -> float:
    """
    Weighted dot-product of features against tunable weights.
    Maps FEATURE_KEYS onto TUNABLE_KEYS in order.
    """
    mapping = dict(zip(FEATURE_KEYS, TUNABLE_KEYS))
    total = 0.0
    for fk, wk in mapping.items():
        total += features.get(fk, 0.0) * weights.get(wk, DEFAULT_W.get(wk, 0.1))
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Gate 5 — Cross-Ray Consistency (inline, fast)
# ══════════════════════════════════════════════════════════════════════════════

def _gate5_cross_ray(fns: list, dim: int, n_vars: int,
                    primary_val: float,
                    dps: int = WALK_DPS,
                    depth: int = 600) -> tuple[bool, list[float]]:
    """
    Walk axes 1..min(n_vars,3)-1, check they all agree with primary_val.
    Returns (consistent, [all_limits]).
    """
    mp.mp.dps = dps + 5
    limits = [primary_val]
    for ax in range(1, min(n_vars, 3)):
        pos = [2] * n_vars
        v   = mp.zeros(dim, 1)
        v[0] = mp.mpf(1)
        fn  = fns[ax % len(fns)]
        ok  = True
        for _ in range(depth):
            pos[ax % n_vars] += 1
            try:
                raw = np.asarray(fn(*pos), dtype=float)
                M   = mp.matrix([[mp.mpf(str(raw[r][c]))
                                   for c in range(dim)] for r in range(dim)])
                v = M * v
            except Exception:
                ok = False; break
            scale = max(abs(v[i]) for i in range(dim))
            if scale > mp.power(10, 40):
                v /= scale
            elif scale < mp.power(10, -40):
                ok = False; break
        if ok and abs(v[dim - 1]) >= mp.power(10, -(dps - 5)):
            limits.append(float(v[0] / v[dim - 1]))

    if len(limits) < 2:
        return True, limits
    spread = max(abs(a - b) for a in limits for b in limits)
    ref    = abs(primary_val) + 1e-30
    return (spread / ref) < GATE5_REL_TOL, limits


# ══════════════════════════════════════════════════════════════════════════════
# Between-round deep verifier
# ══════════════════════════════════════════════════════════════════════════════

def _deep_verify_candidate(candidate: dict) -> bool:
    """
    Re-verify one irrational hit after the inner-loop round.
    Rebuilds fns from stored params, runs:
      - Deeper walk (dps=80, depth=2000)
      - Cross-ray check at depth=1000
      - Slow-rational check via CF depth
    Returns True only if all three checks pass.
    """
    import ast as _ast
    from fractions import Fraction as _Frac

    raw_params = candidate.get("params")
    if raw_params is None:
        return candidate.get("is_irr", False)

    dim    = raw_params["dim"]
    n_vars = raw_params["n_vars"]

    # Reconstruct L_off / U_off keys
    def parse_key(k):
        return _ast.literal_eval(k)

    params = {
        "dim":      dim,
        "n_vars":   n_vars,
        "D_params": [tuple(dp) for dp in raw_params["D_params"]],
        "L_off":    {parse_key(k): v for k, v in raw_params.get("L_off", {}).items()},
        "U_off":    {parse_key(k): v for k, v in raw_params.get("U_off", {}).items()},
    }

    try:
        fns = build_eval_fns(params)
    except Exception:
        return False

    # Deep walk
    mp.mp.dps = VERIFY_DPS + 10
    pos = [2] * n_vars
    v   = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)
    fn  = fns[0]
    ok  = True
    for _ in range(VERIFY_DEPTH):
        pos[0] += 1
        try:
            raw = np.asarray(fn(*pos), dtype=float)
            M   = mp.matrix([[mp.mpf(str(raw[r][c]))
                               for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            ok = False; break
        scale = max(abs(v[i]) for i in range(dim))
        if scale > mp.power(10, 50):
            v /= scale
        elif scale < mp.power(10, -50):
            ok = False; break

    if not ok or abs(v[dim - 1]) < mp.power(10, -(VERIFY_DPS - 5)):
        return False
    hp_val  = v[0] / v[dim - 1]
    hp_float = float(hp_val)

    if not math.isfinite(hp_float):
        return False

    # High-precision rational screen:
    # Check if hp_val matches any rational with denominator <= 10_000
    # at 30+ significant digits. A genuine irrational won't.
    frac = _Frac(hp_float).limit_denominator(10_000)
    frac_mp = mp.mpf(frac.numerator) / mp.mpf(frac.denominator)
    hp_rel = abs(frac_mp - hp_val) / (abs(hp_val) + mp.mpf("1e-30"))
    if hp_rel < mp.mpf("1e-25"):
        return False   # matches small rational at high precision → rational/slow

    # Cross-ray
    g5_ok, _ = _gate5_cross_ray(fns, dim, n_vars, hp_float,
                                 dps=VERIFY_DPS, depth=1000)
    return g5_ok


def between_round_verify(batch: list[dict]) -> list[dict]:
    """
    Deep-verify all is_irr=True candidates from a batch.
    Sets candidate["verified_irr"] = True/False.
    Returns only the candidates that were already marked is_irr=True.
    """
    hits = [c for c in batch if c.get("is_irr", False)]
    if not hits:
        return []
    print(f"  [Verifier] Deep-checking {len(hits)} irrational hits …", flush=True)
    confirmed = 0
    for c in hits:
        ok = _deep_verify_candidate(c)
        c["verified_irr"] = ok
        if ok:
            confirmed += 1
    # Also mark non-hits as verified_irr=False
    for c in batch:
        if "verified_irr" not in c:
            c["verified_irr"] = False
    print(f"  [Verifier] {confirmed}/{len(hits)} hits confirmed "
          f"({100*confirmed/max(1,len(hits)):.0f}% survival rate)", flush=True)
    return hits


def _walk_for_limit(fns: list, dim: int, n_vars: int,
                    dps: int = WALK_DPS,
                    depth: int = WALK_DEPTH) -> Optional[float]:
    """Axis-0 walk — returns float or None."""
    mp.mp.dps = dps + 10
    pos = [2] * n_vars
    v   = mp.zeros(dim, 1)
    v[0] = mp.mpf(1)
    fn  = fns[0]
    for _ in range(depth):
        pos[0] += 1
        try:
            raw = np.asarray(fn(*pos), dtype=float)
            M   = mp.matrix([[mp.mpf(str(raw[r][c]))
                               for c in range(dim)] for r in range(dim)])
            v = M * v
        except Exception:
            return None
        scale = max(abs(v[i]) for i in range(dim))
        if scale > mp.power(10, 40):
            v /= scale
        elif scale < mp.power(10, -40):
            return None
    if abs(v[dim - 1]) < mp.power(10, -(dps - 5)):
        return None
    return float(v[0] / v[dim - 1])


def _is_irrational(fns: list, dim: int, n_vars: int) -> tuple[bool, str]:
    """
    Gates 1-4 irrationality verdict (fast inner-loop screen).
    Gate 5 (cross-ray) is applied in between_round_verify for confirmed hits.
    Returns (is_irrational, label).
    """
    val = _walk_for_limit(fns, dim, n_vars)
    if val is None or not math.isfinite(val):
        return False, "NO_LIMIT"
    if abs(val) < 1e-4 or abs(val) > 1e4:
        return False, "OUT_OF_RANGE"
    if fast_rationality_check(val):
        return False, "TRIVIAL_RATIONAL"
    gate = classify_limit_strict(val, hp_val=mp.mpf(str(val)), verbose=False)
    label = gate["label"]
    is_irr = label in {"IRRATIONAL_UNKNOWN", "TRUE_TRANSCENDENTAL"}
    return is_irr, label


# ══════════════════════════════════════════════════════════════════════════════
# Inner loop — generate one batch of evaluated candidates
# ══════════════════════════════════════════════════════════════════════════════

def generate_batch(agents: list[str], champion_weights: dict,
                   batch_size: int, rng: np.random.Generator,
                   agent_irr_rates: dict) -> list[dict]:
    """
    Sample `batch_size` converging CMFs from `agents`.
    Agent selection is biased toward agents with higher historical irrational rate.
    Returns list of candidate dicts with features + irrationality labels.
    """
    # Weighted agent selection
    rates  = np.array([agent_irr_rates.get(a, 0.5) for a in agents], dtype=float)
    rates  = rates + 0.1   # add small floor so no agent is starved
    probs  = rates / rates.sum()

    candidates   = []
    trials       = 0
    max_trials   = batch_size * 200

    seen_sets = {a: load_seen(a) for a in agents}

    while len(candidates) < batch_size and trials < max_trials:
        if SENTINEL.exists():
            break
        trials += 1

        agent = rng.choice(agents, p=probs)
        cfg   = AGENT_CONFIGS[agent]
        dim   = int(rng.choice(cfg["dims"]))
        n_vars = cfg["n_vars"]

        try:
            params = sample_params(dim, cfg, rng)
            fns    = build_eval_fns(params)
        except Exception:
            continue

        # Deduplication
        fp = fingerprint(fns, dim, n_vars)
        if fp in seen_sets[agent]:
            continue

        # Filters T0-T4
        ok, _ = asymptotic_filter(fns, dim, n_vars)
        if not ok:
            continue
        if not t1_pole_check(fns, dim, n_vars, rng):
            continue
        t2_ok, _ = t2_fast(fns, dim, n_vars)
        if not t2_ok:
            continue
        t3_ok, deltas = t3_thorough(fns, dim, n_vars)
        if not t3_ok:
            continue
        pi_ok, pi_err = t4_flatness(fns, dim, n_vars)

        br = bidir_ratio(fns, dim, n_vars)
        features = _extract_features(deltas, params, br)
        score    = _score(features, champion_weights)

        # Irrationality ground truth (gates 1-5)
        is_irr, irr_label = _is_irrational(fns, dim, n_vars)

        # Serialise params for between-round deep verifier
        serialised_params = {
            "dim":      params["dim"],
            "n_vars":   params["n_vars"],
            "D_params": params["D_params"],
            "L_off":    {str(k): v for k, v in params["L_off"].items()},
            "U_off":    {str(k): v for k, v in params["U_off"].items()},
        }

        candidates.append({
            "agent":        agent,
            "dim":          dim,
            "fp":           fp,
            "features":     features,
            "score":        score,
            "is_irr":       is_irr,
            "verified_irr": False,   # filled in by between_round_verify
            "irr_label":    irr_label,
            "best_delta":   max(deltas) if deltas else 0.0,
            "deltas":       deltas,
            "pi_err":       pi_err,
            "params":       serialised_params,
        })

        seen_sets[agent].add(fp)

        if len(candidates) % 20 == 0:
            print(f"    [{len(candidates)}/{batch_size}] trials={trials}  "
                  f"irr={sum(1 for c in candidates if c['is_irr'])}", flush=True)

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# Middle loop — benchmark partition + metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(candidates: list[dict], weights: dict,
                    top_k_frac: float = TOP_K_FRAC,
                    use_verified: bool = False) -> dict:
    """
    Partition candidates by score+irrationality and compute:
      - precision@K   : fraction of top-K by score that are truly irrational
      - recall        : fraction of irrational CMFs in top-K
      - irr_rate      : global irrational rate in this batch
      - fp_rate       : false-positive rate
    If use_verified=True, uses candidate["verified_irr"] as truth signal.
    """
    # Re-score with given weights (allows challenger evaluation)
    scored = [(c, _score(c["features"], weights)) for c in candidates]
    scored.sort(key=lambda x: -x[1])

    n      = len(scored)
    k      = max(1, int(n * top_k_frac))
    top_k  = scored[:k]
    rest   = scored[k:]

    truth_key = "verified_irr" if use_verified else "is_irr"
    tp = sum(1 for c, _ in top_k if c.get(truth_key, False))
    fp = sum(1 for c, _ in top_k if not c.get(truth_key, False))
    fn = sum(1 for c, _ in rest  if c.get(truth_key, False))
    tn = sum(1 for c, _ in rest  if not c.get(truth_key, False))

    total_irr = tp + fn
    precision_k = tp / max(1, k)
    recall      = tp / max(1, total_irr)
    irr_rate    = total_irr / max(1, n)
    fp_rate     = fp / max(1, k)

    # F1-like combined metric (balanced)
    f1 = 2 * precision_k * recall / max(1e-9, precision_k + recall)

    return {
        "precision_k": precision_k,
        "recall":      recall,
        "irr_rate":    irr_rate,
        "fp_rate":     fp_rate,
        "f1":          f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "k": k, "n": n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Outer loop — statistical critic
# ══════════════════════════════════════════════════════════════════════════════

def statistical_critic(train_set: list[dict],
                       champion_weights: dict,
                       step_size: float = STEP_SIZE,
                       use_verified: bool = False) -> dict:
    """
    Analyse the training partition to propose challenger weights.

    Strategy:
      1. Compute Spearman correlation of each feature with irrationality label.
      2. Identify over-weighted features in false positives (reward leaks).
      3. Identify under-weighted features for true irrationals.
      4. Nudge weights in the direction of correlation, clamped to step_size.
      5. L1-normalise so weights sum to 1.

    If use_verified=True, uses "verified_irr" as truth signal (better quality).
    Returns challenger weights dict.
    """
    if not train_set:
        return champion_weights.copy()

    truth_key = "verified_irr" if use_verified else "is_irr"
    labels   = np.array([1.0 if c.get(truth_key, False) else 0.0 for c in train_set])
    mapping  = dict(zip(FEATURE_KEYS, TUNABLE_KEYS))

    # Spearman correlation: feature → irrationality
    correlations: dict[str, float] = {}
    for fk, wk in mapping.items():
        vals = np.array([c["features"].get(fk, 0.0) for c in train_set])
        if vals.std() < 1e-9:
            correlations[wk] = 0.0
        else:
            rho, _ = scipy_stats.spearmanr(vals, labels)
            correlations[wk] = float(rho) if math.isfinite(rho) else 0.0

    # Identify false positives (high score, not irrational)
    scores   = np.array([c["score"] for c in train_set])
    k        = max(1, int(len(train_set) * TOP_K_FRAC))
    top_idx  = np.argsort(-scores)[:k]
    fp_mask  = np.zeros(len(train_set), dtype=bool)
    fp_mask[top_idx] = True
    fp_mask  = fp_mask & (labels == 0.0)

    # Feature mean in FP vs overall
    fp_deltas: dict[str, float] = {}
    for fk, wk in mapping.items():
        vals    = np.array([c["features"].get(fk, 0.0) for c in train_set])
        overall = vals.mean()
        fp_mean = vals[fp_mask].mean() if fp_mask.any() else overall
        fp_deltas[wk] = fp_mean - overall  # positive → over-represented in FP

    # Propose update: nudge by corr + penalise FP over-representation
    challenger = {}
    for wk in TUNABLE_KEYS:
        corr   = correlations.get(wk, 0.0)
        fp_del = fp_deltas.get(wk, 0.0)
        # Positive correlation with irrationality → boost; FP over-rep → cut
        nudge  = corr * step_size - fp_del * step_size * 0.5
        nudge  = max(-step_size, min(step_size, nudge))
        challenger[wk] = max(MIN_WEIGHT, champion_weights.get(wk, 0.1) + nudge)

    challenger = _normalise(challenger)

    print("  Critic analysis:", flush=True)
    for wk in TUNABLE_KEYS:
        corr = correlations.get(wk, 0.0)
        old  = champion_weights.get(wk, 0.1)
        new  = challenger[wk]
        arrow = "↑" if new > old + 0.001 else ("↓" if new < old - 0.001 else "·")
        print(f"    {wk:22s} corr={corr:+.3f}  "
              f"weight {old:.4f} → {new:.4f}  {arrow}", flush=True)

    return challenger


# ══════════════════════════════════════════════════════════════════════════════
# Agent irrationality rate tracker
# ══════════════════════════════════════════════════════════════════════════════

def update_agent_rates(agent_irr_rates: dict, batch: list[dict]) -> dict:
    """EMA update of per-agent irrational discovery rate."""
    alpha = 0.3
    counts: dict[str, list] = {}
    for c in batch:
        a = c["agent"]
        counts.setdefault(a, []).append(1.0 if c["is_irr"] else 0.0)

    new_rates = dict(agent_irr_rates)
    for a, vals in counts.items():
        rate = float(np.mean(vals))
        old  = new_rates.get(a, 0.5)
        new_rates[a] = alpha * rate + (1.0 - alpha) * old
    return new_rates


# ══════════════════════════════════════════════════════════════════════════════
# Main 10-round loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds",  type=int, default=N_ROUNDS)
    ap.add_argument("--batch",   type=int, default=BATCH_SIZE,
                    help="Candidates to generate per round")
    ap.add_argument("--agents",  nargs="+",
                    default=KLM_AGENTS,
                    help="Agents to use (default: K L M)")
    ap.add_argument("--no-promote", action="store_true",
                    help="Run critic but never promote challenger")
    ap.add_argument("--step",    type=float, default=STEP_SIZE,
                    help="Max weight perturbation per round")
    args = ap.parse_args()

    print(f"\n{'═'*65}", flush=True)
    print(f"  CMF Meta-Agent — {args.rounds} rounds × {args.batch} candidates", flush=True)
    print(f"  Agents: {args.agents}", flush=True)
    print(f"  Holdout: {int(HOLDOUT_FRAC*100)}%  TopK: {int(TOP_K_FRAC*100)}%  "
          f"Promote threshold: +{PROMOTE_MIN_DELTA:.2f}", flush=True)
    print(f"{'═'*65}\n", flush=True)

    rng              = np.random.default_rng(int(time.time()) % 2**31)
    champion_weights = load_champion()
    champion_version = 1
    agent_irr_rates  = {a: 0.5 for a in args.agents}

    # Load existing version number from file
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            d = json.load(f)
        champion_version = d.get("version", 1)

    print("  Initial champion weights:", flush=True)
    for k, v in champion_weights.items():
        print(f"    {k:22s} {v:.4f}", flush=True)
    print(flush=True)

    all_round_logs = []

    for rnd in range(1, args.rounds + 1):
        print(f"{'─'*65}", flush=True)
        print(f"  ROUND {rnd}/{args.rounds}", flush=True)
        print(f"{'─'*65}", flush=True)
        t_round = time.time()

        # ── Layer 1: Generate batch ───────────────────────────────────────────
        print(f"\n  [Layer 1] Generating {args.batch} candidates …", flush=True)
        all_cands = generate_batch(
            args.agents, champion_weights, args.batch, rng, agent_irr_rates
        )

        if not all_cands:
            print("  No candidates generated — skip round.", flush=True)
            continue

        # Update agent rates (using gate 1-5 irrationality)
        agent_irr_rates = update_agent_rates(agent_irr_rates, all_cands)

        irr_count = sum(1 for c in all_cands if c["is_irr"])
        print(f"\n  Batch: {len(all_cands)} candidates, "
              f"{irr_count} gate-passed irrational "
              f"({100*irr_count/max(1,len(all_cands)):.1f}%)", flush=True)

        # ── Between-round deep verification ───────────────────────────────────
        print(f"\n  [Verifier] Between-round deep verification …", flush=True)
        between_round_verify(all_cands)
        verified_count = sum(1 for c in all_cands if c["verified_irr"])
        print(f"  Verified irrational: {verified_count}/{irr_count} "
              f"(survival rate {100*verified_count/max(1,irr_count):.0f}%)",
              flush=True)

        # Print confirmed hits
        confirmed_hits = [c for c in all_cands if c["verified_irr"]]
        if confirmed_hits:
            print(f"  Confirmed hits:", flush=True)
            for h in confirmed_hits:
                print(f"    [{h['agent']}] {h['dim']}×{h['dim']}  "
                      f"fp={h['fp']}  Δ={h['best_delta']:.1f}  "
                      f"{h.get('irr_label','?')}", flush=True)

        # ── Layer 2: Split train/holdout (use verified truth) ─────────────────
        rng.shuffle(all_cands)
        n_holdout  = max(1, int(len(all_cands) * HOLDOUT_FRAC))
        holdout    = all_cands[:n_holdout]
        train_set  = all_cands[n_holdout:]

        champ_metrics = compute_metrics(holdout, champion_weights, use_verified=True)
        print(f"\n  [Layer 2] Champion on holdout ({len(holdout)} cands, verified truth):",
              flush=True)
        print(f"    precision@K={champ_metrics['precision_k']:.3f}  "
              f"recall={champ_metrics['recall']:.3f}  "
              f"F1={champ_metrics['f1']:.3f}  "
              f"irr_rate={champ_metrics['irr_rate']:.3f}", flush=True)

        # ── Layer 3: Statistical critic (verified truth) ──────────────────────
        print(f"\n  [Layer 3] Statistical critic on train set "
              f"({len(train_set)} cands, verified truth) …", flush=True)
        challenger_weights = statistical_critic(train_set, champion_weights,
                                                args.step, use_verified=True)

        # Validate challenger on holdout
        chal_metrics = compute_metrics(holdout, challenger_weights, use_verified=True)
        print(f"\n  Challenger on holdout (verified truth):", flush=True)
        print(f"    precision@K={chal_metrics['precision_k']:.3f}  "
              f"recall={chal_metrics['recall']:.3f}  "
              f"F1={chal_metrics['f1']:.3f}", flush=True)

        # Promotion decision
        gain = chal_metrics["f1"] - champ_metrics["f1"]
        promoted = False
        if gain >= PROMOTE_MIN_DELTA and not args.no_promote:
            champion_version += 1
            champion_weights  = challenger_weights
            save_champion(champion_weights, champion_version, rnd, chal_metrics)
            promoted = True
            print(f"  ✅ PROMOTED challenger v{champion_version} (F1 gain={gain:+.3f})",
                  flush=True)
        else:
            reason = "below threshold" if gain < PROMOTE_MIN_DELTA else "--no-promote flag"
            print(f"  · Champion retained  (F1 gain={gain:+.3f} — {reason})", flush=True)

        # Agent rate summary
        print(f"\n  Agent irrational rates:", flush=True)
        for a in sorted(agent_irr_rates):
            print(f"    {a}: {agent_irr_rates[a]:.3f}", flush=True)

        round_elapsed = round(time.time() - t_round, 1)

        # Log round
        round_log = {
            "round":              rnd,
            "n_candidates":       len(all_cands),
            "n_irrational":       irr_count,
            "n_verified":         verified_count,
            "verified_rate":      verified_count / max(1, len(all_cands)),
            "irr_rate":           irr_count / max(1, len(all_cands)),
            "champion_metrics":   champ_metrics,
            "challenger_metrics": chal_metrics,
            "f1_gain":            gain,
            "promoted":           promoted,
            "champion_version":   champion_version,
            "champion_weights":   dict(champion_weights),
            "agent_irr_rates":    dict(agent_irr_rates),
            "elapsed_s":          round_elapsed,
            "timestamp":          time.time(),
        }
        all_round_logs.append(round_log)
        with open(LOG_FILE, "a") as lf:
            lf.write(json.dumps(round_log) + "\n")

        print(f"\n  Round {rnd} complete — {round_elapsed}s", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}", flush=True)
    print(f"  META-AGENT COMPLETE — {args.rounds} rounds", flush=True)
    print(f"{'═'*65}", flush=True)

    n_promoted   = sum(1 for r in all_round_logs if r["promoted"])
    total_irr    = sum(r["n_irrational"] for r in all_round_logs)
    total_verif  = sum(r.get("n_verified", 0) for r in all_round_logs)
    total_cand   = sum(r["n_candidates"] for r in all_round_logs)

    print(f"  Promotions:             {n_promoted}/{args.rounds}", flush=True)
    print(f"  Total candidates:       {total_cand}", flush=True)
    print(f"  Gate-passed irrational: {total_irr}  "
          f"({100*total_irr/max(1,total_cand):.1f}%)", flush=True)
    print(f"  Deep-verified irrational: {total_verif}  "
          f"({100*total_verif/max(1,total_irr):.0f}% survival)", flush=True)
    print(f"\n  Final champion weights (v{champion_version}):", flush=True)
    for k, v in champion_weights.items():
        print(f"    {k:22s} {v:.4f}", flush=True)

    if all_round_logs:
        best_rnd = max(all_round_logs, key=lambda r: r.get("n_verified", 0))
        print(f"\n  Best round: {best_rnd['round']}  "
              f"F1={best_rnd['champion_metrics']['f1']:.3f}  "
              f"irr_rate={best_rnd['irr_rate']:.3f}", flush=True)

    # Per-agent summary
    agent_totals: dict[str, list] = {}
    for rnd_log in all_round_logs:
        for a, rate in rnd_log["agent_irr_rates"].items():
            agent_totals.setdefault(a, []).append(rate)

    print(f"\n  Final agent irrational rates (EMA):", flush=True)
    sorted_agents = sorted(agent_totals, key=lambda a: -np.mean(agent_totals[a]))
    for a in sorted_agents:
        rates = agent_totals[a]
        print(f"    {a}: {np.mean(rates):.3f}  (last={rates[-1]:.3f})", flush=True)

    print(f"\n  Log saved → {LOG_FILE}", flush=True)
    print(f"  Weights  → {WEIGHTS_FILE}", flush=True)
    print(f"{'═'*65}\n", flush=True)


if __name__ == "__main__":
    main()
