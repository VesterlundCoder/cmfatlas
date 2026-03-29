"""Multi-objective ranking function.

Score = Q^α · N^β · P^γ · R^δ

Where:
    Q = quality (convergence + stability + error)
    N = novelty (1 - local_density / max_density)
    P = proof-likelihood (integrality, simplicity, Apéry-like signals)
    R = recognizability penalty (1 if unidentified, 0 if identified)

Also provides Pareto frontier extraction.
"""

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from cmf_atlas.density.heatmaps import _load_features_df


# ── Default exponents ────────────────────────────────────────────────────
DEFAULT_ALPHA = 1.2   # quality matters most
DEFAULT_BETA = 1.0    # novelty
DEFAULT_GAMMA = 1.0   # proof signals
DEFAULT_DELTA = 0.8   # reward "still unidentified"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_quality(feat: dict, E: float = 50.0) -> float:
    """Q = 0.45*conv + 0.35*stability + 0.20*clip01(-log10_error / E)."""
    conv = feat.get("conv_score") or 0.0
    stab = feat.get("stability_score") or 0.0
    log_err = feat.get("log10_error")
    if log_err is not None and log_err < 0:
        err_term = _clip01(-log_err / E)
    else:
        err_term = 0.0

    return 0.45 * conv + 0.35 * stab + 0.20 * err_term


def compute_novelty(feat: dict, max_density: float = 100.0) -> float:
    """N = 1 - local_density / max_density.

    Uses novelty_score if pre-computed, otherwise falls back to
    complexity-based heuristic.
    """
    if "novelty_score" in feat and feat["novelty_score"] is not None:
        return float(feat["novelty_score"])

    # Fallback: higher complexity = slightly more novel (weak heuristic)
    cs = feat.get("complexity_score", 0.0) or 0.0
    return _clip01(cs / 10.0)


def compute_proof_signal(feat: dict) -> float:
    """P = proof-likelihood signal based on arithmetic structure.

    Group-specific heuristics:
    - D-finite: low order + low degree + integrality
    - Hypergeometric: low degree balance + small coefficients
    - PCF: low degrees + small coefficients + Apéry-like patterns
    """
    group = feat.get("primary_group", "")

    if group == "dfinite":
        order = feat.get("rec_order", 10) or 10
        deg = feat.get("max_poly_degree", 10) or 10
        integ = feat.get("integrality_rate") or 0.0

        # Low complexity = higher proof chance
        simplicity = _clip01(1.0 - (order * deg) / 50.0)
        return 0.5 * simplicity + 0.5 * integ

    elif group == "hypergeometric":
        degP = feat.get("degP", 5) or 5
        degQ = feat.get("degQ", 5) or 5
        balance = abs(feat.get("deg_balance", 0) or 0)

        simplicity = _clip01(1.0 - (degP + degQ) / 12.0)
        balance_score = _clip01(1.0 - balance / 4.0)
        return 0.5 * simplicity + 0.5 * balance_score

    elif group == "pcf":
        deg_a = feat.get("deg_a", 5) or 5
        deg_b = feat.get("deg_b", 5) or 5
        coeff_max = feat.get("coeff_log10_max", 3.0) or 3.0

        simplicity = _clip01(1.0 - (deg_a + deg_b) / 10.0)
        small_coeffs = _clip01(1.0 - coeff_max / 6.0)
        return 0.5 * simplicity + 0.5 * small_coeffs

    return 0.5


def compute_recognizability_reward(feat: dict) -> float:
    """R = 1 if unidentified (reward), 0 if identified (no reward).

    Near-misses get partial reward.
    """
    recognized = feat.get("recognized")
    if recognized is None:
        return 0.5  # Unknown state, neutral

    if recognized:
        return 0.0  # Already known — no novelty reward

    # Not recognized: check if there's a near-miss
    best_res = feat.get("best_residual_log10")
    if best_res is not None and best_res < -20:
        # Very close to something known — small penalty
        return 0.3

    return 1.0  # Fully unidentified — maximum reward


def compute_score(
    feat: dict,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    max_density: float = 100.0,
) -> float:
    """Compute the composite ranking score S = Q^α · N^β · P^γ · R^δ.

    All components are clipped to [epsilon, 1] before exponentiation
    to avoid zero-product collapse.
    """
    eps = 0.01

    Q = max(eps, compute_quality(feat))
    N = max(eps, compute_novelty(feat, max_density))
    P = max(eps, compute_proof_signal(feat))
    R = max(eps, compute_recognizability_reward(feat))

    return (Q ** alpha) * (N ** beta) * (P ** gamma) * (R ** delta)


def rank_representations(
    session,
    group: str | None = None,
    top_k: int = 50,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
) -> pd.DataFrame:
    """Rank all representations by composite score.

    Parameters
    ----------
    session : SQLAlchemy session
    group : if specified, rank only this group; otherwise rank all
    top_k : return top K results
    alpha, beta, gamma, delta : scoring exponents

    Returns
    -------
    DataFrame with representation_id, primary_group, Q, N, P, R, score
    """
    groups = [group] if group else ["dfinite", "hypergeometric", "pcf"]

    records = []
    for g in groups:
        df = _load_features_df(session, g)
        if df.empty:
            continue

        for _, row in df.iterrows():
            feat = row.to_dict()
            Q = compute_quality(feat)
            N = compute_novelty(feat)
            P = compute_proof_signal(feat)
            R = compute_recognizability_reward(feat)
            S = compute_score(feat, alpha, beta, gamma, delta)

            records.append({
                "representation_id": feat.get("representation_id"),
                "primary_group": g,
                "Q": round(Q, 4),
                "N": round(N, 4),
                "P": round(P, 4),
                "R": round(R, 4),
                "score": round(S, 6),
            })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return result


def pareto_frontier(
    session,
    group: str | None = None,
    objectives: list[str] | None = None,
) -> pd.DataFrame:
    """Extract the Pareto frontier on (Q, N, P) objectives.

    Returns representations that are non-dominated on the chosen objectives.
    """
    if objectives is None:
        objectives = ["Q", "N", "P"]

    ranked = rank_representations(session, group=group, top_k=10000)
    if ranked.empty:
        return ranked

    # Extract Pareto-optimal points
    points = ranked[objectives].values
    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if j >= i in all objectives and j > i in at least one
            if all(points[j, k] >= points[i, k] for k in range(len(objectives))) and \
               any(points[j, k] > points[i, k] for k in range(len(objectives))):
                is_dominated[i] = True
                break

    frontier = ranked[~is_dominated].reset_index(drop=True)
    return frontier
