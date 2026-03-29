"""Novelty detection — isolation forest and local outlier factor.

Provides novelty scores for representations based on their feature vectors,
complementing the histogram-based gap detection.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from cmf_atlas.density.heatmaps import _load_features_df, GROUP_AXES


def _get_numeric_feature_matrix(df: pd.DataFrame, group: str) -> tuple[np.ndarray, list[str]]:
    """Extract numeric feature columns for a group into a matrix."""
    axes = GROUP_AXES[group]
    x_col, y_col = axes["x"], axes["y"]

    # Core numeric features common to all groups
    candidates = [x_col, y_col, "complexity_score", "canonical_length", "coeff_log10_max"]

    if group == "dfinite":
        candidates += ["coeff_l1", "d_finite_rank_proxy", "singularity_proxy_count"]
    elif group == "hypergeometric":
        candidates += ["coeff_norm_P", "coeff_norm_Q", "deg_balance", "ratio_complexity"]
    elif group == "pcf":
        candidates += ["coeff_l1_a", "coeff_l1_b"]

    # Keep only columns that exist and are numeric
    cols = [c for c in candidates if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    sub = sub.fillna(0)

    return sub.values, list(sub.columns)


def compute_novelty_scores(
    session,
    group: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Compute novelty scores for all representations in a group.

    Parameters
    ----------
    session : SQLAlchemy session
    group : "dfinite" | "hypergeometric" | "pcf"
    method : "isolation_forest" or "lof" (Local Outlier Factor)
    contamination : expected fraction of outliers

    Returns
    -------
    DataFrame with columns: representation_id, novelty_score (0..1, higher = more novel)
    """
    df = _load_features_df(session, group)
    if df.empty or len(df) < 5:
        return pd.DataFrame(columns=["representation_id", "novelty_score"])

    X, cols = _get_numeric_feature_matrix(df, group)
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["representation_id", "novelty_score"])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "isolation_forest":
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        # score_samples returns negative anomaly scores; more negative = more anomalous
        raw_scores = model.fit(X_scaled).score_samples(X_scaled)
        # Normalize to 0..1 (higher = more novel)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            novelty = 1.0 - (raw_scores - s_min) / (s_max - s_min)
        else:
            novelty = np.zeros_like(raw_scores)

    elif method == "lof":
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=min(20, len(X) - 1),
            novelty=False,
        )
        labels = model.fit_predict(X_scaled)
        raw_scores = model.negative_outlier_factor_
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            novelty = 1.0 - (raw_scores - s_min) / (s_max - s_min)
        else:
            novelty = np.zeros_like(raw_scores)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    result = pd.DataFrame({
        "representation_id": df["representation_id"].values,
        "novelty_score": novelty,
    })

    return result.sort_values("novelty_score", ascending=False).reset_index(drop=True)
