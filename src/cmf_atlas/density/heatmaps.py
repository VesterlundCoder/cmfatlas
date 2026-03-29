"""Heatmap generation for the three-group density atlas.

Produces Plotly heatmaps for:
    - D-finite: (rec_order, max_poly_degree)
    - Hypergeometric: (degP, degQ)
    - PCF: (deg_a, deg_b)

Each heatmap supports overlays for conv_score, recognized fraction, etc.
"""

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Axis definitions per group ───────────────────────────────────────────
GROUP_AXES = {
    "dfinite": {
        "x": "rec_order",
        "y": "max_poly_degree",
        "x_label": "Recurrence Order",
        "y_label": "Max Polynomial Degree",
    },
    "hypergeometric": {
        "x": "degP",
        "y": "degQ",
        "x_label": "deg(Numerator)",
        "y_label": "deg(Denominator)",
    },
    "pcf": {
        "x": "deg_a",
        "y": "deg_b",
        "x_label": "deg(a)",
        "y_label": "deg(b)",
    },
}


def _load_features_df(session, group: str) -> pd.DataFrame:
    """Load features for a group into a DataFrame."""
    from cmf_atlas.db.models import Features, Representation

    rows = (
        session.query(Representation.id, Features.feature_json)
        .join(Features, Features.representation_id == Representation.id)
        .filter(Representation.primary_group == group)
        .all()
    )

    records = []
    for rep_id, feat_json in rows:
        try:
            feat = json.loads(feat_json)
            feat["representation_id"] = rep_id
            records.append(feat)
        except Exception:
            continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def build_density_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D histogram grid from feature data.

    Returns (x_edges, y_edges, counts) where counts[i,j] is the number
    of items with x in [x_edges[i], x_edges[i+1]) and y in [y_edges[j], y_edges[j+1]).
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return np.array([0, 1]), np.array([0, 1]), np.zeros((1, 1))

    x_vals = df[x_col].dropna().astype(int)
    y_vals = df[y_col].dropna().astype(int)

    if x_range is None:
        x_range = (int(x_vals.min()), int(x_vals.max()) + 1)
    if y_range is None:
        y_range = (int(y_vals.min()), int(y_vals.max()) + 1)

    x_bins = np.arange(x_range[0], x_range[1] + 1)
    y_bins = np.arange(y_range[0], y_range[1] + 1)

    counts, _, _ = np.histogram2d(
        x_vals.values, y_vals.values,
        bins=[x_bins, y_bins],
    )

    return x_bins, y_bins, counts


def build_overlay_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    agg: str = "mean",
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D grid of aggregated values (e.g., mean conv_score per cell)."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return np.array([0, 1]), np.array([0, 1]), np.full((1, 1), np.nan)

    valid = df[[x_col, y_col, value_col]].dropna()
    if valid.empty:
        return np.array([0, 1]), np.array([0, 1]), np.full((1, 1), np.nan)

    x_vals = valid[x_col].astype(int)
    y_vals = valid[y_col].astype(int)

    if x_range is None:
        x_range = (int(x_vals.min()), int(x_vals.max()) + 1)
    if y_range is None:
        y_range = (int(y_vals.min()), int(y_vals.max()) + 1)

    grid = defaultdict(list)
    for _, row in valid.iterrows():
        xi = int(row[x_col])
        yi = int(row[y_col])
        grid[(xi, yi)].append(float(row[value_col]))

    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    result = np.full((nx, ny), np.nan)

    for (xi, yi), vals in grid.items():
        ix = xi - x_range[0]
        iy = yi - y_range[0]
        if 0 <= ix < nx and 0 <= iy < ny:
            if agg == "mean":
                result[ix, iy] = np.mean(vals)
            elif agg == "fraction":
                result[ix, iy] = np.mean([1 if v else 0 for v in vals])
            elif agg == "count":
                result[ix, iy] = len(vals)

    x_bins = np.arange(x_range[0], x_range[1] + 1)
    y_bins = np.arange(y_range[0], y_range[1] + 1)
    return x_bins, y_bins, result


def make_group_heatmap(
    df: pd.DataFrame,
    group: str,
    overlay: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create a Plotly heatmap for one group.

    Parameters
    ----------
    df : DataFrame with feature columns
    group : "dfinite" | "hypergeometric" | "pcf"
    overlay : None (density) | "conv_score" | "recognized" | "stability_score"
    title : optional title override
    """
    axes = GROUP_AXES[group]
    x_col, y_col = axes["x"], axes["y"]

    if overlay is None or overlay == "density":
        x_bins, y_bins, grid = build_density_grid(df, x_col, y_col)
        colorbar_title = "Count"
        colorscale = "YlOrRd"
        z = grid.T  # Plotly expects (y, x) ordering
    elif overlay == "recognized":
        x_bins, y_bins, grid = build_overlay_grid(
            df, x_col, y_col, "recognized", agg="fraction"
        )
        colorbar_title = "Frac. Recognized"
        colorscale = "RdYlGn"
        z = grid.T
    else:
        x_bins, y_bins, grid = build_overlay_grid(
            df, x_col, y_col, overlay, agg="mean"
        )
        colorbar_title = overlay
        colorscale = "Viridis"
        z = grid.T

    if title is None:
        title = f"{group.upper()} — {overlay or 'density'}"

    # Cell centers for axis labels
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2 if len(x_bins) > 1 else x_bins
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2 if len(y_bins) > 1 else y_bins

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_centers,
        y=y_centers,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        hoverongaps=False,
        hovertemplate=(
            f"{axes['x_label']}: %{{x}}<br>"
            f"{axes['y_label']}: %{{y}}<br>"
            f"{colorbar_title}: %{{z}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=axes["x_label"],
        yaxis_title=axes["y_label"],
        width=600,
        height=500,
    )

    return fig


def make_triple_heatmap(
    session,
    overlay: str | None = None,
) -> go.Figure:
    """Create the three-panel density atlas (one heatmap per group)."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["D-finite", "Hypergeometric", "PCF"],
        horizontal_spacing=0.08,
    )

    for i, group in enumerate(["dfinite", "hypergeometric", "pcf"], 1):
        df = _load_features_df(session, group)
        if df.empty:
            continue

        axes = GROUP_AXES[group]
        x_col, y_col = axes["x"], axes["y"]

        if overlay is None or overlay == "density":
            x_bins, y_bins, grid = build_density_grid(df, x_col, y_col)
            colorscale = "YlOrRd"
        else:
            x_bins, y_bins, grid = build_overlay_grid(
                df, x_col, y_col, overlay, agg="mean"
            )
            colorscale = "Viridis"

        x_centers = (x_bins[:-1] + x_bins[1:]) / 2 if len(x_bins) > 1 else x_bins
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2 if len(y_bins) > 1 else y_bins

        fig.add_trace(
            go.Heatmap(
                z=grid.T,
                x=x_centers,
                y=y_centers,
                colorscale=colorscale,
                showscale=(i == 3),
                hoverongaps=False,
            ),
            row=1, col=i,
        )

        fig.update_xaxes(title_text=axes["x_label"], row=1, col=i)
        fig.update_yaxes(title_text=axes["y_label"], row=1, col=i)

    fig.update_layout(
        title_text=f"CMF Atlas — Density Map ({overlay or 'count'})",
        height=500,
        width=1600,
    )

    return fig
