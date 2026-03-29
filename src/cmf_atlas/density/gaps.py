"""Gap detection — identify low-density regions in the feature space.

A "gap" is a cell in the heatmap grid with density below a threshold,
constrained to regions that are not mathematically impossible.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from cmf_atlas.density.heatmaps import (
    GROUP_AXES,
    _load_features_df,
    build_density_grid,
)


def detect_gaps(
    session,
    group: str,
    max_density: int = 2,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
    exclude_unstable: bool = True,
) -> list[dict]:
    """Find low-density cells for a given group.

    Parameters
    ----------
    session : SQLAlchemy session
    group : "dfinite" | "hypergeometric" | "pcf"
    max_density : cells with count <= this are considered gaps
    x_range, y_range : optional bounds for the grid
    exclude_unstable : if True, skip cells known to be degenerate

    Returns
    -------
    List of gap dicts sorted by density (ascending), each with:
        group, x_coord, y_coord, density, x_label, y_label,
        suggested_generator_preset
    """
    axes = GROUP_AXES[group]
    x_col, y_col = axes["x"], axes["y"]

    df = _load_features_df(session, group)

    if x_range is None:
        if group == "dfinite":
            x_range = (1, 8)
        elif group == "hypergeometric":
            x_range = (1, 8)
        elif group == "pcf":
            x_range = (0, 6)
        else:
            x_range = (0, 10)

    if y_range is None:
        if group == "dfinite":
            y_range = (1, 8)
        elif group == "hypergeometric":
            y_range = (1, 8)
        elif group == "pcf":
            y_range = (0, 6)
        else:
            y_range = (0, 10)

    x_bins, y_bins, counts = build_density_grid(
        df, x_col, y_col, x_range=x_range, y_range=y_range
    )

    gaps = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            density = int(counts[i, j])
            if density <= max_density:
                x_val = int(x_bins[i])
                y_val = int(y_bins[j])

                # Skip degenerate/trivial cells
                if exclude_unstable:
                    if group == "pcf" and x_val == 0 and y_val == 0:
                        continue  # constant CF, trivial
                    if group == "dfinite" and x_val == 0:
                        continue  # order 0, not a recurrence
                    # For telescope CMFs (polynomial f(x,y)):
                    #   rec_order = total_degree(f)
                    #   max_poly_degree = max(deg_x, deg_y)
                    # Constraints:
                    #   max_poly_degree <= total_degree (obvious)
                    #   total_degree <= deg_x + deg_y <= 2 * max_poly_degree
                    # So: ceil(rec_order/2) <= max_poly_degree <= rec_order
                    if group == "dfinite" and y_val > x_val:
                        continue
                    if group == "dfinite" and x_val > 2 * y_val:
                        continue  # total_degree > 2*max_var_degree impossible

                preset = _suggest_generator_preset(group, x_val, y_val)

                gaps.append({
                    "group": group,
                    "x_coord": x_val,
                    "y_coord": y_val,
                    "x_label": axes["x_label"],
                    "y_label": axes["y_label"],
                    "density": density,
                    "suggested_preset": preset,
                })

    gaps.sort(key=lambda g: g["density"])
    return gaps


def _suggest_generator_preset(group: str, x: int, y: int) -> dict:
    """Suggest generator parameters to fill a gap cell."""
    if group == "dfinite":
        return {
            "generator": "dfinite_gen",
            "rec_order": x,
            "max_poly_degree": y,
            "n_candidates": 100,
        }
    elif group == "hypergeometric":
        return {
            "generator": "hypergeom_gen",
            "degP": x,
            "degQ": y,
            "n_candidates": 100,
        }
    elif group == "pcf":
        return {
            "generator": "pcf_gen",
            "deg_a": x,
            "deg_b": y,
            "n_candidates": 100,
        }
    return {}


def build_gap_atlas(session, max_density: int = 2) -> pd.DataFrame:
    """Build the full gap atlas across all three groups.

    Returns a DataFrame with all gaps, sorted by density then group.
    """
    all_gaps = []
    for group in ["dfinite", "hypergeometric", "pcf"]:
        gaps = detect_gaps(session, group, max_density=max_density)
        all_gaps.extend(gaps)

    if not all_gaps:
        return pd.DataFrame()

    df = pd.DataFrame(all_gaps)
    df = df.sort_values(["density", "group", "x_coord", "y_coord"]).reset_index(drop=True)
    return df
