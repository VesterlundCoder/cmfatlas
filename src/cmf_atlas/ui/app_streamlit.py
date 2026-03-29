"""CMF Atlas — Interactive Streamlit Dashboard.

Three-panel density atlas with gap detection, ranking, and detail views.

Run with:  streamlit run src/cmf_atlas/ui/app_streamlit.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cmf_atlas.db.session import get_engine, get_session, init_db
from cmf_atlas.db.models import Representation, Features, CMF, EvalRun, RecognitionAttempt
from cmf_atlas.density.heatmaps import (
    GROUP_AXES,
    _load_features_df,
    make_group_heatmap,
    make_triple_heatmap,
)
from cmf_atlas.density.gaps import build_gap_atlas, detect_gaps
from cmf_atlas.density.ranking import rank_representations, pareto_frontier

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMF Atlas",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database connection ──────────────────────────────────────────────────
DB_PATH = _PROJECT_ROOT / "data" / "atlas.db"


@st.cache_resource
def get_db_engine():
    engine = get_engine(DB_PATH)
    init_db(engine)
    return engine


engine = get_db_engine()


def fresh_session():
    return get_session(engine)


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("🗺️ CMF Atlas")
page = st.sidebar.radio(
    "Navigation",
    ["Global Overview", "D-finite", "Hypergeometric", "PCF", "Gap Atlas", "Rankings", "Detail View"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
min_conv = st.sidebar.slider("Min convergence score", 0.0, 1.0, 0.0, 0.05)
unrecognized_only = st.sidebar.checkbox("Unrecognized only", value=False)
max_complexity = st.sidebar.slider("Max complexity score", 0.0, 20.0, 20.0, 0.5)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to a features DataFrame."""
    if df.empty:
        return df
    if "conv_score" in df.columns:
        mask = df["conv_score"].fillna(0) >= min_conv
        df = df[mask]
    if unrecognized_only and "recognized" in df.columns:
        df = df[df["recognized"].fillna(0) == 0]
    if "complexity_score" in df.columns:
        df = df[df["complexity_score"].fillna(0) <= max_complexity]
    return df


# ── Database stats ───────────────────────────────────────────────────────
def get_stats():
    session = fresh_session()
    try:
        total = session.query(Representation).count()
        by_group = {}
        for g in ["dfinite", "hypergeometric", "pcf"]:
            by_group[g] = session.query(Representation).filter_by(primary_group=g).count()
        n_features = session.query(Features).count()
        n_cmfs = session.query(CMF).count()
        n_evals = session.query(EvalRun).count()
        n_recog = session.query(RecognitionAttempt).count()
        return {
            "total": total,
            "by_group": by_group,
            "features": n_features,
            "cmfs": n_cmfs,
            "eval_runs": n_evals,
            "recognition_attempts": n_recog,
        }
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: Global Overview
# ══════════════════════════════════════════════════════════════════════════
if page == "Global Overview":
    st.title("🗺️ CMF Atlas — Global Overview")

    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Representations", stats["total"])
    col2.metric("D-finite", stats["by_group"].get("dfinite", 0))
    col3.metric("Hypergeometric", stats["by_group"].get("hypergeometric", 0))
    col4.metric("PCF", stats["by_group"].get("pcf", 0))

    st.markdown("---")

    overlay = st.selectbox(
        "Overlay",
        ["density", "conv_score", "stability_score", "recognized"],
        index=0,
    )

    session = fresh_session()
    try:
        fig = make_triple_heatmap(session, overlay=overlay if overlay != "density" else None)
        st.plotly_chart(fig, width="stretch")
    finally:
        session.close()

    st.markdown("---")
    st.subheader("Database Summary")
    st.json({
        "representations": stats["total"],
        "features_computed": stats["features"],
        "cmf_objects": stats["cmfs"],
        "evaluation_runs": stats["eval_runs"],
        "recognition_attempts": stats["recognition_attempts"],
    })


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: Single Group Heatmap
# ══════════════════════════════════════════════════════════════════════════
elif page in ("D-finite", "Hypergeometric", "PCF"):
    group_map = {"D-finite": "dfinite", "Hypergeometric": "hypergeometric", "PCF": "pcf"}
    group = group_map[page]

    st.title(f"📊 {page} Density Map")

    session = fresh_session()
    try:
        df = _load_features_df(session, group)
        df = apply_filters(df)

        st.metric(f"Representations in {page}", len(df))

        overlay = st.selectbox(
            "Overlay",
            ["density", "conv_score", "stability_score", "complexity_score"],
            index=0,
        )

        if not df.empty:
            fig = make_group_heatmap(
                df, group,
                overlay=overlay if overlay != "density" else None,
                title=f"{page} — {overlay}",
            )
            st.plotly_chart(fig, width="stretch")

            # Cell click → show items
            st.markdown("---")
            st.subheader("Browse by Cell")
            axes = GROUP_AXES[group]

            if axes["x"] in df.columns and axes["y"] in df.columns:
                x_vals = sorted(df[axes["x"]].dropna().unique())
                y_vals = sorted(df[axes["y"]].dropna().unique())

                col1, col2 = st.columns(2)
                sel_x = col1.selectbox(f"{axes['x_label']}", x_vals if x_vals else [0])
                sel_y = col2.selectbox(f"{axes['y_label']}", y_vals if y_vals else [0])

                cell_df = df[(df[axes["x"]] == sel_x) & (df[axes["y"]] == sel_y)]
                st.write(f"**{len(cell_df)} items** in cell ({sel_x}, {sel_y})")

                if not cell_df.empty:
                    display_cols = [c for c in cell_df.columns if c not in ("representation_id", "_version")]
                    st.dataframe(cell_df[display_cols[:12]].head(50), width="stretch")
        else:
            st.info("No data available for this group. Import data first.")
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: Gap Atlas
# ══════════════════════════════════════════════════════════════════════════
elif page == "Gap Atlas":
    st.title("🕳️ Gap Atlas — Low-Density Regions")

    max_density = st.slider("Max density threshold", 0, 20, 2)

    session = fresh_session()
    try:
        gap_df = build_gap_atlas(session, max_density=max_density)

        if gap_df.empty:
            st.info("No gaps found at this threshold. Try increasing the max density.")
        else:
            st.metric("Total Gaps Found", len(gap_df))

            # Show per group
            for group in ["dfinite", "hypergeometric", "pcf"]:
                group_gaps = gap_df[gap_df["group"] == group]
                if not group_gaps.empty:
                    st.subheader(f"{group.upper()} — {len(group_gaps)} gaps")
                    st.dataframe(
                        group_gaps[["x_coord", "y_coord", "density", "x_label", "y_label"]],
                        width="stretch",
                    )

            st.markdown("---")
            st.subheader("Generator Presets")
            st.markdown("Select a gap to see the suggested generator preset:")

            if len(gap_df) > 0:
                gap_idx = st.selectbox(
                    "Select gap",
                    range(len(gap_df)),
                    format_func=lambda i: (
                        f"{gap_df.iloc[i]['group']} ({gap_df.iloc[i]['x_coord']}, "
                        f"{gap_df.iloc[i]['y_coord']}) — density={gap_df.iloc[i]['density']}"
                    ),
                )
                st.json(gap_df.iloc[gap_idx]["suggested_preset"])
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: Rankings
# ══════════════════════════════════════════════════════════════════════════
elif page == "Rankings":
    st.title("🏆 Rankings — Multi-Objective Scoring")

    col1, col2 = st.columns(2)
    alpha = col1.slider("α (quality)", 0.0, 3.0, 1.2, 0.1)
    beta = col1.slider("β (novelty)", 0.0, 3.0, 1.0, 0.1)
    gamma = col2.slider("γ (proof signal)", 0.0, 3.0, 1.0, 0.1)
    delta = col2.slider("δ (unidentified reward)", 0.0, 3.0, 0.8, 0.1)

    top_k = st.slider("Top K", 10, 200, 50)

    group_filter = st.selectbox("Group", ["All", "dfinite", "hypergeometric", "pcf"])
    group = None if group_filter == "All" else group_filter

    session = fresh_session()
    try:
        ranked = rank_representations(
            session, group=group, top_k=top_k,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        )

        if ranked.empty:
            st.info("No ranked items. Import data and compute features first.")
        else:
            st.subheader(f"Top {len(ranked)} by S = Q^{alpha} · N^{beta} · P^{gamma} · R^{delta}")
            st.dataframe(ranked, width="stretch")

            st.markdown("---")
            st.subheader("Pareto Frontier (Q, N, P)")
            frontier = pareto_frontier(session, group=group)
            if not frontier.empty:
                st.write(f"**{len(frontier)} Pareto-optimal points**")
                st.dataframe(frontier, width="stretch")

                # 3D scatter of Pareto frontier
                fig = go.Figure(data=[go.Scatter3d(
                    x=frontier["Q"],
                    y=frontier["N"],
                    z=frontier["P"],
                    mode="markers",
                    marker=dict(size=6, color=frontier["score"], colorscale="Viridis", showscale=True),
                    text=frontier["representation_id"].astype(str),
                )])
                fig.update_layout(
                    title="Pareto Frontier",
                    scene=dict(xaxis_title="Quality", yaxis_title="Novelty", zaxis_title="Proof Signal"),
                    width=700, height=500,
                )
                st.plotly_chart(fig, width="stretch")
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: Detail View
# ══════════════════════════════════════════════════════════════════════════
elif page == "Detail View":
    st.title("🔍 Representation Detail")

    rep_id = st.number_input("Representation ID", min_value=1, step=1)

    session = fresh_session()
    try:
        rep = session.query(Representation).filter_by(id=rep_id).first()

        if rep is None:
            st.warning(f"No representation with ID {rep_id}")
        else:
            st.subheader(f"Representation #{rep.id}")
            col1, col2 = st.columns(2)
            col1.markdown(f"**Group:** `{rep.primary_group}`")
            col2.markdown(f"**Fingerprint:** `{rep.canonical_fingerprint}`")

            st.markdown("**Canonical Payload:**")
            try:
                st.json(json.loads(rep.canonical_payload))
            except Exception:
                st.code(rep.canonical_payload)

            # Features
            if rep.features:
                st.markdown("---")
                st.subheader("Features")
                try:
                    st.json(json.loads(rep.features.feature_json))
                except Exception:
                    st.code(rep.features.feature_json)

            # CMFs
            if rep.cmfs:
                st.markdown("---")
                st.subheader(f"CMF Objects ({len(rep.cmfs)})")
                for cmf in rep.cmfs:
                    with st.expander(f"CMF #{cmf.id} (dim={cmf.dimension})"):
                        try:
                            st.json(json.loads(cmf.cmf_payload))
                        except Exception:
                            st.code(cmf.cmf_payload)

                        # Eval runs
                        for run in cmf.eval_runs:
                            st.markdown(f"**Eval Run #{run.id}** ({run.run_type}, {run.precision_digits} digits)")
                            st.write(f"Limit: `{run.limit_estimate}`")
                            st.write(f"Conv: {run.convergence_score}, Stability: {run.stability_score}")

                            # Recognition attempts
                            for att in run.recognition_attempts:
                                status = "✅ Identified" if att.success else "❌ Not identified"
                                st.markdown(f"  Recognition: {status}")
                                if att.identified_as:
                                    st.write(f"  → `{att.identified_as}` (residual: 1e{att.residual_log10:.1f})")
                                if att.attempt_log:
                                    with st.expander("Attempt log"):
                                        st.code(att.attempt_log)

            # Series info
            if rep.series:
                st.markdown("---")
                st.subheader("Series")
                st.write(f"**Name:** {rep.series.name}")
                st.write(f"**Definition:** {rep.series.definition}")
                st.write(f"**Generator:** {rep.series.generator_type}")
                st.write(f"**Provenance:** {rep.series.provenance}")
    finally:
        session.close()
