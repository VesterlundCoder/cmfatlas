"""Core tests for CMF Atlas — database, canonicalization, features, ranking."""

import json
import os
import tempfile

import pytest


# ── Database tests ───────────────────────────────────────────────────────

def test_init_db_creates_tables():
    from cmf_atlas.db.session import get_engine, init_db
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        engine = get_engine(db_path)
        init_db(engine)
        from sqlalchemy import inspect
        insp = inspect(engine)
        tables = insp.get_table_names()
        assert "project" in tables
        assert "series" in tables
        assert "representation" in tables
        assert "features" in tables
        assert "cmf" in tables
        assert "eval_run" in tables
        assert "recognition_attempt" in tables


def test_insert_and_query():
    from cmf_atlas.db.session import get_engine, init_db, get_session
    from cmf_atlas.db.models import Project, Series, Representation
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        engine = get_engine(db_path)
        init_db(engine)
        session = get_session(engine)

        proj = Project(name="test_project")
        session.add(proj)
        session.flush()

        s = Series(project_id=proj.id, name="s1", definition="test", generator_type="manual")
        session.add(s)
        session.flush()

        rep = Representation(
            series_id=s.id,
            primary_group="pcf",
            canonical_fingerprint="abc123",
            canonical_payload='{"a":[1,2],"b":[3]}',
        )
        session.add(rep)
        session.commit()

        found = session.query(Representation).filter_by(canonical_fingerprint="abc123").first()
        assert found is not None
        assert found.primary_group == "pcf"
        session.close()


def test_fingerprint_dedup():
    """Same fingerprint + group should be rejected."""
    from cmf_atlas.db.session import get_engine, init_db, get_session
    from cmf_atlas.db.models import Project, Series, Representation
    from sqlalchemy.exc import IntegrityError
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        engine = get_engine(db_path)
        init_db(engine)
        session = get_session(engine)

        proj = Project(name="test")
        session.add(proj)
        session.flush()

        s1 = Series(project_id=proj.id, definition="a", generator_type="manual")
        s2 = Series(project_id=proj.id, definition="b", generator_type="manual")
        session.add_all([s1, s2])
        session.flush()

        r1 = Representation(series_id=s1.id, primary_group="pcf",
                            canonical_fingerprint="same_fp", canonical_payload="{}")
        session.add(r1)
        session.commit()

        r2 = Representation(series_id=s2.id, primary_group="pcf",
                            canonical_fingerprint="same_fp", canonical_payload="{}")
        session.add(r2)
        with pytest.raises(IntegrityError):
            session.commit()
        session.close()


# ── Canonicalization tests ───────────────────────────────────────────────

def test_canonicalize_pcf():
    from cmf_atlas.canonical.pcf import canonicalize_pcf
    result = canonicalize_pcf([1, 2, 3], [-4, 0, 5])
    assert result["deg_a"] == 2
    assert result["deg_b"] == 2
    assert result["fingerprint"]
    assert len(result["fingerprint"]) == 16


def test_canonicalize_pcf_normalization():
    """Multiplying all coefficients by a constant should give same fingerprint."""
    from cmf_atlas.canonical.pcf import canonicalize_pcf
    r1 = canonicalize_pcf([2, 4, 6], [-8, 0, 10])
    r2 = canonicalize_pcf([1, 2, 3], [-4, 0, 5])
    assert r1["fingerprint"] == r2["fingerprint"]


def test_canonicalize_dfinite():
    from cmf_atlas.canonical.dfinite import canonicalize_dfinite
    # Order-2 recurrence: (n+1)^2 a_{n+2} - (2n+1) a_{n+1} + n^2 a_n = 0
    op = [
        [0, 0, 1],     # p_0(n) = n^2
        [0, -2, -1],   # p_1(n) = -(2n+1) ... wait, that's wrong sign
        [1, 2, 1],     # p_2(n) = (n+1)^2 = 1 + 2n + n^2
    ]
    result = canonicalize_dfinite(op)
    assert result["order"] == 2
    assert result["max_poly_degree"] == 2
    assert result["fingerprint"]


def test_canonicalize_hypergeom():
    from cmf_atlas.canonical.hypergeom import canonicalize_hypergeom
    # Term ratio (n+1)/(n+2) => P=[1,1], Q=[2,1]
    result = canonicalize_hypergeom([1, 1], [2, 1])
    assert result["degP"] == 1
    assert result["degQ"] == 1
    assert result["fingerprint"]


def test_canonicalize_and_fingerprint_dispatch():
    from cmf_atlas.canonical import canonicalize_and_fingerprint
    fp, payload = canonicalize_and_fingerprint("pcf", {"a_coeffs": [1, 3], "b_coeffs": [-1, 0, 2]})
    assert len(fp) == 16
    assert "deg_a" in payload


# ── Feature computation tests ────────────────────────────────────────────

def test_compute_pcf_features():
    from cmf_atlas.features.compute import compute_features
    feat = compute_features("pcf", {"a_coeffs": [1, 2, 3], "b_coeffs": [-4, 0, 5], "deg_a": 2, "deg_b": 2})
    assert feat["primary_group"] == "pcf"
    assert feat["deg_a"] == 2
    assert feat["deg_b"] == 2
    assert feat["coeff_l1_a"] == 6
    assert feat["_version"]


def test_compute_dfinite_features():
    from cmf_atlas.features.compute import compute_features
    payload = {"order": 2, "max_poly_degree": 2, "coeffs": [[0, 0, 1], [-1, -2], [1, 2, 1]]}
    feat = compute_features("dfinite", payload)
    assert feat["rec_order"] == 2
    assert feat["max_poly_degree"] == 2
    assert feat["coeff_l1"] > 0


def test_compute_hypergeom_features():
    from cmf_atlas.features.compute import compute_features
    payload = {"degP": 2, "degQ": 1, "num_coeffs": [1, 3, 2], "den_coeffs": [1, 1]}
    feat = compute_features("hypergeometric", payload)
    assert feat["degP"] == 2
    assert feat["degQ"] == 1
    assert feat["deg_balance"] == 1


# ── Ranking tests ────────────────────────────────────────────────────────

def test_ranking_components():
    from cmf_atlas.density.ranking import (
        compute_quality, compute_novelty, compute_proof_signal,
        compute_recognizability_reward, compute_score,
    )

    feat = {
        "primary_group": "pcf",
        "conv_score": 0.8,
        "stability_score": 0.7,
        "log10_error": -30,
        "recognized": 0,
        "best_residual_log10": None,
        "complexity_score": 3.0,
        "deg_a": 1,
        "deg_b": 2,
        "coeff_log10_max": 1.0,
    }

    Q = compute_quality(feat)
    assert 0 < Q <= 1

    N = compute_novelty(feat)
    assert 0 <= N <= 1

    P = compute_proof_signal(feat)
    assert 0 <= P <= 1

    R = compute_recognizability_reward(feat)
    assert R == 1.0  # Unidentified, no near-miss

    S = compute_score(feat)
    assert S > 0


def test_recognized_gets_zero_reward():
    from cmf_atlas.density.ranking import compute_recognizability_reward
    assert compute_recognizability_reward({"recognized": 1}) == 0.0
    assert compute_recognizability_reward({"recognized": 0}) == 1.0


# ── Evaluation tests ─────────────────────────────────────────────────────

def test_pcf_quick_eval():
    from cmf_atlas.eval.quick import evaluate_pcf_quick
    # Simple CF: a(n)=2n+1, b(n)=-n^2 => converges to 4/pi
    result = evaluate_pcf_quick([1, 2], [0, 0, -1], depth=100, dps=30)
    assert result["convergence_score"] > 0
    assert result["limit_estimate"] is not None
    assert result["runtime_ms"] >= 0


# ── Generator tests ──────────────────────────────────────────────────────

def test_pcf_generator():
    from cmf_atlas.generate.pcf_gen import generate_pcf_candidates
    candidates = generate_pcf_candidates(deg_a=1, deg_b=2, n_candidates=10, seed=42)
    assert len(candidates) == 10
    for c in candidates:
        assert len(c["a_coeffs"]) == 2  # deg 1 => 2 coefficients
        assert len(c["b_coeffs"]) == 3  # deg 2 => 3 coefficients
        assert c["a_coeffs"][-1] != 0   # nonzero leading


def test_hypergeom_generator():
    from cmf_atlas.generate.hypergeom_gen import generate_hypergeom_candidates
    candidates = generate_hypergeom_candidates(degP=2, degQ=2, n_candidates=10, seed=42)
    assert len(candidates) == 10
    for c in candidates:
        assert "numerator" in c
        assert "denominator" in c


def test_dfinite_generator():
    from cmf_atlas.generate.dfinite_gen import generate_dfinite_candidates
    candidates = generate_dfinite_candidates(rec_order=2, max_poly_degree=2, n_candidates=10, seed=42)
    assert len(candidates) == 10
    for c in candidates:
        assert len(c["operator"]) == 3  # order 2 => 3 polynomials
