"""Feature computation engine.

Computes all features for a given representation based on its primary_group
and canonical payload. Returns a flat dict ready for feature_json storage.
"""

import math
from typing import Any

from cmf_atlas.features.defs import FEATURE_VERSION
from cmf_atlas.util.json import loads


def _safe_log10(x: float) -> float:
    if x <= 0:
        return 0.0
    return math.log10(x)


def _l1_norm(coeffs: list) -> int:
    return sum(abs(int(c)) for c in coeffs)


def _max_abs(coeffs: list) -> int:
    if not coeffs:
        return 0
    return max(abs(int(c)) for c in coeffs)


def _count_integer_roots(poly_coeffs: list[int], window: int = 100) -> int:
    """Count integer roots of polynomial in [-window, window]."""
    if not poly_coeffs:
        return 0
    count = 0
    for n in range(-window, window + 1):
        val = sum(c * n**i for i, c in enumerate(poly_coeffs))
        if val == 0:
            count += 1
    return count


def compute_dfinite_features(payload: dict) -> dict:
    """Compute features for a D-finite representation.

    Handles both bare operator payloads and rich telescope payloads
    (with total_degree, deg_x, deg_y, n_monomials, conjugacy, source_type).
    """
    coeffs = payload.get("coeffs", [])
    order = payload.get("order", 0)
    max_poly_degree = payload.get("max_poly_degree", 0)
    source_type = payload.get("source_type", "")

    # Collect all integer coefficients
    all_c = []
    for p in coeffs:
        all_c.extend(int(c) for c in p)

    coeff_l1 = sum(abs(c) for c in all_c) if all_c else 0
    coeff_max = max(abs(c) for c in all_c) if all_c else 0

    # Leading polynomial
    leading_poly = coeffs[-1] if coeffs else []
    leading_deg = len(leading_poly) - 1 if leading_poly else 0

    # Content GCD
    from math import gcd
    from functools import reduce
    nonzero = [abs(c) for c in all_c if c != 0]
    content = reduce(gcd, nonzero) if nonzero else 1

    # Singularity proxy: integer roots of leading polynomial
    sing_count = _count_integer_roots(leading_poly) if leading_poly else 0

    # Rich telescope fields (from canonicalize_dfinite_rich)
    total_degree = payload.get("total_degree", max_poly_degree)
    deg_x = payload.get("deg_x", 0)
    deg_y = payload.get("deg_y", 0)
    n_monomials = payload.get("n_monomials", len(all_c))
    dimension = payload.get("dimension", 1)
    conjugacy = payload.get("conjugacy", "")

    # For telescope CMFs, use total_degree and n_monomials for better
    # complexity scoring — the operator order isn't meaningful in the
    # same way as for bare recurrences.
    if source_type == "telescope":
        complexity = (
            math.log1p(total_degree) * 2
            + math.log1p(n_monomials)
            + math.log1p(coeff_l1)
        )
        # Use total_degree as rec_order proxy for heatmap axes
        rec_order_eff = total_degree
        max_poly_deg_eff = max(deg_x, deg_y)
    else:
        complexity = math.log1p(order) + math.log1p(max_poly_degree) + math.log1p(len(all_c))
        rec_order_eff = order
        max_poly_deg_eff = max_poly_degree

    return {
        "primary_group": "dfinite",
        "rec_order": rec_order_eff,
        "max_poly_degree": max_poly_deg_eff,
        "coeff_l1": coeff_l1,
        "coeff_log10_max": _safe_log10(coeff_max),
        "leading_poly_degree": leading_deg,
        "content_gcd": content,
        "singularity_proxy_count": sing_count,
        "d_finite_rank_proxy": rec_order_eff * (1 + max_poly_deg_eff),
        "complexity_score": complexity,
        "canonical_length": len(str(coeffs)),
        # Telescope-specific features
        "source_type": source_type,
        "total_degree": total_degree,
        "deg_x": deg_x,
        "deg_y": deg_y,
        "n_monomials": n_monomials,
        "dimension": dimension,
        "conjugacy": conjugacy,
        # Placeholders for eval-dependent features
        "integrality_rate": None,
        "denom_growth_rate": None,
        "conv_score": None,
        "stability_score": None,
        "log10_error": None,
        "recognized": None,
        "best_residual_log10": None,
        "best_relation_height": None,
    }


def compute_hypergeom_features(payload: dict) -> dict:
    """Compute features for a hypergeometric representation."""
    num = payload.get("num_coeffs", [])
    den = payload.get("den_coeffs", [])
    degP = payload.get("degP", len(num) - 1 if num else 0)
    degQ = payload.get("degQ", len(den) - 1 if den else 0)

    norm_P = _l1_norm(num)
    norm_Q = _l1_norm(den)
    max_c = max(_max_abs(num), _max_abs(den))

    # Count linear factors (integer roots of num/den)
    num_factors = _count_integer_roots(num, window=50) if num else 0
    den_factors = _count_integer_roots(den, window=50) if den else 0

    # Ratio limit estimate: leading_coeff(P) / leading_coeff(Q)
    lead_P = num[-1] if num else 0
    lead_Q = den[-1] if den else 1
    ratio_limit = float(lead_P) / float(lead_Q) if lead_Q != 0 else float("inf")

    # Expected radius proxy
    if degP != degQ:
        radius_proxy = 0.0 if degP > degQ else float("inf")
    else:
        radius_proxy = abs(ratio_limit) if ratio_limit != float("inf") else 0.0

    return {
        "primary_group": "hypergeometric",
        "degP": degP,
        "degQ": degQ,
        "deg_balance": degP - degQ,
        "num_factors": num_factors,
        "den_factors": den_factors,
        "coeff_norm_P": norm_P,
        "coeff_norm_Q": norm_Q,
        "coeff_log10_max": _safe_log10(max_c),
        "ratio_complexity": math.log1p(degP + degQ) + math.log1p(max_c),
        "ratio_limit": ratio_limit,
        "expected_radius_proxy": radius_proxy,
        "complexity_score": math.log1p(degP + degQ) + math.log1p(max_c),
        "canonical_length": len(str(num)) + len(str(den)),
        # Placeholders
        "conv_score": None,
        "stability_score": None,
        "log10_error": None,
        "recognized": None,
        "best_residual_log10": None,
        "best_relation_height": None,
    }


def compute_pcf_features(payload: dict) -> dict:
    """Compute features for a PCF representation."""
    a = payload.get("a_coeffs", [])
    b = payload.get("b_coeffs", [])
    deg_a = payload.get("deg_a", len(a) - 1 if a else 0)
    deg_b = payload.get("deg_b", len(b) - 1 if b else 0)

    l1_a = _l1_norm(a)
    l1_b = _l1_norm(b)
    max_c = max(_max_abs(a), _max_abs(b))

    lead_a = a[-1] if a else 0
    lead_b = b[-1] if b else 0
    signature = (deg_a, deg_b, int(lead_a), int(lead_b))

    return {
        "primary_group": "pcf",
        "deg_a": deg_a,
        "deg_b": deg_b,
        "coeff_l1_a": l1_a,
        "coeff_l1_b": l1_b,
        "coeff_log10_max": _safe_log10(max_c),
        "pcf_signature": list(signature),
        "complexity_score": math.log1p(deg_a + deg_b) + math.log1p(max_c),
        "canonical_length": len(str(a)) + len(str(b)),
        # Placeholders for eval-dependent features
        "convergence_regime": None,
        "oscillation_flag": None,
        "conv_score": None,
        "stability_score": None,
        "log10_error": None,
        "recognized": None,
        "best_residual_log10": None,
        "best_relation_height": None,
    }


def compute_features(primary_group: str, canonical_payload: str | dict) -> dict:
    """Compute features for a representation.

    Parameters
    ----------
    primary_group : str
    canonical_payload : str (JSON) or dict

    Returns
    -------
    dict of feature_name -> value
    """
    if isinstance(canonical_payload, str):
        payload = loads(canonical_payload)
    else:
        payload = canonical_payload

    if primary_group == "dfinite":
        features = compute_dfinite_features(payload)
    elif primary_group == "hypergeometric":
        features = compute_hypergeom_features(payload)
    elif primary_group == "pcf":
        features = compute_pcf_features(payload)
    else:
        features = {
            "primary_group": primary_group,
            "complexity_score": 0.0,
            "canonical_length": len(str(payload)),
        }

    features["_version"] = FEATURE_VERSION
    return features


def compute_features_batch(session) -> int:
    """Compute features for all representations that don't have them yet.

    Returns count of features computed.
    """
    from cmf_atlas.db.models import Features, Representation

    reps = (
        session.query(Representation)
        .outerjoin(Features)
        .filter(Features.representation_id.is_(None))
        .all()
    )

    count = 0
    for rep in reps:
        try:
            feat_dict = compute_features(rep.primary_group, rep.canonical_payload)
            from cmf_atlas.util.json import dumps
            feat = Features(
                representation_id=rep.id,
                feature_json=dumps(feat_dict),
                feature_version=FEATURE_VERSION,
            )
            session.add(feat)
            count += 1
        except Exception as e:
            continue

    session.commit()
    return count
