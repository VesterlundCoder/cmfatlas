"""Feature definitions — exact specifications per group.

Feature version is bumped when definitions change.
"""

FEATURE_VERSION = "1.0.0"

# ── Shared / core features (all groups) ──────────────────────────────────
SHARED_FEATURES = {
    "primary_group":      "str — dfinite | hypergeometric | pcf",
    "complexity_score":   "float — log1p(num_params) + log1p(max_degree) + log1p(op_order)",
    "canonical_length":   "int — length of canonical payload string",
    # Numeric outcome (filled after eval)
    "conv_score":         "float 0..1 — convergence score from eval run",
    "stability_score":    "float 0..1 — consistency across precision/direction/seeds",
    "log10_error":        "float — log10(error_estimate)",
    # Recognition (filled after recognition attempts)
    "recognized":         "int 0/1",
    "best_residual_log10":"float — min residual across attempts",
    "best_relation_height":"float or null",
}

# ── D-finite features ────────────────────────────────────────────────────
DFINITE_FEATURES = {
    "rec_order":            "int — order r of recurrence",
    "max_poly_degree":      "int — max degree of coefficient polynomials",
    "coeff_l1":             "int — sum of absolute values of all coefficients",
    "coeff_log10_max":      "float — log10(max |coeff|)",
    "leading_poly_degree":  "int — degree of leading polynomial p_r",
    "content_gcd":          "int — GCD of all integer coefficients (should be 1 after canon)",
    "singularity_proxy_count": "int — number of integer roots of leading poly in [-100,100]",
    "d_finite_rank_proxy":  "int — rec_order * (1 + max_poly_degree)",
    "integrality_rate":     "float — fraction of first M terms that are integers",
    "denom_growth_rate":    "float — slope of log(denom(a_n)) vs n",
}

# ── Hypergeometric features ──────────────────────────────────────────────
HYPERGEOM_FEATURES = {
    "degP":                "int — degree of numerator polynomial",
    "degQ":                "int — degree of denominator polynomial",
    "deg_balance":         "int — degP - degQ",
    "num_factors":         "int — number of linear factors of P over Z",
    "den_factors":         "int — number of linear factors of Q over Z",
    "coeff_norm_P":        "int — L1 norm of P coefficients",
    "coeff_norm_Q":        "int — L1 norm of Q coefficients",
    "ratio_complexity":    "float — log1p(degP+degQ) + log1p(coeff_log10_max)",
    "ratio_limit":         "float — estimated lim t_{n+1}/t_n from leading coeffs",
    "expected_radius_proxy":"float — derived from deg_balance and leading ratio",
}

# ── PCF features ─────────────────────────────────────────────────────────
PCF_FEATURES = {
    "deg_a":               "int — degree of a(n)",
    "deg_b":               "int — degree of b(n)",
    "coeff_l1_a":          "int — L1 norm of a(n) coefficients",
    "coeff_l1_b":          "int — L1 norm of b(n) coefficients",
    "coeff_log10_max":     "float — log10(max |coeff| across a,b)",
    "pcf_signature":       "tuple — (deg_a, deg_b, lead_a, lead_b)",
    "convergence_regime":  "str — fast/medium/slow/unstable",
    "oscillation_flag":    "int 0/1",
}
