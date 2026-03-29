"""Unified canonicalization dispatcher."""

from cmf_atlas.canonical.dfinite import canonicalize_dfinite, canonicalize_dfinite_rich
from cmf_atlas.canonical.hypergeom import canonicalize_hypergeom
from cmf_atlas.canonical.pcf import canonicalize_pcf
from cmf_atlas.util.json import dumps


def canonicalize_and_fingerprint(primary_group: str, payload: dict) -> tuple[str, str]:
    """Canonicalize a representation and return (fingerprint, canonical_payload_json).

    Parameters
    ----------
    primary_group : str
        One of "dfinite", "hypergeometric", "pcf".
    payload : dict
        Raw representation data. Expected keys depend on group:
        - dfinite: {"operator": [...], "source_type": ..., "f_poly": ..., ...}
        - hypergeometric: {"numerator": [int, ...], "denominator": [int, ...]}
        - pcf: {"a_coeffs": [int, ...], "b_coeffs": [int, ...]}

    Returns
    -------
    (fingerprint, canonical_payload_json)
    """
    if primary_group == "dfinite":
        # Use rich canonicalization if payload has extra metadata
        if "source_type" in payload or "f_poly" in payload or "K1_str" in payload:
            result = canonicalize_dfinite_rich(payload)
        else:
            result = canonicalize_dfinite(payload.get("operator", []))
        return result["fingerprint"], dumps(result)

    elif primary_group == "hypergeometric":
        result = canonicalize_hypergeom(
            payload.get("numerator", payload.get("num_coeffs", [])),
            payload.get("denominator", payload.get("den_coeffs", [])),
        )
        return result["fingerprint"], dumps(result)

    elif primary_group == "pcf":
        result = canonicalize_pcf(
            payload.get("a_coeffs", []),
            payload.get("b_coeffs", []),
        )
        return result["fingerprint"], dumps(result)

    else:
        raise ValueError(f"Unknown primary_group: {primary_group!r}")
