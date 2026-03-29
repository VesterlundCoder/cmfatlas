"""Canonicalization for PCF/CF (polynomial continued fraction) representations.

A PCF is defined by polynomial sequences a(n), b(n).
The continued fraction is:  a(0) + b(1)/(a(1) + b(2)/(a(2) + ...))

Canonical form:
    - Polynomial coefficients are primitive integers
    - Leading coefficient of a(n) is positive
    - Content (integer GCD) factored out
    - Minimal degree representation (trailing zeros stripped)
"""

from math import gcd
from functools import reduce

from cmf_atlas.util.hashing import stable_hash


def _make_primitive_positive_lead(coeffs: list[int]) -> list[int]:
    """Normalize polynomial: primitive + positive leading coeff."""
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    nonzero = [abs(c) for c in coeffs if c != 0]
    if not nonzero:
        return [0]
    g = reduce(gcd, nonzero)
    result = [c // g for c in coeffs]
    if result[-1] < 0:
        result = [-x for x in result]
    return result


def canonicalize_pcf(
    a_coeffs: list[int],
    b_coeffs: list[int],
) -> dict:
    """Canonicalize a PCF defined by polynomials a(n), b(n).

    Parameters
    ----------
    a_coeffs : list of int
        Coefficients of a(n) = c_0 + c_1*n + c_2*n^2 + ...
    b_coeffs : list of int
        Coefficients of b(n) = c_0 + c_1*n + c_2*n^2 + ...

    Returns
    -------
    dict with deg_a, deg_b, a_coeffs, b_coeffs, signature, fingerprint
    """
    a = _make_primitive_positive_lead(list(a_coeffs))
    b = list(b_coeffs)
    while len(b) > 1 and b[-1] == 0:
        b = b[:-1]

    # For b, we normalize magnitude but preserve sign pattern
    b_nonzero = [abs(c) for c in b if c != 0]
    if b_nonzero:
        gb = reduce(gcd, b_nonzero)
        b = [c // gb for c in b]

    deg_a = len(a) - 1
    deg_b = len(b) - 1

    # Signature: (deg_a, deg_b, leading_a, leading_b)
    lead_a = a[-1] if a else 0
    lead_b = b[-1] if b else 0
    signature = (deg_a, deg_b, lead_a, lead_b)

    fp = stable_hash({"pcf": {"a": a, "b": b}})

    return {
        "deg_a": deg_a,
        "deg_b": deg_b,
        "a_coeffs": a,
        "b_coeffs": b,
        "signature": signature,
        "fingerprint": fp,
    }
