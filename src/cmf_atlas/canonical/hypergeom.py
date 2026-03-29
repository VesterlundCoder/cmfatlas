"""Canonicalization for hypergeometric / almost-hypergeometric representations.

A hypergeometric series has term ratio t_{n+1}/t_n = P(n)/Q(n)
where P, Q are coprime polynomials.

Canonical form:
    - P, Q coprime (GCD removed)
    - Coefficients are primitive integers
    - Leading coefficient of Q is positive
    - Stored as factored form when possible (linear factors over ℤ)
"""

from math import gcd
from functools import reduce

from cmf_atlas.util.hashing import stable_hash


def _make_primitive(coeffs: list[int]) -> list[int]:
    """Divide by GCD, ensure leading coeff positive."""
    nonzero = [abs(c) for c in coeffs if c != 0]
    if not nonzero:
        return [0]
    g = reduce(gcd, nonzero)
    result = [c // g for c in coeffs]
    # Fix sign: leading nonzero coeff positive
    for c in reversed(result):
        if c != 0:
            if c < 0:
                result = [-x for x in result]
            break
    return result


def _strip_trailing_zeros(coeffs: list[int]) -> list[int]:
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    return coeffs


def canonicalize_hypergeom(
    numerator_coeffs: list[int],
    denominator_coeffs: list[int],
) -> dict:
    """Canonicalize a hypergeometric term ratio P(n)/Q(n).

    Parameters
    ----------
    numerator_coeffs : list of int
        Coefficients of P(n) = c_0 + c_1*n + c_2*n^2 + ...
    denominator_coeffs : list of int
        Coefficients of Q(n) = c_0 + c_1*n + c_2*n^2 + ...

    Returns
    -------
    dict with degP, degQ, num_coeffs, den_coeffs, fingerprint
    """
    num = _strip_trailing_zeros(list(numerator_coeffs))
    den = _strip_trailing_zeros(list(denominator_coeffs))

    # Remove common polynomial GCD using sympy if available
    try:
        import sympy as sp
        n = sp.Symbol('n')
        P = sum(c * n**i for i, c in enumerate(num))
        Q = sum(c * n**i for i, c in enumerate(den))
        g = sp.gcd(P, Q)
        if g != 1 and g != 0:
            P = sp.cancel(P / g)
            Q = sp.cancel(Q / g)
            p_poly = sp.Poly(P, n)
            q_poly = sp.Poly(Q, n)
            num = [int(c) for c in p_poly.all_coeffs()[::-1]]
            den = [int(c) for c in q_poly.all_coeffs()[::-1]]
    except Exception:
        pass

    num = _make_primitive(num)
    den = _make_primitive(den)

    degP = len(num) - 1
    degQ = len(den) - 1

    fp = stable_hash({"hypergeom": {"num": num, "den": den}})

    return {
        "degP": degP,
        "degQ": degQ,
        "deg_balance": degP - degQ,
        "num_coeffs": num,
        "den_coeffs": den,
        "fingerprint": fp,
    }
