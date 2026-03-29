"""Canonicalization for D-finite / holonomic representations.

A D-finite series satisfies a linear recurrence with polynomial coefficients:
    p_r(n) a_{n+r} + ... + p_1(n) a_{n+1} + p_0(n) a_n = 0

Canonical form:
    - Coefficients are primitive integers (GCD of all integer coeffs = 1)
    - Leading polynomial p_r has positive leading coefficient
    - Operator stored as list of polynomial coefficient lists, ordered by shift index
"""

from math import gcd
from functools import reduce

from cmf_atlas.util.hashing import stable_hash


def _int_gcd_list(vals: list[int]) -> int:
    """GCD of a list of integers."""
    nonzero = [abs(v) for v in vals if v != 0]
    if not nonzero:
        return 1
    return reduce(gcd, nonzero)


def canonicalize_dfinite(operator: list[list[int]]) -> dict:
    """Canonicalize a D-finite recurrence operator.

    Parameters
    ----------
    operator : list of list of int
        operator[k] = polynomial coefficients of p_k(n) as [c_0, c_1, ..., c_d]
        where p_k(n) = c_0 + c_1*n + c_2*n^2 + ...
        k ranges from 0 (shift 0) to r (shift r).

    Returns
    -------
    dict with keys: order, max_poly_degree, coeffs (normalized), fingerprint
    """
    if not operator or all(all(c == 0 for c in p) for p in operator):
        return {
            "order": 0,
            "max_poly_degree": 0,
            "coeffs": [],
            "fingerprint": stable_hash({"dfinite": []}),
        }

    # Strip trailing zero operators
    while operator and all(c == 0 for c in operator[-1]):
        operator = operator[:-1]
    # Strip leading zero operators
    while operator and all(c == 0 for c in operator[0]):
        operator = operator[1:]

    order = len(operator) - 1

    # Collect all integer coefficients for GCD
    all_coeffs = []
    for p in operator:
        all_coeffs.extend(int(c) for c in p)

    g = _int_gcd_list(all_coeffs)

    # Normalize: divide by GCD, fix sign so leading poly has positive leading coeff
    leading_poly = operator[-1]
    leading_coeff = 0
    for c in reversed(leading_poly):
        if c != 0:
            leading_coeff = c
            break
    sign = 1 if leading_coeff > 0 else -1

    normalized = []
    for p in operator:
        normalized.append([sign * int(c) // g for c in p])

    # Strip trailing zeros from each polynomial
    clean = []
    for p in normalized:
        while len(p) > 1 and p[-1] == 0:
            p = p[:-1]
        clean.append(p)

    max_poly_degree = max(len(p) - 1 for p in clean)

    fp = stable_hash({"dfinite": clean})

    return {
        "order": order,
        "max_poly_degree": max_poly_degree,
        "coeffs": clean,
        "fingerprint": fp,
    }


def _degrees_from_fpoly(f_poly_str: str) -> tuple[int, int, int, int]:
    """Extract (total_degree, deg_x, deg_y, n_monomials) from a polynomial string."""
    if not f_poly_str:
        return 0, 0, 0, 0
    try:
        import sympy as sp
        x, y = sp.symbols("x y")
        expr = sp.sympify(f_poly_str.replace("^", "**"))
        free = expr.free_symbols
        if y in free:
            poly = sp.Poly(expr, x, y)
        elif x in free:
            poly = sp.Poly(expr, x)
        else:
            return 0, 0, 0, 1
        monoms = poly.as_dict()
        total_deg = poly.total_degree()
        deg_x = max((e[0] for e in monoms.keys()), default=0)
        deg_y = max((e[1] if len(e) > 1 else 0 for e in monoms.keys()), default=0)
        return total_deg, deg_x, deg_y, len(monoms)
    except Exception:
        return 0, 0, 0, 0


def canonicalize_dfinite_rich(payload: dict) -> dict:
    """Canonicalize a D-finite payload that may contain extra metadata.

    Handles telescope (conjugate_poly), ore_algebra, and bare operator payloads.
    The fingerprint is computed from the operator; extra fields are preserved
    in the canonical result for feature extraction.
    """
    operator = payload.get("operator", [])
    source_type = payload.get("source_type", "")
    f_poly = payload.get("f_poly", "")
    fbar_poly = payload.get("fbar_poly", "")

    # Compute degrees from f_poly if not already provided
    deg_x = payload.get("deg_x", 0)
    deg_y = payload.get("deg_y", 0)
    total_degree = payload.get("total_degree", 0)
    n_monomials = payload.get("n_monomials", 0)

    if f_poly and (deg_x == 0 and deg_y == 0 and total_degree == 0):
        total_degree, deg_x, deg_y, n_monomials = _degrees_from_fpoly(f_poly)
    elif f_poly and deg_x == 0 and deg_y == 0:
        _, deg_x, deg_y, n_monomials = _degrees_from_fpoly(f_poly)

    # For telescope CMFs with empty operator but f_poly, hash the f_poly+fbar
    if not operator or all(all(c == 0 for c in p) for p in operator if p):
        k1_str = payload.get("K1_str", "")
        rec_str = payload.get("recurrence_str", "")

        # Build a fingerprint from whatever identifying data we have
        fp_data = {
            "source_type": source_type,
            "f_poly": f_poly,
            "fbar_poly": fbar_poly,
            "K1_str": k1_str[:200] if k1_str else "",
            "recurrence": rec_str[:200] if rec_str else "",
        }
        fp = stable_hash({"dfinite_rich": fp_data})

        return {
            "order": 0,
            "max_poly_degree": total_degree,
            "coeffs": [],
            "fingerprint": fp,
            "f_poly": f_poly,
            "fbar_poly": fbar_poly,
            "source_type": source_type,
            "total_degree": total_degree,
            "deg_x": deg_x,
            "deg_y": deg_y,
            "n_monomials": n_monomials,
            "dimension": payload.get("dimension", 1),
            "conjugacy": payload.get("conjugacy", ""),
        }

    # Canonicalize the operator normally
    base = canonicalize_dfinite(operator)

    # Enrich with metadata
    base["f_poly"] = f_poly
    base["fbar_poly"] = fbar_poly
    base["source_type"] = source_type
    base["total_degree"] = total_degree
    base["deg_x"] = deg_x
    base["deg_y"] = deg_y
    base["n_monomials"] = n_monomials
    base["dimension"] = payload.get("dimension", 1)
    base["conjugacy"] = payload.get("conjugacy", "")

    return base
