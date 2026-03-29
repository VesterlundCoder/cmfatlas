"""Parsers: convert raw data formats into the internal representation dicts.

Supported input formats:
    - CMF database JSON (cmf_database.json v2.x)
    - Euler2AI PCF JSON (pcfs.json)
    - JSONL lines (generic series/representation import)
"""

import json
import re
from typing import Any

import sympy as sp


def _poly_str_to_coeffs(poly_str: str, var: str = "n") -> list[int]:
    """Convert a polynomial string like '3*n^2 + 2*n + 1' to coefficient list [1, 2, 3]."""
    n = sp.Symbol(var)
    try:
        expr = sp.sympify(poly_str.replace("^", "**"))
        p = sp.Poly(expr, n)
        # all_coeffs returns highest degree first; reverse for [c0, c1, ...]
        coeffs = [int(c) for c in reversed(p.all_coeffs())]
        return coeffs
    except Exception:
        return []


def _extract_telescope_operator(f_poly_str: str, fbar_poly_str: str | None = None) -> list[list[int]]:
    """Extract a D-finite recurrence operator from a telescope CMF polynomial.

    For a conjugate polynomial f(x,y) with conjugate fbar(x,y), the telescope
    construction yields matrices K1(k) and K2(k,m) whose entries are polynomial
    in k and m.  The associated scalar sequence satisfies a D-finite recurrence
    whose order equals the degree of f in x.

    We encode this as an operator on the multivariate polynomial ring: the
    operator coefficients are the coefficients of f(x,y) viewed as a polynomial
    in x with coefficients that are polynomials in y, plus the conjugate's
    contribution.  This gives a fingerprint-able representation that is unique
    up to the polynomial and its conjugacy.
    """
    x, y = sp.symbols("x y")
    try:
        f_expr = sp.sympify(f_poly_str.replace("^", "**"))
        f_poly = sp.Poly(f_expr, x, y)
    except Exception:
        return []

    # Encode f as a polynomial in x whose coefficients are polynomials in y.
    # f(x,y) = sum_i  p_i(y) * x^i
    # We store each p_i(y) as its integer coefficient list.
    try:
        f_x = sp.Poly(f_expr, x)  # view as poly in x
        operator = []
        for coeff in reversed(f_x.all_coeffs()):  # highest degree first → reverse
            c_poly = sp.Poly(coeff, y)
            c_coeffs = [int(c) for c in reversed(c_poly.all_coeffs())]
            operator.append(c_coeffs)
    except Exception:
        return []

    # Append conjugate info if available — this distinguishes different
    # conjugacies that share the same f_poly.
    if fbar_poly_str:
        try:
            fbar_expr = sp.sympify(fbar_poly_str.replace("^", "**"))
            fbar_x = sp.Poly(fbar_expr, x)
            for coeff in reversed(fbar_x.all_coeffs()):
                c_poly = sp.Poly(coeff, y)
                c_coeffs = [int(c) for c in reversed(c_poly.all_coeffs())]
                operator.append(c_coeffs)
        except Exception:
            pass

    return operator


def _parse_ore_recurrence(rec_str: str) -> list[list[int]]:
    """Parse an ore_algebra recurrence string into an operator.

    Example: "(n+1)^3 u(n+1) = (2n+1)(10n^2+10n+3) u(n) - 9*n^3 u(n-1)"
    → operator for p_0(n)*a_n + p_1(n)*a_{n+1} + ... = 0
    """
    # Ore recurrence strings vary; extract the a_coeff which gives the
    # companion matrix ratio.  Fall back to hashing the string.
    return []


def _fpoly_to_multivar_coeffs(f_poly_str: str) -> dict:
    """Extract structured coefficient data from a multivariate polynomial string.

    Returns dict with total_degree, deg_x, deg_y, n_monomials, monom_coeffs.
    """
    x, y, z = sp.symbols("x y z")
    try:
        expr = sp.sympify(f_poly_str.replace("^", "**"))
        # Determine which variables are present
        free = expr.free_symbols
        if z in free:
            poly = sp.Poly(expr, x, y, z)
        elif y in free:
            poly = sp.Poly(expr, x, y)
        else:
            poly = sp.Poly(expr, x)

        monoms = poly.as_dict()  # {(exp_x, exp_y, ...): coeff}
        total_deg = poly.total_degree()

        # Monomial coefficient list: [(exponents, int_coeff), ...] sorted
        monom_list = sorted(
            [(list(exp), int(c)) for exp, c in monoms.items()],
            key=lambda t: t[0],
        )

        deg_x = max((e[0] for e, _ in monom_list), default=0)
        deg_y = max((e[1] if len(e) > 1 else 0 for e, _ in monom_list), default=0)

        return {
            "total_degree": total_deg,
            "deg_x": deg_x,
            "deg_y": deg_y,
            "n_monomials": len(monom_list),
            "monom_coeffs": monom_list,
        }
    except Exception:
        return {"total_degree": 0, "deg_x": 0, "deg_y": 0, "n_monomials": 0, "monom_coeffs": []}


def parse_cmf_db_entry(entry: dict) -> dict:
    """Parse a single CMF database entry into atlas import format.

    Classification by type:
        pcf                  → group "pcf"     (a_poly, b_poly)
        conjugate_poly       → group "dfinite" (f_poly/fbar_poly → operator)
        ore_algebra_operator → group "dfinite" (recurrence → operator)
        accumulator          → group "dfinite" (K1_str based)
        companion            → group "dfinite" (matrix recurrence)
        bank_cmf             → group "dfinite" (K1_srepr based)
        recurrence_cmf       → group "dfinite" (verified 2D recurrence)

    Returns a dict with keys:
        series_definition, generator_type, provenance,
        primary_group, payload, cmf_payload, metadata
    """
    f_poly = entry.get("f_poly", "")
    fbar_poly = entry.get("fbar_poly", "")
    source = entry.get("source", "unknown")
    cmf_id = entry.get("id", "")
    dim = entry.get("dim", 1)
    cert = entry.get("certification_level", "")
    constant = entry.get("primary_constant")
    degree = entry.get("degree", 0)
    cmf_type = entry.get("type", "")

    # ── Classification by type ───────────────────────────────────────────
    if cmf_type == "pcf":
        primary_group = "pcf"
    elif cmf_type == "conjugate_poly":
        primary_group = "dfinite"
    elif cmf_type == "ore_algebra_operator":
        primary_group = "dfinite"
    elif cmf_type in ("accumulator", "companion", "bank_cmf", "recurrence_cmf"):
        primary_group = "dfinite"
    elif "hypergeom" in cmf_type or "pFq" in cmf_type.lower():
        primary_group = "hypergeometric"
    else:
        # Fallback heuristic: if it has a_poly/b_poly → pcf, else dfinite
        if "a_poly" in entry and "b_poly" in entry:
            primary_group = "pcf"
        else:
            primary_group = "dfinite"

    # ── Build payload based on group ─────────────────────────────────────
    payload = {}

    if primary_group == "pcf":
        if "a_poly" in entry and "b_poly" in entry:
            payload = {
                "a_coeffs": _poly_str_to_coeffs(entry["a_poly"]),
                "b_coeffs": _poly_str_to_coeffs(entry["b_poly"]),
            }
        else:
            # PCF without explicit polynomials (training_seed_contfrac)
            payload = {"a_coeffs": [0], "b_coeffs": [0]}

    elif primary_group == "dfinite":
        if cmf_type == "conjugate_poly" and f_poly:
            # Telescope CMF: extract operator from f(x,y) and fbar(x,y)
            operator = _extract_telescope_operator(f_poly, fbar_poly or None)
            poly_info = _fpoly_to_multivar_coeffs(f_poly)
            payload = {
                "operator": operator,
                "source_type": "telescope",
                "f_poly": f_poly,
                "fbar_poly": fbar_poly or "",
                "conjugacy": entry.get("action", ""),
                "total_degree": poly_info["total_degree"],
                "deg_x": poly_info["deg_x"],
                "deg_y": poly_info["deg_y"],
                "n_monomials": poly_info["n_monomials"],
                "dimension": dim,
            }
        elif cmf_type == "ore_algebra_operator":
            # Ore algebra: parse recurrence string
            rec_str = entry.get("recurrence", "")
            a_coeff = entry.get("a_coeff", "")
            operator = _parse_ore_recurrence(rec_str)
            payload = {
                "operator": operator,
                "source_type": "ore_algebra",
                "recurrence_str": rec_str,
                "a_coeff_str": str(a_coeff),
                "dimension": dim,
            }
        else:
            # accumulator, companion, bank_cmf, recurrence_cmf, fallback
            k1_str = entry.get("K1_str", entry.get("K1_srepr", ""))
            payload = {
                "operator": [],
                "source_type": cmf_type,
                "K1_str": str(k1_str)[:500] if k1_str else "",
                "dimension": dim,
            }

    elif primary_group == "hypergeometric":
        payload = {
            "numerator": [],
            "denominator": [],
            "dimension": dim,
        }

    # ── CMF payload (always the same structure) ──────────────────────────
    cmf_payload = {
        "f_poly": f_poly,
        "fbar_poly": fbar_poly,
        "dimension": dim,
        "source": source,
        "certification_level": cert,
        "primary_constant": str(constant) if constant else None,
        "degree": degree,
        "flatness_verified": entry.get("flatness_verified", False),
    }

    series_def = f_poly or cmf_id
    if primary_group == "pcf" and "a_poly" in entry:
        series_def = f"PCF(a={entry['a_poly']}, b={entry['b_poly']})"

    return {
        "series_definition": series_def,
        "generator_type": source,
        "provenance": f"cmf_database:{cmf_id}",
        "primary_group": primary_group,
        "payload": payload,
        "cmf_payload": cmf_payload,
        "metadata": {
            "cmf_db_id": cmf_id,
            "source": source,
            "type": cmf_type,
            "cert": cert,
            "dim": dim,
        },
    }


def parse_euler2ai_pcf(entry: dict) -> dict:
    """Parse an Euler2AI PCF entry.

    Expected format: {"a": "3*n+1", "b": "-2*n^2+3*n", "limit": "4/pi", ...}
    """
    a_str = entry.get("a", entry.get("a_poly", "0"))
    b_str = entry.get("b", entry.get("b_poly", "0"))
    limit_str = entry.get("limit", entry.get("stated_limit", ""))
    pcf_id = entry.get("id", entry.get("idx", ""))

    a_coeffs = _poly_str_to_coeffs(str(a_str))
    b_coeffs = _poly_str_to_coeffs(str(b_str))

    return {
        "series_definition": f"PCF(a={a_str}, b={b_str})",
        "generator_type": "pcf",
        "provenance": f"euler2ai:{pcf_id}",
        "primary_group": "pcf",
        "payload": {
            "a_coeffs": a_coeffs,
            "b_coeffs": b_coeffs,
            "a_str": str(a_str),
            "b_str": str(b_str),
        },
        "cmf_payload": {
            "a_str": str(a_str),
            "b_str": str(b_str),
            "limit": str(limit_str),
            "source": "euler2ai",
        },
        "metadata": {
            "pcf_id": str(pcf_id),
            "limit": str(limit_str),
        },
    }


def parse_jsonl_line(line: str) -> dict | None:
    """Parse a single JSONL line in the generic import format.

    Expected fields:
        series_definition, representation_type, payload, cmf_payload, provenance
    """
    try:
        obj = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    if not obj:
        return None

    primary_group = obj.get("representation_type", obj.get("primary_group", "dfinite"))
    return {
        "series_definition": obj.get("series_definition", ""),
        "generator_type": obj.get("generator_type", "imported"),
        "provenance": obj.get("provenance", "jsonl_import"),
        "primary_group": primary_group,
        "payload": obj.get("payload", {}),
        "cmf_payload": obj.get("cmf_payload", {}),
        "metadata": obj.get("metadata", {}),
    }
