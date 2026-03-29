"""PSLQ-based constant recognition.

Given a high-precision limit estimate, attempt to identify it as a
linear combination of known constants using PSLQ (integer relation finding).
"""

import time
from typing import Any

import mpmath

from cmf_atlas.recognition.bases import basis_to_mpf_dict, build_standard_basis


def run_pslq(
    limit_str: str,
    basis: dict[str, str] | None = None,
    dps: int = 100,
    max_relation_height: float = 1e12,
) -> dict:
    """Run PSLQ to find integer relations between the limit and basis constants.

    Parameters
    ----------
    limit_str : high-precision string representation of the limit
    basis : dict of {name: value_string}; uses standard basis if None
    dps : working precision
    max_relation_height : reject relations with coefficients larger than this

    Returns
    -------
    dict with: success, identified_as, relation, relation_height,
               residual_log10, attempt_log
    """
    mpmath.mp.dps = dps + 30
    t0 = time.time()

    if basis is None:
        basis = build_standard_basis(dps)

    target = mpmath.mpf(limit_str)
    basis_mpf = basis_to_mpf_dict(basis, dps)
    basis_names = list(basis_mpf.keys())
    basis_vals = list(basis_mpf.values())

    log_lines = [
        f"PSLQ recognition attempt at dps={dps}",
        f"Target: {mpmath.nstr(target, 30)}",
        f"Basis size: {len(basis_names)}",
    ]

    best_result = {
        "success": 0,
        "identified_as": None,
        "relation": None,
        "relation_height": None,
        "residual_log10": None,
        "attempt_log": "",
    }

    # Strategy 1: Direct match against each constant (ratio test)
    for name, val in zip(basis_names, basis_vals):
        if val == 0:
            continue
        ratio = target / val
        # Check if ratio is close to a simple rational p/q
        for q in range(1, 50):
            p_approx = ratio * q
            p_round = mpmath.nint(p_approx)
            if p_round == 0:
                continue
            residual = abs(p_approx - p_round)
            if residual < mpmath.mpf(10) ** (-(dps * 0.6)):
                p = int(p_round)
                identified = f"{p}/{q}*{name}" if q > 1 else f"{p}*{name}"
                if p == 1 and q == 1:
                    identified = name
                elif p == -1 and q == 1:
                    identified = f"-{name}"
                elif q == 1:
                    identified = f"{p}*{name}"

                res_log10 = float(mpmath.log10(max(residual, mpmath.mpf(10) ** (-dps))))
                height = abs(p) + abs(q)

                log_lines.append(f"Direct match: {identified} (residual=1e{res_log10:.1f}, height={height})")

                if height < max_relation_height and (
                    best_result["residual_log10"] is None
                    or res_log10 < best_result["residual_log10"]
                ):
                    best_result = {
                        "success": 1,
                        "identified_as": identified,
                        "relation": {"coefficients": {name: f"{p}/{q}"}, "constant": 0},
                        "relation_height": float(height),
                        "residual_log10": res_log10,
                    }

    # Strategy 2: PSLQ with small basis subsets (groups of 5-6)
    # (Full PSLQ on 100+ constants is unstable; use targeted subsets)
    priority_bases = [
        ["pi", "pi^2", "ln(2)", "euler_gamma", "catalan"],
        ["zeta(2)", "zeta(3)", "zeta(4)", "zeta(5)", "zeta(6)"],
        ["zeta(7)", "zeta(8)", "zeta(9)", "zeta(10)"],
        ["ln(2)", "ln(3)", "ln(5)", "ln(7)"],
        ["sqrt(2)", "sqrt(3)", "sqrt(5)", "golden_ratio"],
    ]

    for group in priority_bases:
        names_in_group = [n for n in group if n in basis_mpf]
        if not names_in_group:
            continue

        vec = [target] + [basis_mpf[n] for n in names_in_group]
        try:
            rel = mpmath.pslq(vec, maxcoeff=int(max_relation_height), maxsteps=1000)
        except Exception:
            rel = None

        if rel is not None:
            # rel[0]*target + rel[1]*c1 + ... = 0
            # target = -sum(rel[i]*c_i) / rel[0]
            if rel[0] != 0:
                reconstructed = -sum(
                    mpmath.mpf(rel[i + 1]) * basis_mpf[names_in_group[i]]
                    for i in range(len(names_in_group))
                ) / mpmath.mpf(rel[0])
                residual = abs(target - reconstructed)
                res_log10 = float(mpmath.log10(max(residual, mpmath.mpf(10) ** (-dps))))
                height = sum(abs(r) for r in rel)

                # Build human-readable identification
                terms = []
                for i, name in enumerate(names_in_group):
                    coeff = -rel[i + 1]  # negative because rel[0]*target + ... = 0
                    if coeff != 0:
                        terms.append(f"{coeff}*{name}")
                denom = rel[0]
                identified = f"({' + '.join(terms)}) / {denom}" if denom != 1 else " + ".join(terms)

                log_lines.append(
                    f"PSLQ match ({','.join(names_in_group)}): "
                    f"{identified} (residual=1e{res_log10:.1f}, height={height})"
                )

                if height < max_relation_height and (
                    best_result["residual_log10"] is None
                    or res_log10 < best_result["residual_log10"]
                ):
                    best_result = {
                        "success": 1,
                        "identified_as": identified,
                        "relation": {
                            "coefficients": {
                                names_in_group[i]: int(-rel[i + 1])
                                for i in range(len(names_in_group))
                            },
                            "denominator": int(rel[0]),
                        },
                        "relation_height": float(height),
                        "residual_log10": res_log10,
                    }

    elapsed_ms = int((time.time() - t0) * 1000)
    log_lines.append(f"Total time: {elapsed_ms}ms")

    best_result["attempt_log"] = "\n".join(log_lines)
    best_result["runtime_ms"] = elapsed_ms
    best_result["basis_name"] = "standard_v1"

    return best_result
