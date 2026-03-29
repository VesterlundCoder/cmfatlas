"""Hypergeometric generator — produce term ratios P(n)/Q(n) constrained to target degrees.

Generates hypergeometric series candidates with specified numerator/denominator degrees using:
    - Factored form generation (products of linear factors)
    - Random polynomial coefficient sampling
    - Grammar-based rational function construction
"""

import random
from typing import Any


def generate_hypergeom_candidates(
    degP: int,
    degQ: int,
    n_candidates: int = 100,
    coeff_range: int = 10,
    seed: int | None = None,
) -> list[dict]:
    """Generate hypergeometric candidates with specified term ratio degrees.

    Parameters
    ----------
    degP : degree of numerator P(n)
    degQ : degree of denominator Q(n)
    n_candidates : number of candidates
    coeff_range : max absolute value of coefficients / shifts
    seed : random seed

    Returns
    -------
    List of dicts with keys: numerator, denominator, provenance
    """
    if seed is not None:
        random.seed(seed)

    candidates = []

    # Strategy 1: Factored form — products of (n + shift)
    n_factored = n_candidates // 3
    for i in range(n_factored):
        num, den = _factored_ratio(degP, degQ, coeff_range)
        candidates.append({
            "numerator": num,
            "denominator": den,
            "provenance": f"factored_hyper(degP={degP},degQ={degQ},idx={i})",
        })

    # Strategy 2: Random polynomial coefficients
    n_random = n_candidates // 3
    for i in range(n_random):
        num = _random_poly(degP, coeff_range)
        den = _random_poly(degQ, coeff_range, positive_lead=True)
        candidates.append({
            "numerator": num,
            "denominator": den,
            "provenance": f"random_hyper(degP={degP},degQ={degQ},idx={i})",
        })

    # Strategy 3: pFq-inspired templates
    n_template = n_candidates - n_factored - n_random
    for i in range(n_template):
        num, den = _pfq_template(degP, degQ, coeff_range)
        candidates.append({
            "numerator": num,
            "denominator": den,
            "provenance": f"pfq_hyper(degP={degP},degQ={degQ},idx={i})",
        })

    return candidates


def _factored_ratio(degP: int, degQ: int, max_shift: int) -> tuple[list[int], list[int]]:
    """Generate P(n)/Q(n) as products of linear factors (n+a_i)/(n+b_j).

    P(n) = prod_{i=1}^{degP} (n + a_i)
    Q(n) = prod_{j=1}^{degQ} (n + b_j)

    Returns expanded polynomial coefficients.
    """
    import sympy as sp
    n = sp.Symbol('n')

    shifts_p = [random.randint(-max_shift, max_shift) for _ in range(degP)]
    shifts_q = [random.randint(1, max_shift) for _ in range(degQ)]  # positive to avoid division by zero

    P = sp.Integer(1)
    for a in shifts_p:
        P *= (n + a)

    Q = sp.Integer(1)
    for b in shifts_q:
        Q *= (n + b)

    P_expanded = sp.Poly(sp.expand(P), n)
    Q_expanded = sp.Poly(sp.expand(Q), n)

    num = [int(c) for c in reversed(P_expanded.all_coeffs())]
    den = [int(c) for c in reversed(Q_expanded.all_coeffs())]

    return num, den


def _random_poly(deg: int, coeff_range: int, positive_lead: bool = False) -> list[int]:
    """Generate random polynomial coefficients [c0, c1, ..., c_deg]."""
    coeffs = [random.randint(-coeff_range, coeff_range) for _ in range(deg)]
    lead = random.randint(1, coeff_range) if positive_lead else random.choice(
        [c for c in range(-coeff_range, coeff_range + 1) if c != 0]
    )
    coeffs.append(lead)
    return coeffs


def _pfq_template(degP: int, degQ: int, max_shift: int) -> tuple[list[int], list[int]]:
    """Generate pFq-inspired hypergeometric term ratios.

    For p+1_F_q: ratio = prod(n+a_i) / (prod(n+b_j) * (n+1))
    """
    import sympy as sp
    n = sp.Symbol('n')

    # Upper parameters (Pochhammer numerators)
    p = degP
    upper = sorted(random.sample(range(-max_shift, max_shift + 1), min(p, 2 * max_shift + 1)))
    while len(upper) < p:
        upper.append(random.randint(-max_shift, max_shift))

    # Lower parameters (Pochhammer denominators, positive)
    q = degQ
    lower = sorted(random.sample(range(1, max_shift + 1), min(q, max_shift)))
    while len(lower) < q:
        lower.append(random.randint(1, max_shift))

    P = sp.Integer(1)
    for a in upper:
        P *= (n + a)

    Q = sp.Integer(1)
    for b in lower:
        Q *= (n + b)

    P_expanded = sp.Poly(sp.expand(P), n)
    Q_expanded = sp.Poly(sp.expand(Q), n)

    num = [int(c) for c in reversed(P_expanded.all_coeffs())]
    den = [int(c) for c in reversed(Q_expanded.all_coeffs())]

    return num, den
