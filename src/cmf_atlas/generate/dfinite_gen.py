"""D-finite generator — produce linear recurrence operators constrained to target order/degree.

Generates holonomic recurrence operators with specified order and polynomial degree:
    - Random coefficient polynomials with bounded norms
    - Ore-algebra inspired constructions
    - Mutation from existing operators
"""

import random
from typing import Any


def generate_dfinite_candidates(
    rec_order: int,
    max_poly_degree: int,
    n_candidates: int = 100,
    coeff_range: int = 10,
    seed: int | None = None,
) -> list[dict]:
    """Generate D-finite recurrence operators with specified order and degree.

    Parameters
    ----------
    rec_order : order r of the recurrence
    max_poly_degree : maximum degree of coefficient polynomials p_k(n)
    n_candidates : number of candidates
    coeff_range : max absolute value of coefficients
    seed : random seed

    Returns
    -------
    List of dicts with keys: operator, provenance
    """
    if seed is not None:
        random.seed(seed)

    candidates = []

    # Strategy 1: Random operators with bounded coefficients
    n_random = n_candidates // 2
    for i in range(n_random):
        op = _random_operator(rec_order, max_poly_degree, coeff_range)
        candidates.append({
            "operator": op,
            "provenance": f"random_dfinite(order={rec_order},deg={max_poly_degree},idx={i})",
        })

    # Strategy 2: Structured operators (Apéry-like, hypergeometric-derived)
    n_structured = n_candidates - n_random
    for i in range(n_structured):
        op = _structured_operator(rec_order, max_poly_degree, coeff_range)
        candidates.append({
            "operator": op,
            "provenance": f"structured_dfinite(order={rec_order},deg={max_poly_degree},idx={i})",
        })

    return candidates


def _random_operator(order: int, max_deg: int, coeff_range: int) -> list[list[int]]:
    """Generate a random recurrence operator.

    operator[k] = coefficients of p_k(n) = c_0 + c_1*n + ... + c_d*n^d
    for k = 0, 1, ..., order.
    """
    op = []
    for k in range(order + 1):
        deg = random.randint(0, max_deg)
        coeffs = [random.randint(-coeff_range, coeff_range) for _ in range(deg)]
        # Leading coefficient nonzero
        lead = random.choice([c for c in range(-coeff_range, coeff_range + 1) if c != 0])
        coeffs.append(lead)
        op.append(coeffs)

    # Ensure leading polynomial (highest shift) has nonzero leading coefficient
    if op[-1][-1] == 0:
        op[-1][-1] = random.choice([-1, 1])

    return op


def _structured_operator(order: int, max_deg: int, coeff_range: int) -> list[list[int]]:
    """Generate structured operators inspired by known families.

    Templates:
    - Apéry-type: (n+1)^r * a_{n+1} - poly(n) * a_n + ...
    - Balanced: symmetric coefficient patterns
    - Sparse: most coefficients zero
    """
    template = random.choice(["apery_type", "balanced", "sparse", "factored_leading"])

    if template == "apery_type" and order >= 2:
        # (n+1)^d * a_{n+r} - P(n) * a_{n+r-1} + Q(n) * a_n
        op = [[0] for _ in range(order + 1)]

        # Leading: (n+1)^min(max_deg, 3)
        d = min(max_deg, 3)
        import sympy as sp
        n = sp.Symbol('n')
        leading = sp.Poly(sp.expand((n + 1) ** d), n)
        op[-1] = [int(c) for c in reversed(leading.all_coeffs())]

        # Middle terms: random but bounded
        for k in range(1, order):
            deg_k = random.randint(0, max_deg)
            op[k] = [random.randint(-coeff_range // 2, coeff_range // 2) for _ in range(deg_k + 1)]
            if not op[k]:
                op[k] = [0]

        # Trailing: negative to create alternating pattern
        deg_0 = random.randint(1, max_deg)
        op[0] = [random.randint(-coeff_range, 0) for _ in range(deg_0 + 1)]
        if op[0][-1] == 0:
            op[0][-1] = -1

        return op

    elif template == "balanced":
        # Symmetric coefficient patterns: p_k ≈ ±p_{order-k}
        op = []
        half = (order + 1) // 2
        for k in range(half):
            deg_k = random.randint(0, max_deg)
            coeffs = [random.randint(-coeff_range, coeff_range) for _ in range(deg_k + 1)]
            if coeffs[-1] == 0:
                coeffs[-1] = random.choice([-1, 1])
            op.append(coeffs)

        # Mirror
        while len(op) <= order:
            mirror_idx = order - len(op)
            if mirror_idx < len(op):
                sign = random.choice([-1, 1])
                op.append([sign * c for c in op[mirror_idx]])
            else:
                op.append([random.randint(-coeff_range, coeff_range)])

        return op

    elif template == "sparse":
        # Only leading, trailing, and possibly one middle term nonzero
        op = [[0] for _ in range(order + 1)]

        # Leading
        deg_lead = random.randint(1, max_deg)
        op[-1] = [random.randint(-coeff_range, coeff_range) for _ in range(deg_lead + 1)]
        if op[-1][-1] == 0:
            op[-1][-1] = 1

        # Trailing
        deg_trail = random.randint(0, max_deg)
        op[0] = [random.randint(-coeff_range, coeff_range) for _ in range(deg_trail + 1)]
        if op[0][-1] == 0:
            op[0][-1] = -1

        # One random middle term
        if order >= 2:
            mid = random.randint(1, order - 1)
            deg_mid = random.randint(0, max_deg)
            op[mid] = [random.randint(-coeff_range, coeff_range) for _ in range(deg_mid + 1)]

        return op

    else:  # factored_leading
        # Leading polynomial is a product of small linear factors
        import sympy as sp
        n = sp.Symbol('n')

        d = min(max_deg, 4)
        shifts = [random.randint(0, 5) for _ in range(d)]
        leading = sp.Integer(1)
        for s in shifts:
            leading *= (n + s)
        leading_poly = sp.Poly(sp.expand(leading), n)
        lead_coeffs = [int(c) for c in reversed(leading_poly.all_coeffs())]

        op = []
        for k in range(order):
            deg_k = random.randint(0, max_deg)
            op.append([random.randint(-coeff_range, coeff_range) for _ in range(deg_k + 1)])
        op.append(lead_coeffs)

        return op


def mutate_operator(
    operator: list[list[int]],
    mutation_strength: float = 0.3,
) -> list[list[int]]:
    """Mutate a D-finite operator by perturbing coefficients."""
    op = [list(p) for p in operator]

    for k in range(len(op)):
        for i in range(len(op[k])):
            if random.random() < mutation_strength:
                op[k][i] += random.choice([-2, -1, 1, 2])

    # Ensure leading polynomial has nonzero leading coefficient
    if op[-1][-1] == 0:
        op[-1][-1] = random.choice([-1, 1])

    return op
