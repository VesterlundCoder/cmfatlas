"""PCF generator — produce polynomial continued fractions constrained to target degrees.

Generates PCFs with a(n), b(n) of specified degrees using:
    - Random coefficient sampling with bounded norms
    - Template-based generation (Apéry-like, RM-like patterns)
    - Mutation/crossover from existing PCFs
"""

import random
from typing import Any


def generate_pcf_candidates(
    deg_a: int,
    deg_b: int,
    n_candidates: int = 100,
    coeff_range: int = 10,
    seed: int | None = None,
    templates: list[str] | None = None,
) -> list[dict]:
    """Generate PCF candidates with specified polynomial degrees.

    Parameters
    ----------
    deg_a : target degree for a(n)
    deg_b : target degree for b(n)
    n_candidates : number of candidates to generate
    coeff_range : max absolute value of random coefficients
    seed : random seed
    templates : optional list of template names to bias generation

    Returns
    -------
    List of dicts with keys: a_coeffs, b_coeffs, provenance
    """
    if seed is not None:
        random.seed(seed)

    candidates = []

    # Strategy 1: Pure random with leading coefficient fixed
    n_random = n_candidates // 2
    for i in range(n_random):
        a = [random.randint(-coeff_range, coeff_range) for _ in range(deg_a)]
        a.append(random.choice([c for c in range(-coeff_range, coeff_range + 1) if c != 0]))

        b = [random.randint(-coeff_range, coeff_range) for _ in range(deg_b)]
        b.append(random.choice([c for c in range(-coeff_range, coeff_range + 1) if c != 0]))

        candidates.append({
            "a_coeffs": a,
            "b_coeffs": b,
            "provenance": f"random_pcf(deg_a={deg_a},deg_b={deg_b},idx={i})",
        })

    # Strategy 2: Template-based (Apéry-like, RM-like patterns)
    n_template = n_candidates - n_random
    for i in range(n_template):
        a, b = _template_pcf(deg_a, deg_b, coeff_range)
        candidates.append({
            "a_coeffs": a,
            "b_coeffs": b,
            "provenance": f"template_pcf(deg_a={deg_a},deg_b={deg_b},idx={i})",
        })

    return candidates


def _template_pcf(deg_a: int, deg_b: int, coeff_range: int) -> tuple[list[int], list[int]]:
    """Generate a PCF using structured templates.

    Templates include:
    - Linear a + quadratic b (Apéry-like)
    - Pochhammer-ratio patterns
    - Alternating sign patterns for b
    """
    template = random.choice(["apery_like", "alternating_b", "shifted_linear", "sparse"])

    if template == "apery_like":
        # a(n) = c1*n + c0, b(n) = -c2*n^2 + c3*n (when deg allows)
        a = [0] * (deg_a + 1)
        a[0] = random.randint(0, coeff_range)
        if deg_a >= 1:
            a[1] = random.randint(1, coeff_range)

        b = [0] * (deg_b + 1)
        if deg_b >= 2:
            b[2] = -random.randint(1, coeff_range)
            b[1] = random.randint(0, coeff_range)
        elif deg_b >= 1:
            b[1] = -random.randint(1, coeff_range)
        b[0] = random.randint(-coeff_range, coeff_range)

    elif template == "alternating_b":
        # b with alternating sign pattern in coefficients
        a = [random.randint(-coeff_range, coeff_range) for _ in range(deg_a + 1)]
        if a[-1] == 0:
            a[-1] = 1

        b = []
        for j in range(deg_b + 1):
            sign = 1 if j % 2 == 0 else -1
            b.append(sign * random.randint(1, coeff_range))

    elif template == "shifted_linear":
        # a(n) = (2n+1)*c, b(n) = -(n*(n+1))^k pattern
        a = [0] * (deg_a + 1)
        c = random.randint(1, coeff_range)
        if deg_a >= 1:
            a[1] = 2 * c
            a[0] = c
        else:
            a[0] = c

        b = [0] * (deg_b + 1)
        if deg_b >= 2:
            b[2] = -random.randint(1, coeff_range // 2 + 1)
            b[1] = -b[2]  # -(n^2 + n) pattern
        elif deg_b >= 1:
            b[1] = -random.randint(1, coeff_range)

    else:  # sparse
        # Most coefficients zero, only leading + constant
        a = [0] * (deg_a + 1)
        a[0] = random.randint(1, coeff_range)
        a[-1] = random.randint(1, coeff_range)

        b = [0] * (deg_b + 1)
        b[0] = random.choice([-1, 1]) * random.randint(1, coeff_range)
        b[-1] = random.choice([-1, 1]) * random.randint(1, coeff_range)

    return a, b


def mutate_pcf(
    a_coeffs: list[int],
    b_coeffs: list[int],
    mutation_strength: float = 0.3,
) -> tuple[list[int], list[int]]:
    """Mutate a PCF by perturbing coefficients."""
    a = list(a_coeffs)
    b = list(b_coeffs)

    for i in range(len(a)):
        if random.random() < mutation_strength:
            a[i] += random.choice([-2, -1, 1, 2])

    for i in range(len(b)):
        if random.random() < mutation_strength:
            b[i] += random.choice([-2, -1, 1, 2])

    # Ensure leading coefficients are nonzero
    if a[-1] == 0:
        a[-1] = random.choice([-1, 1])
    if b[-1] == 0:
        b[-1] = random.choice([-1, 1])

    return a, b
