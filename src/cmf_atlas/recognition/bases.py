"""Constant bases for PSLQ recognition.

Each basis is a named collection of high-precision constants used as
PSLQ targets. Bases are versioned and stored as JSON for auditability.
"""

import mpmath


def build_standard_basis(dps: int = 100) -> dict[str, str]:
    """Build the standard constant basis at the given precision.

    Returns dict of {name: high_precision_string}.
    """
    mpmath.mp.dps = dps + 20

    constants = {}

    # Pi variants
    constants["pi"] = mpmath.nstr(mpmath.pi, dps)
    constants["pi^2"] = mpmath.nstr(mpmath.pi ** 2, dps)
    constants["pi^3"] = mpmath.nstr(mpmath.pi ** 3, dps)
    constants["pi^4"] = mpmath.nstr(mpmath.pi ** 4, dps)
    constants["1/pi"] = mpmath.nstr(1 / mpmath.pi, dps)
    constants["pi^2/6"] = mpmath.nstr(mpmath.pi ** 2 / 6, dps)
    constants["pi^4/90"] = mpmath.nstr(mpmath.pi ** 4 / 90, dps)

    # Zeta values
    for s in range(2, 16):
        constants[f"zeta({s})"] = mpmath.nstr(mpmath.zeta(s), dps)

    # Logarithms
    for k in range(2, 8):
        constants[f"ln({k})"] = mpmath.nstr(mpmath.log(k), dps)
    constants["ln(2)^2"] = mpmath.nstr(mpmath.log(2) ** 2, dps)

    # Classical constants
    constants["euler_gamma"] = mpmath.nstr(mpmath.euler, dps)
    constants["catalan"] = mpmath.nstr(mpmath.catalan, dps)
    constants["apery"] = mpmath.nstr(mpmath.zeta(3), dps)  # alias
    constants["sqrt(2)"] = mpmath.nstr(mpmath.sqrt(2), dps)
    constants["sqrt(3)"] = mpmath.nstr(mpmath.sqrt(3), dps)
    constants["sqrt(5)"] = mpmath.nstr(mpmath.sqrt(5), dps)
    constants["golden_ratio"] = mpmath.nstr(mpmath.phi, dps)

    # Reciprocals of factorials
    for k in range(1, 11):
        fk = mpmath.factorial(k)
        constants[f"1/{k}!"] = mpmath.nstr(1 / fk, dps)

    # Digamma closed forms
    for p in range(1, 7):
        for q in range(p + 1, 7):
            try:
                val = mpmath.digamma(mpmath.mpf(p) / mpmath.mpf(q))
                constants[f"psi({p}/{q})"] = mpmath.nstr(val, dps)
            except Exception:
                pass

    return constants


def build_extended_basis(dps: int = 100) -> dict[str, str]:
    """Extended basis including polylogarithms and more combinations."""
    constants = build_standard_basis(dps)
    mpmath.mp.dps = dps + 20

    # Polylogarithms
    for s in range(2, 6):
        for x_num, x_den in [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]:
            x = mpmath.mpf(x_num) / mpmath.mpf(x_den)
            try:
                val = mpmath.polylog(s, x)
                constants[f"Li_{s}({x_num}/{x_den})"] = mpmath.nstr(val, dps)
            except Exception:
                pass

    # Mixed products
    constants["pi*ln(2)"] = mpmath.nstr(mpmath.pi * mpmath.log(2), dps)
    constants["pi*euler_gamma"] = mpmath.nstr(mpmath.pi * mpmath.euler, dps)
    constants["zeta(3)/pi^2"] = mpmath.nstr(mpmath.zeta(3) / mpmath.pi ** 2, dps)

    return constants


def basis_to_mpf_dict(basis: dict[str, str], dps: int = 100) -> dict[str, mpmath.mpf]:
    """Convert string basis to mpmath.mpf values."""
    mpmath.mp.dps = dps + 20
    return {name: mpmath.mpf(val) for name, val in basis.items()}
