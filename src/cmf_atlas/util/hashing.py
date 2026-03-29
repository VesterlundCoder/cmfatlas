"""Stable hashing for canonical fingerprints."""

import hashlib
import json


def stable_hash(obj) -> str:
    """Produce a deterministic SHA-256 hex digest (first 16 chars) for any JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def poly_fingerprint(coeffs: dict | list, *, normalize: bool = True) -> str:
    """Fingerprint a polynomial given as {exponent: coeff} or list of coeffs.

    If *normalize* is True, divide all coefficients by their GCD and fix
    the sign so the leading coefficient is positive.
    """
    if isinstance(coeffs, dict):
        items = sorted(coeffs.items())
    else:
        items = list(enumerate(coeffs))

    vals = [v for _, v in items if v != 0]
    if not vals:
        return stable_hash({"poly": []})

    if normalize:
        from math import gcd
        from functools import reduce

        g = reduce(gcd, (abs(int(v)) for v in vals if int(v) != 0), 0)
        if g == 0:
            g = 1
        sign = 1 if vals[-1] > 0 else -1
        items = [(k, sign * int(v) // g) for k, v in items if v != 0]

    return stable_hash({"poly": items})
