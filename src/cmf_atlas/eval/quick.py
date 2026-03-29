"""Quick evaluation — low-precision convergence diagnostics for triage.

Evaluates PCF/series objects at moderate precision to estimate:
    - convergence score (0..1)
    - stability score (0..1)
    - limit estimate
    - error estimate
"""

import time
from typing import Any

import mpmath


def evaluate_pcf_quick(
    a_coeffs: list[int],
    b_coeffs: list[int],
    depth: int = 200,
    dps: int = 50,
) -> dict:
    """Quick evaluation of a PCF defined by polynomial a(n), b(n).

    Computes the continued fraction a(0) + b(1)/(a(1) + b(2)/(a(2) + ...))
    using backward recurrence (Lentz's method variant).

    Returns dict with limit_estimate, error_estimate, convergence_score,
    stability_score, runtime_ms.
    """
    mpmath.mp.dps = dps
    t0 = time.time()

    def a_fn(n):
        return sum(mpmath.mpf(c) * mpmath.power(n, i) for i, c in enumerate(a_coeffs))

    def b_fn(n):
        return sum(mpmath.mpf(c) * mpmath.power(n, i) for i, c in enumerate(b_coeffs))

    try:
        # Forward recurrence: p_{-1}=1, p_0=a(0), q_{-1}=0, q_0=1
        # p_n = a(n)*p_{n-1} + b(n)*p_{n-2}
        # q_n = a(n)*q_{n-1} + b(n)*q_{n-2}
        p_prev, p_curr = mpmath.mpf(1), a_fn(0)
        q_prev, q_curr = mpmath.mpf(0), mpmath.mpf(1)

        estimates = []
        for n in range(1, depth + 1):
            an = a_fn(n)
            bn = b_fn(n)
            p_new = an * p_curr + bn * p_prev
            q_new = an * q_curr + bn * q_prev
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new

            if q_curr != 0:
                est = p_curr / q_curr
                estimates.append(float(est))

        if not estimates:
            return _fail_result(time.time() - t0)

        # Convergence analysis
        limit = estimates[-1]
        if len(estimates) >= 10:
            tail = estimates[-10:]
            error = max(abs(tail[i] - tail[i - 1]) for i in range(1, len(tail)))
        else:
            error = abs(estimates[-1] - estimates[-2]) if len(estimates) >= 2 else float("inf")

        # Convergence score: how many digits stabilized
        if error > 0 and error < 1e10:
            import math
            stable_digits = max(0, -math.log10(max(error, 1e-50)))
            conv_score = min(1.0, stable_digits / dps)
        else:
            conv_score = 0.0

        # Stability score: monotonicity check on last 20 estimates
        stability = _compute_stability(estimates[-min(20, len(estimates)):])

        # Oscillation flag
        oscillation = _check_oscillation(estimates[-min(20, len(estimates)):])

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "limit_estimate": str(mpmath.nstr(mpmath.mpf(limit), dps - 5)),
            "error_estimate": float(error),
            "convergence_score": float(conv_score),
            "stability_score": float(stability),
            "oscillation_flag": int(oscillation),
            "runtime_ms": elapsed_ms,
            "depth": depth,
            "dps": dps,
        }

    except Exception as e:
        return _fail_result(time.time() - t0, str(e))


def _compute_stability(estimates: list[float]) -> float:
    """Score 0..1 based on how smoothly estimates converge."""
    if len(estimates) < 3:
        return 0.0

    diffs = [abs(estimates[i] - estimates[i - 1]) for i in range(1, len(estimates))]
    if not diffs or max(diffs) == 0:
        return 1.0

    # Check if diffs are monotonically decreasing (good convergence)
    decreasing = sum(1 for i in range(1, len(diffs)) if diffs[i] <= diffs[i - 1] * 1.1)
    ratio = decreasing / max(len(diffs) - 1, 1)

    # Also factor in final convergence level
    import math
    final_diff = diffs[-1]
    if final_diff > 0:
        digit_score = min(1.0, max(0, -math.log10(max(final_diff, 1e-50))) / 30)
    else:
        digit_score = 1.0

    return 0.6 * ratio + 0.4 * digit_score


def _check_oscillation(estimates: list[float]) -> bool:
    """Check if estimates oscillate (sign changes in consecutive differences)."""
    if len(estimates) < 4:
        return False
    diffs = [estimates[i] - estimates[i - 1] for i in range(1, len(estimates))]
    sign_changes = sum(
        1 for i in range(1, len(diffs))
        if diffs[i] * diffs[i - 1] < 0
    )
    return sign_changes > len(diffs) * 0.4


def evaluate_series_quick(
    terms: list[float | str],
    dps: int = 50,
) -> dict:
    """Quick evaluation of a series given explicit terms.

    Computes partial sums and convergence diagnostics.
    """
    mpmath.mp.dps = dps
    t0 = time.time()

    try:
        partial_sums = []
        s = mpmath.mpf(0)
        for t in terms:
            s += mpmath.mpf(str(t))
            partial_sums.append(float(s))

        if not partial_sums:
            return _fail_result(time.time() - t0)

        limit = partial_sums[-1]
        if len(partial_sums) >= 10:
            tail = partial_sums[-10:]
            error = max(abs(tail[i] - tail[i - 1]) for i in range(1, len(tail)))
        else:
            error = abs(partial_sums[-1] - partial_sums[0]) if len(partial_sums) >= 2 else float("inf")

        import math
        stable_digits = max(0, -math.log10(max(error, 1e-50))) if error > 0 else dps
        conv_score = min(1.0, stable_digits / dps)
        stability = _compute_stability(partial_sums[-min(20, len(partial_sums)):])

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "limit_estimate": str(limit),
            "error_estimate": float(error),
            "convergence_score": float(conv_score),
            "stability_score": float(stability),
            "oscillation_flag": 0,
            "runtime_ms": elapsed_ms,
            "n_terms": len(terms),
            "dps": dps,
        }

    except Exception as e:
        return _fail_result(time.time() - t0, str(e))


def _fail_result(elapsed: float, error_msg: str = "") -> dict:
    return {
        "limit_estimate": None,
        "error_estimate": float("inf"),
        "convergence_score": 0.0,
        "stability_score": 0.0,
        "oscillation_flag": 0,
        "runtime_ms": int(elapsed * 1000),
        "error": error_msg,
    }
