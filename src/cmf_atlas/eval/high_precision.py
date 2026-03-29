"""High-precision evaluation — hundreds to thousands of digits.

Used for the recognition pipeline and final verification of candidates.
"""

import time

import mpmath


def evaluate_pcf_hp(
    a_coeffs: list[int],
    b_coeffs: list[int],
    depth: int = 2000,
    dps: int = 500,
    checkpoints: list[int] | None = None,
) -> dict:
    """High-precision PCF evaluation with checkpoint convergence tracking.

    Parameters
    ----------
    a_coeffs, b_coeffs : polynomial coefficients for a(n), b(n)
    depth : walk depth
    dps : decimal digits of precision
    checkpoints : list of depths at which to record intermediate limits

    Returns
    -------
    dict with limit_estimate (high-precision string), checkpoint_estimates,
    convergence_rate, stable_digits, runtime_ms
    """
    mpmath.mp.dps = dps + 50  # extra guard digits
    t0 = time.time()

    if checkpoints is None:
        checkpoints = [depth // 4, depth // 2, 3 * depth // 4, depth]

    def a_fn(n):
        return sum(mpmath.mpf(c) * mpmath.power(n, i) for i, c in enumerate(a_coeffs))

    def b_fn(n):
        return sum(mpmath.mpf(c) * mpmath.power(n, i) for i, c in enumerate(b_coeffs))

    try:
        p_prev, p_curr = mpmath.mpf(1), a_fn(0)
        q_prev, q_curr = mpmath.mpf(0), mpmath.mpf(1)

        checkpoint_results = {}
        checkpoint_set = set(checkpoints)

        for n in range(1, depth + 1):
            an = a_fn(n)
            bn = b_fn(n)
            p_new = an * p_curr + bn * p_prev
            q_new = an * q_curr + bn * q_prev
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new

            if n in checkpoint_set and q_curr != 0:
                est = p_curr / q_curr
                checkpoint_results[n] = mpmath.nstr(est, dps)

        if q_curr == 0:
            return {
                "limit_estimate": None,
                "stable_digits": 0,
                "runtime_ms": int((time.time() - t0) * 1000),
                "error": "q=0 at final depth",
            }

        final_limit = p_curr / q_curr
        limit_str = mpmath.nstr(final_limit, dps)

        # Estimate stable digits by comparing last two checkpoints
        stable_digits = 0
        sorted_cp = sorted(checkpoint_results.keys())
        if len(sorted_cp) >= 2:
            prev_est = mpmath.mpf(checkpoint_results[sorted_cp[-2]])
            curr_est = mpmath.mpf(checkpoint_results[sorted_cp[-1]])
            diff = abs(curr_est - prev_est)
            if diff > 0:
                stable_digits = max(0, int(-mpmath.log10(diff)))
            else:
                stable_digits = dps

        # Convergence rate: log10(error) / log10(depth)
        conv_rate = None
        if len(sorted_cp) >= 3:
            d1 = sorted_cp[-3]
            d2 = sorted_cp[-1]
            e1 = abs(mpmath.mpf(checkpoint_results[d1]) - final_limit)
            e2 = abs(mpmath.mpf(checkpoint_results[sorted_cp[-2]]) - final_limit)
            if e1 > 0 and e2 > 0 and d2 > d1:
                conv_rate = float(
                    (mpmath.log10(e2) - mpmath.log10(e1))
                    / (mpmath.log10(d2) - mpmath.log10(d1))
                )

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "limit_estimate": limit_str,
            "stable_digits": stable_digits,
            "convergence_rate": conv_rate,
            "checkpoints": {str(k): v for k, v in checkpoint_results.items()},
            "depth": depth,
            "dps": dps,
            "runtime_ms": elapsed_ms,
        }

    except Exception as e:
        return {
            "limit_estimate": None,
            "stable_digits": 0,
            "runtime_ms": int((time.time() - t0) * 1000),
            "error": str(e),
        }
