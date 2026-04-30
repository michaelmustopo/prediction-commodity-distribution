"""Percentile inversion — find x such that f(x) = target on a monotone curve.

Two functions, one per monotonicity direction:

* `invert_percentile` — for monotone-INCREASING curves (e.g. a CDF
  F(x) = P(X ≤ x) on settlement-style markets).

* `invert_decreasing` — for monotone-DECREASING curves (e.g. touch
  probabilities falling away from spot — "yes_price decreases as
  strike moves further out-of-the-money").

Both use cubic bisection inside the observed `[xs[0], xs[-1]]` range
and linear extrapolation from the two nearest interior points beyond
the tails. Each returns `(x, was_extrapolated)` so callers can flag
percentile bands that fell outside the observed market range.
"""

from __future__ import annotations

import numpy as np

from .hermite import hermite_eval


def invert_percentile(
    xs: np.ndarray,
    ys: np.ndarray,
    slopes: np.ndarray,
    target_p: float,
) -> tuple[float, bool]:
    """Find x such that CDF(x) = target_p on a monotone-INCREASING curve.

    Returns (x, was_extrapolated).

    Inside `[xs[0], xs[-1]]`: 40 iterations of cubic bisection (well
    within Decimal-4 precision). Outside: linear extrapolation from the
    two nearest interior points.
    """
    y_lo = float(ys[0])
    y_hi = float(ys[-1])

    if target_p <= y_lo:
        # Linear extrapolation LEFT from first two points
        if len(xs) < 2 or ys[1] == ys[0]:
            return float(xs[0]), True
        slope = (xs[1] - xs[0]) / (ys[1] - ys[0])
        return float(xs[0] - slope * (y_lo - target_p)), True

    if target_p >= y_hi:
        # Linear extrapolation RIGHT from last two points
        if len(xs) < 2 or ys[-1] == ys[-2]:
            return float(xs[-1]), True
        slope = (xs[-1] - xs[-2]) / (ys[-1] - ys[-2])
        return float(xs[-1] + slope * (target_p - y_hi)), True

    # Bisect inside the observed range — 40 iterations of float bisection
    # converges well inside Decimal-4 precision.
    lo, hi = float(xs[0]), float(xs[-1])
    for _ in range(40):
        mid = (lo + hi) / 2
        y_mid = hermite_eval(xs, ys, slopes, mid)
        if y_mid < target_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def invert_decreasing(
    xs: np.ndarray,
    ys: np.ndarray,
    slopes: np.ndarray,
    target_p: float,
) -> tuple[float, bool]:
    """Find x such that f(x) = target_p on a monotone-DECREASING curve.

    Sister to `invert_percentile` (which assumes monotone-increasing).
    Returns (x, was_extrapolated). Linear extrapolation beyond the
    observed range, same convention as the increasing version.

    NOTE: dx/dy is NEGATIVE on a decreasing curve, so the extrapolated
    delta has form `xs[edge] + dxdy * (target_p - y_edge)` for both
    tails (the sign of `(target_p - y_edge)` flips with which tail).
    Earlier draft of this helper had the signs inverted, producing
    extrapolated p10 closer to spot than p25 — caught by the
    asymmetric-band test cases.
    """
    y_lo = float(ys[-1])  # smallest y is at the END for decreasing
    y_hi = float(ys[0])  # largest y is at the START

    if target_p >= y_hi:
        # target_p above highest observed y → extrapolate LEFT (smaller x).
        # dx/dy < 0; (target_p - y_hi) > 0, so delta is negative → x left of xs[0].
        if len(xs) < 2 or ys[0] == ys[1]:
            return float(xs[0]), True
        dxdy = (xs[1] - xs[0]) / (ys[1] - ys[0])
        return float(xs[0] + dxdy * (target_p - y_hi)), True

    if target_p <= y_lo:
        # target_p below lowest observed y → extrapolate RIGHT (larger x).
        # dx/dy < 0; (target_p - y_lo) < 0, so delta is positive → x right of xs[-1].
        if len(xs) < 2 or ys[-1] == ys[-2]:
            return float(xs[-1]), True
        dxdy = (xs[-1] - xs[-2]) / (ys[-1] - ys[-2])
        return float(xs[-1] + dxdy * (target_p - y_lo)), True

    lo, hi = float(xs[0]), float(xs[-1])
    for _ in range(40):
        mid = (lo + hi) / 2
        y_mid = hermite_eval(xs, ys, slopes, mid)
        # Decreasing curve: if y_mid > target, lower y is to the right.
        if y_mid > target_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False
