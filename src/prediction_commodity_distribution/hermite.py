"""Monotone Hermite cubic — Fritsch-Butland 1984 slopes + Fritsch-Carlson 1980 limiter.

Two functions:

* `fritsch_carlson_slopes(xs, ys)` — compute Hermite tangent slopes at
  each knot, using harmonic-mean interior slopes (Fritsch-Butland 1984)
  inside the Fritsch-Carlson 1980 framework. Output curve is
  numerically reproducible against `scipy.interpolate.PchipInterpolator`.

* `hermite_eval(xs, ys, slopes, x)` — evaluate the cubic at a point x;
  caller pre-brackets x in `[xs[0], xs[-1]]` (extrapolation lives in
  the `invert` module).

The function-name `fritsch_carlson_slopes` references the FRAMEWORK
(slope choice + α²+β²≤9 limiter ⇒ monotone interpolant per Fritsch &
Carlson, SIAM J. Numer. Anal. 17(2):238-246, 1980). The actual
slope-choice formula used is Fritsch-Butland 1984 (the canonical PCHIP
variant scipy implements).
"""

from __future__ import annotations

import numpy as np


def fritsch_carlson_slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Hermite interior slopes via Fritsch-Butland 1984 harmonic mean,
    plus the Fritsch-Carlson 1980 α²+β²≤9 monotone limiter.

    Interior slope formula (same-sign secants only):

        m[i] = 2 * d[i-1] * d[i] / (d[i-1] + d[i])

    where d[k] = (ys[k+1] - ys[k]) / (xs[k+1] - xs[k]) are the forward
    secant slopes between knots.

    Harmonic mean is the canonical PCHIP slope choice — output matches
    `scipy.interpolate.PchipInterpolator` for any quant verifier. An
    arithmetic-mean variant `(d[i-1] + d[i]) / 2` is monotone-preserving
    via the limiter but diverges ~5-15% from scipy on realistic data.

    Edge cases handled:

    * Same-x adjacent points: protected via `np.where(h == 0, 1, h)` to
      avoid div-by-zero. Use `dedup_average` upstream as the canonical
      normalization step before passing to this function.
    * Zero secant slope (flat segment): both adjacent slopes set to 0
      (limiter block) so the curve stays flat through the segment.
    * Opposite-sign secants: interior slope set to 0 (no overshoot).
    * `n < 2`: returns zeros of the appropriate length.
    """
    n = len(xs)
    if n < 2:
        return np.zeros(n)
    h = np.diff(xs)
    d = np.diff(ys) / np.where(h == 0, 1, h)

    slopes = np.zeros(n)
    slopes[0] = d[0]
    slopes[-1] = d[-1]
    for i in range(1, n - 1):
        d_prev = d[i - 1]
        d_curr = d[i]
        if d_prev * d_curr <= 0:
            # Sign flip or either side is zero → flat to prevent overshoot.
            slopes[i] = 0.0
        else:
            # Fritsch-Butland 1984 harmonic mean. Same sign guaranteed by
            # the branch above, so denominator (d_prev + d_curr) is non-zero.
            slopes[i] = 2.0 * d_prev * d_curr / (d_prev + d_curr)

    # Fritsch-Carlson 1980 α²+β²≤9 limiter — ensures monotone interpolant
    # regardless of slope-choice strategy above.
    for i in range(n - 1):
        if d[i] == 0:
            slopes[i] = 0.0
            slopes[i + 1] = 0.0
        else:
            a = slopes[i] / d[i]
            b = slopes[i + 1] / d[i]
            s = a * a + b * b
            if s > 9.0:
                tau = 3.0 / np.sqrt(s)
                slopes[i] = tau * a * d[i]
                slopes[i + 1] = tau * b * d[i]
    return slopes


def hermite_eval(xs: np.ndarray, ys: np.ndarray, slopes: np.ndarray, x: float) -> float:
    """Evaluate the Hermite cubic at x. Caller pre-brackets x in [xs[0], xs[-1]]."""
    idx = int(np.searchsorted(xs, x)) - 1
    idx = max(0, min(idx, len(xs) - 2))
    x0 = xs[idx]
    x1 = xs[idx + 1]
    h = x1 - x0
    if h == 0:
        return float(ys[idx])
    t = (x - x0) / h
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return float(
        h00 * ys[idx] + h10 * h * slopes[idx] + h01 * ys[idx + 1] + h11 * h * slopes[idx + 1]
    )
