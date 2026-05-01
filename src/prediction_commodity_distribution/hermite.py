"""Monotone Hermite cubic — scipy.PchipInterpolator's exact algorithm.

Two functions:

* `fritsch_carlson_slopes(xs, ys)` — compute Hermite tangent slopes at
  each knot using the canonical PCHIP algorithm: Fritsch-Butland 1984
  weighted harmonic mean for interior slopes + Cleve Moler 3-point
  one-sided estimate (with sign + shape correction) for endpoint
  slopes. Byte-equivalent to `scipy.interpolate.PchipInterpolator`
  for any quant verifier (locked by `tests/test_scipy_parity.py`).

* `hermite_eval(xs, ys, slopes, x)` — evaluate the cubic at a point x;
  caller pre-brackets x in `[xs[0], xs[-1]]` (extrapolation lives in
  the `invert` module).

Function name `fritsch_carlson_slopes` references the broader
Fritsch-Carlson 1980 monotone-Hermite framework (SIAM J. Numer. Anal.
17(2):238-246) under which PCHIP sits. Slope choice is the
Fritsch-Butland 1984 weighted variant scipy adopted; endpoint formula
is from Cleve Moler, *Numerical Computing with MATLAB* §3.6
(`pchiptx.m`).
"""

from __future__ import annotations

import numpy as np


def fritsch_carlson_slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Hermite tangent slopes — scipy.PchipInterpolator's exact algorithm.

    Interior knots (i in [1, n-2]) — Fritsch-Butland 1984 weighted
    harmonic mean::

        h_prev = xs[i] - xs[i-1]
        h_curr = xs[i+1] - xs[i]
        d_prev = (ys[i] - ys[i-1]) / h_prev
        d_curr = (ys[i+1] - ys[i]) / h_curr

        if sign(d_prev) != sign(d_curr) or d_prev == 0 or d_curr == 0:
            slope[i] = 0
        else:
            w1 = 2*h_curr + h_prev   # weight on d_prev
            w2 = h_curr + 2*h_prev   # weight on d_curr
            slope[i] = (w1 + w2) / (w1/d_prev + w2/d_curr)

    For uniform spacing (h_prev == h_curr) this collapses to the simple
    harmonic mean `2*d_prev*d_curr/(d_prev+d_curr)`.

    Endpoints — Cleve Moler 3-point one-sided estimate (left endpoint
    shown; right endpoint mirrors with h0=h[-1], h1=h[-2], m0=m[-1],
    m1=m[-2])::

        d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
        if sign(d) != sign(m0):
            d = 0                          # sign correction
        elif sign(m0) != sign(m1) and |d| > 3*|m0|:
            d = 3*m0                       # shape correction (FC 1980 §4)

    Edge cases:

    * `n < 2`: returns zeros of length n.
    * `n == 2`: linear — both slopes equal the single secant.
    * Same-x adjacent points: protected via `np.where(h==0, 1, h)` to
      avoid div-by-zero. Use `dedup_average` upstream as the canonical
      normalization step before passing to this function.
    * Zero or sign-flipped secants in interior: slope set to 0 (no
      overshoot, matches scipy's `condition` mask).
    """
    n = len(xs)
    if n < 2:
        return np.zeros(n)

    h = np.diff(xs)
    h_safe = np.where(h == 0, 1.0, h)
    m = np.diff(ys) / h_safe  # secants, length n-1

    if n == 2:
        # Linear fallback — both endpoints get the single secant slope.
        return np.array([m[0], m[0]])

    slopes = np.zeros(n)

    # Interior — weighted harmonic mean (scipy / Fritsch-Butland 1984).
    h_prev = h[:-1]  # h[0..n-3] — intervals before interior knots
    h_curr = h[1:]   # h[1..n-2] — intervals after interior knots
    m_prev = m[:-1]  # m[0..n-3] — secants before interior knots
    m_curr = m[1:]   # m[1..n-2] — secants after interior knots

    w1 = 2.0 * h_curr + h_prev
    w2 = h_curr + 2.0 * h_prev

    same_sign = (np.sign(m_prev) == np.sign(m_curr)) & (m_prev != 0) & (m_curr != 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        whmean = (w1 / m_prev + w2 / m_curr) / (w1 + w2)
        interior = np.where(same_sign, 1.0 / whmean, 0.0)

    slopes[1:-1] = interior

    # Endpoints — Cleve Moler 3-point one-sided estimate.
    slopes[0] = _three_point_endpoint(h[0], h[1], m[0], m[1])
    slopes[-1] = _three_point_endpoint(h[-1], h[-2], m[-1], m[-2])

    return slopes


def _three_point_endpoint(h0: float, h1: float, m0: float, m1: float) -> float:
    """One-sided 3-point endpoint slope with scipy's sign + shape correction.

    From Cleve Moler, *Numerical Computing with MATLAB* §3.6 (`pchiptx.m`),
    matching `scipy.interpolate._cubic.PchipInterpolator._edge_case`.
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    s_d = int(np.sign(d))
    s_m0 = int(np.sign(m0))
    s_m1 = int(np.sign(m1))
    if s_d != s_m0:
        # Sign correction — derivative would overshoot opposite the secant.
        return 0.0
    if s_m0 != s_m1 and abs(d) > 3.0 * abs(m0):
        # Shape correction — clamp to 3*m0 per Fritsch-Carlson 1980 §4.
        return 3.0 * m0
    return float(d)


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
