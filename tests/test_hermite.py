"""Tests for `prediction_commodity_distribution.hermite` — slopes + cubic eval."""

from __future__ import annotations

import numpy as np
import pytest

from prediction_commodity_distribution import (
    fritsch_carlson_slopes,
    hermite_eval,
)


def test_hermite_monotone_within_observed_range() -> None:
    xs = np.array([100.0, 200.0, 300.0, 400.0])
    ys = np.array([0.1, 0.3, 0.6, 0.9])
    slopes = fritsch_carlson_slopes(xs, ys)
    # Monotone at 11 interior probe points
    probes = [100.0 + i * 30.0 for i in range(11)]
    evaluated = [hermite_eval(xs, ys, slopes, p) for p in probes]
    assert all(evaluated[i] <= evaluated[i + 1] + 1e-9 for i in range(len(evaluated) - 1))


def test_fritsch_carlson_slopes_uses_weighted_harmonic_mean() -> None:
    """Pin the Fritsch-Butland 1984 weighted-harmonic-mean interior formula
    + Cleve Moler 3-point endpoints.

    Reference example: xs=[0,1,2], ys=[1,3,4].
      d[0] = (3-1)/1 = 2
      d[1] = (4-3)/1 = 1

    Interior (uniform spacing collapses weighted → simple harmonic):
      slopes[1] = 2*d[0]*d[1] / (d[0]+d[1]) = 4/3 ≈ 1.333

    Left endpoint (Cleve Moler 3-point):
      d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
        = ((2 + 1)*2 - 1*1) / 2 = 5/2 = 2.5
      sign(d) == sign(m0), same-sign secants → no correction.

    Right endpoint (mirror, h0=h[-1]=1, h1=h[-2]=1, m0=m[-1]=1, m1=m[-2]=2):
      d = ((2 + 1)*1 - 1*2) / 2 = 1/2 = 0.5
      sign(d) == sign(m0), same-sign secants → no correction.

    An arithmetic-mean interior `(d[i-1] + d[i]) / 2` would yield 1.5
    instead of 4/3 — diverges ~12% from scipy on this input.
    """
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([1.0, 3.0, 4.0])
    slopes = fritsch_carlson_slopes(xs, ys)

    # Endpoints — Cleve Moler 3-point one-sided estimate.
    assert slopes[0] == pytest.approx(2.5, abs=1e-12)
    assert slopes[2] == pytest.approx(0.5, abs=1e-12)
    # Interior — weighted harmonic mean (uniform spacing → simple harmonic).
    assert slopes[1] == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert slopes[1] != pytest.approx(1.5, abs=1e-6)


def test_fritsch_carlson_slopes_flat_curve_zero_slopes() -> None:
    """Flat input (all ys equal) → all slopes are exactly 0.

    Catches a regression where the slope formula divides by zero or
    propagates NaN when all secant slopes (`d`) are 0. The limiter
    zeroes both adjacent slopes when `d[i] == 0`, so the whole curve
    stays flat. Important because mature markets often have `yes_price`
    clustered around a consensus value — a flat curve is the realistic
    edge case, not a synthetic one.
    """
    xs = np.array([100.0, 200.0, 300.0, 400.0])
    ys = np.array([0.5, 0.5, 0.5, 0.5])
    slopes = fritsch_carlson_slopes(xs, ys)

    assert slopes.shape == (4,)
    assert not np.isnan(slopes).any(), "flat curve should not produce NaN slopes"
    assert (slopes == 0.0).all(), f"all slopes should be 0 on flat input, got {slopes!r}"


def test_fritsch_carlson_slopes_opposite_signs_zero_interior() -> None:
    """V-shape curve → interior slope at the apex is 0 (no overshoot)."""
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 0.0])  # V-shape: up then down
    slopes = fritsch_carlson_slopes(xs, ys)
    # d[0] = +1, d[1] = -1 → opposite sign → interior slope set to 0
    assert slopes[1] == 0.0


def test_fritsch_carlson_slopes_short_input_returns_zeros() -> None:
    """n < 2 → return zeros of correct shape."""
    xs = np.array([5.0])
    ys = np.array([0.5])
    slopes = fritsch_carlson_slopes(xs, ys)
    assert slopes.shape == (1,)
    assert slopes[0] == 0.0


def test_hermite_eval_at_knot_returns_knot_y() -> None:
    """At an exact knot, the cubic should evaluate to the knot's y."""
    xs = np.array([100.0, 200.0, 300.0])
    ys = np.array([0.2, 0.5, 0.8])
    slopes = fritsch_carlson_slopes(xs, ys)
    # Within float-precision tolerance because t = 0 is exact.
    assert hermite_eval(xs, ys, slopes, 200.0) == pytest.approx(0.5, abs=1e-9)
