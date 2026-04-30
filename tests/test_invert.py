"""Tests for `prediction_commodity_distribution.invert` — percentile inversion."""

from __future__ import annotations

import numpy as np

from prediction_commodity_distribution import (
    fritsch_carlson_slopes,
    invert_decreasing,
    invert_percentile,
)


def test_invert_percentile_interior_matches_known_point() -> None:
    xs = np.array([100.0, 200.0, 300.0, 400.0])
    ys = np.array([0.1, 0.3, 0.6, 0.9])
    slopes = fritsch_carlson_slopes(xs, ys)
    x, extrap = invert_percentile(xs, ys, slopes, 0.3)
    assert not extrap
    assert abs(x - 200.0) < 0.5


def test_invert_percentile_left_tail_extrapolates() -> None:
    xs = np.array([100.0, 200.0, 300.0])
    ys = np.array([0.2, 0.5, 0.8])
    slopes = fritsch_carlson_slopes(xs, ys)
    x, extrap = invert_percentile(xs, ys, slopes, 0.10)
    assert extrap  # 0.10 < ys[0]=0.2 → extrapolate
    assert x < 100.0  # below outermost observed strike


def test_invert_percentile_right_tail_extrapolates() -> None:
    xs = np.array([100.0, 200.0, 300.0])
    ys = np.array([0.2, 0.5, 0.8])
    slopes = fritsch_carlson_slopes(xs, ys)
    x, extrap = invert_percentile(xs, ys, slopes, 0.95)
    assert extrap
    assert x > 300.0


def test_invert_decreasing_finds_known_crossing() -> None:
    xs = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    ys = np.array([0.90, 0.70, 0.50, 0.30, 0.10])
    slopes = fritsch_carlson_slopes(xs, ys)
    # Exact crossing at x=200 → y=0.50
    x, extrap = invert_decreasing(xs, ys, slopes, 0.50)
    assert abs(x - 200.0) < 0.5
    assert extrap is False
    # Beyond left tail → extrapolated, x < xs[0]
    x_left, extrap_left = invert_decreasing(xs, ys, slopes, 0.99)
    assert extrap_left is True
    assert x_left < 0.0
    # Beyond right tail → extrapolated, x > xs[-1]
    x_right, extrap_right = invert_decreasing(xs, ys, slopes, 0.01)
    assert extrap_right is True
    assert x_right > 400.0
