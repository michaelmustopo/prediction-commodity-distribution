"""scipy.PchipInterpolator parity — locks the OSS lib's byte-equivalence claim.

Run with: ``pip install -e .[dev]`` (scipy is a dev/optional dep, not runtime).

These tests are the load-bearing reproducibility hook from pm-27 ADR. If
they regress, the README + integration docstring claims of "byte-equivalent
to scipy.PchipInterpolator" become false — fix the math, not the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from prediction_commodity_distribution import (
    fritsch_carlson_slopes,
    hermite_eval,
)

scipy = pytest.importorskip("scipy.interpolate")


def _scipy_slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Slopes that scipy.PchipInterpolator would assign at each knot."""
    return scipy.PchipInterpolator(xs, ys).derivative()(xs)


def test_scipy_parity_uniform_spacing() -> None:
    """5-knot uniform spacing — interior + endpoints byte-equivalent."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 0.5, 0.7, 0.85, 1.0])
    ours = fritsch_carlson_slopes(xs, ys)
    theirs = _scipy_slopes(xs, ys)
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=0)


def test_scipy_parity_non_uniform_spacing() -> None:
    """Non-uniform spacing — locks weighted harmonic mean (simple harmonic
    would diverge here)."""
    xs = np.array([0.0, 0.3, 1.5, 2.1, 5.0])
    ys = np.array([0.0, 0.2, 0.4, 0.55, 1.0])
    ours = fritsch_carlson_slopes(xs, ys)
    theirs = _scipy_slopes(xs, ys)
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=0)


def test_scipy_parity_realistic_gold_cdf() -> None:
    """Realistic gold-CDF input: 8 strikes between $4500-$5500, monotone-isotonized
    probabilities. This is the test CC's audit used to surface the ~1.06% endpoint
    divergence under the v0.1.0 linear-secant endpoint formula. Locks the fix."""
    xs = np.array([4500.0, 4600.0, 4750.0, 4900.0, 5000.0, 5150.0, 5300.0, 5500.0])
    ys = np.array([0.05, 0.12, 0.25, 0.42, 0.55, 0.72, 0.85, 0.95])
    ours = fritsch_carlson_slopes(xs, ys)
    theirs = _scipy_slopes(xs, ys)
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=0)


def test_scipy_parity_curve_evaluation_matches_scipy() -> None:
    """Curve values match scipy at 100 sample points spanning the knot range."""
    xs = np.array([4500.0, 4600.0, 4750.0, 4900.0, 5000.0, 5150.0, 5300.0, 5500.0])
    ys = np.array([0.05, 0.12, 0.25, 0.42, 0.55, 0.72, 0.85, 0.95])
    slopes = fritsch_carlson_slopes(xs, ys)
    interp = scipy.PchipInterpolator(xs, ys)

    samples = np.linspace(xs[0], xs[-1], 100)
    ours = np.array([hermite_eval(xs, ys, slopes, float(x)) for x in samples])
    theirs = interp(samples)
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=0)


def test_scipy_parity_two_knot_linear_fallback() -> None:
    """n == 2 — both endpoints get the single secant slope (matches scipy)."""
    xs = np.array([0.0, 1.0])
    ys = np.array([0.2, 0.8])
    ours = fritsch_carlson_slopes(xs, ys)
    theirs = _scipy_slopes(xs, ys)
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=0)


def test_scipy_parity_flat_curve() -> None:
    """All-equal ys → all slopes 0 in both implementations."""
    xs = np.array([100.0, 200.0, 300.0, 400.0])
    ys = np.array([0.5, 0.5, 0.5, 0.5])
    ours = fritsch_carlson_slopes(xs, ys)
    theirs = _scipy_slopes(xs, ys)
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=0)
