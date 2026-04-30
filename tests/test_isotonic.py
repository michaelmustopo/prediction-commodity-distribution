"""Tests for `prediction_commodity_distribution.isotonic` — PAV (weighted + unweighted)."""

from __future__ import annotations

from prediction_commodity_distribution import (
    pool_adjacent_violators,
    weighted_pool_adjacent_violators,
)


def test_pav_forces_monotone_on_anti_monotone_input() -> None:
    points = [(10.0, 0.9), (20.0, 0.7), (30.0, 0.5), (40.0, 0.3)]
    out = pool_adjacent_violators(points)
    # Input is already strictly decreasing (violates "non-decreasing y").
    # PAV should collapse everything to a single average block.
    ys = [p[1] for p in out]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))
    assert len(set(ys)) <= len(points)


def test_pav_preserves_already_monotone_input() -> None:
    points = [(10.0, 0.1), (20.0, 0.2), (30.0, 0.5), (40.0, 0.9)]
    out = pool_adjacent_violators(points)
    assert out == points


def test_pav_smooths_local_inversion() -> None:
    """Real Kalshi shape: small inversions at otherwise monotone points."""
    points = [(10.0, 0.1), (20.0, 0.3), (25.0, 0.2), (30.0, 0.5)]
    out = pool_adjacent_violators(points)
    ys = [p[1] for p in out]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))


def test_pav_empty_input_returns_empty() -> None:
    assert pool_adjacent_violators([]) == []


def test_pav_single_point_passes_through() -> None:
    assert pool_adjacent_violators([(50.0, 0.42)]) == [(50.0, 0.42)]


def test_weighted_pav_pulls_toward_high_weight_on_violation() -> None:
    """When two adjacent points violate decreasing-monotonicity, the
    pooled value is the WEIGHTED mean — high-weight point dominates.
    """
    # x=10 has y=0.40 with weight 100; x=20 has y=0.50 with weight 1.
    # 0.50 > 0.40 violates DECREASING. Pooled mean weighted ≈ 0.401.
    pooled = weighted_pool_adjacent_violators([(10.0, 0.40, 100.0), (20.0, 0.50, 1.0)])
    assert len(pooled) == 2
    # Both x's get the SAME pooled y after merging.
    assert pooled[0][1] == pooled[1][1]
    # Pooled y closer to 0.40 (heavy) than to 0.50 (light).
    assert 0.40 < pooled[0][1] < 0.42


def test_weighted_pav_passes_through_already_decreasing_input() -> None:
    points = [(10.0, 0.9, 1.0), (20.0, 0.7, 1.0), (30.0, 0.5, 1.0), (40.0, 0.3, 1.0)]
    out = weighted_pool_adjacent_violators(points)
    assert [(x, y) for x, y, _ in points] == out


def test_weighted_pav_empty_input_returns_empty() -> None:
    assert weighted_pool_adjacent_violators([]) == []
