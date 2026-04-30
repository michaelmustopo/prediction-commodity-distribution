"""Tests for `prediction_commodity_distribution.dedup`."""

from __future__ import annotations

from prediction_commodity_distribution import dedup_average


def test_dedup_average_collapses_same_x() -> None:
    points = [(100.0, 0.3), (100.0, 0.5), (200.0, 0.7)]
    out = dedup_average(points)
    assert out == [(100.0, 0.4), (200.0, 0.7)]


def test_dedup_average_empty_input_returns_empty() -> None:
    assert dedup_average([]) == []


def test_dedup_average_single_point_passes_through() -> None:
    assert dedup_average([(50.0, 0.42)]) == [(50.0, 0.42)]


def test_dedup_average_sorts_x_ascending() -> None:
    points = [(300.0, 0.7), (100.0, 0.2), (200.0, 0.4)]
    out = dedup_average(points)
    xs = [x for x, _ in out]
    assert xs == sorted(xs)
