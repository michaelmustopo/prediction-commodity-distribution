"""Dedup-average for x-collisions on (x, y) curves."""

from __future__ import annotations

from collections import defaultdict


def dedup_average(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse same-x collisions via average(y). Keeps x values sorted ascending.

    When two or more market touch-points land at exactly the same strike
    after PAV pooling or upstream filtering, this collapses them to a
    single (x, mean_y) pair. Used as a normalization step before cubic
    interpolation, which expects distinct x values.

    Empty input returns []. Single-point input returns unchanged. The
    output is sorted by x ascending regardless of input order.
    """
    if not points:
        return []
    buckets: dict[float, list[float]] = defaultdict(list)
    for x, y in points:
        buckets[x].append(y)
    return sorted((x, sum(ys) / len(ys)) for x, ys in buckets.items())
