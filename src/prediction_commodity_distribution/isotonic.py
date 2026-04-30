"""Pool-Adjacent-Violators (PAV) isotonic regression — weighted + unweighted.

Two flavors:

* `pool_adjacent_violators` — enforces NON-DECREASING y across already-
  x-sorted (x, y) points. Each point gets weight 1.

* `weighted_pool_adjacent_violators` — enforces DECREASING-monotonic y
  with per-point weights `w` in the (x, y, w) tuples. Pooled blocks
  pull toward the high-weight side.

Both run in O(n) and operate on Python list/tuple structures (no numpy
required at this layer — keeps the dependency surface minimal).
"""

from __future__ import annotations

from typing import Any


def pool_adjacent_violators(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Enforce non-decreasing y across already-x-sorted (x, y) points.

    Canonical O(n) PAV — when y[i+1] < y[i], merge the two adjacent
    blocks into one with the weighted average y. Repeat leftward until
    monotonicity is restored locally, then advance.
    """
    if len(points) <= 1:
        return list(points)

    # Each block: [x_list, y_sum, weight]
    blocks: list[list[Any]] = [[[x], y, 1] for x, y in points]
    i = 1
    while i < len(blocks):
        if blocks[i][1] / blocks[i][2] < blocks[i - 1][1] / blocks[i - 1][2]:
            merged_x = blocks[i - 1][0] + blocks[i][0]
            merged_y = blocks[i - 1][1] + blocks[i][1]
            merged_w = blocks[i - 1][2] + blocks[i][2]
            blocks[i - 1 : i + 1] = [[merged_x, merged_y, merged_w]]
            if i > 1:
                i -= 1
        else:
            i += 1

    out: list[tuple[float, float]] = []
    for xs, y_sum, w in blocks:
        avg_y = y_sum / w
        for x in xs:
            out.append((x, avg_y))
    return out


def weighted_pool_adjacent_violators(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    """Liquidity-weighted PAV. Input: (x, y, w) sorted by x ascending,
    expects monotone-DECREASING y for the touch-distance representation
    (greater distance from spot → lower touch probability).

    Each block tracks (x_list, weighted_y_sum, total_weight). When the
    next block's weighted-mean y exceeds the previous block's, merge
    leftward (pooling weighted means + summing weights). Returns
    (x, y_pooled) preserving the original x ordering.

    For the gold-CDF use case (per ADR pm-3 in the host project),
    weights are typically `yes_price * liquidity_usd` — strikes with
    more conviction-weighted capital pull harder during pooling.
    """
    if len(points) <= 1:
        return [(x, y) for x, y, _ in points]

    blocks: list[list[Any]] = [[[x], y * w, w] for x, y, w in points]
    i = 1
    while i < len(blocks):
        avg_curr = blocks[i][1] / blocks[i][2]
        avg_prev = blocks[i - 1][1] / blocks[i - 1][2]
        # Decreasing-monotone violation: current avg > previous avg.
        if avg_curr > avg_prev:
            merged_x = blocks[i - 1][0] + blocks[i][0]
            merged_y = blocks[i - 1][1] + blocks[i][1]
            merged_w = blocks[i - 1][2] + blocks[i][2]
            blocks[i - 1 : i + 1] = [[merged_x, merged_y, merged_w]]
            if i > 1:
                i -= 1
        else:
            i += 1

    out: list[tuple[float, float]] = []
    for xs, y_sum, w in blocks:
        avg_y = y_sum / w
        for x in xs:
            out.append((x, avg_y))
    return out
