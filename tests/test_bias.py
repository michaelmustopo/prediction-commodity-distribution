"""Regression test for the V1 money-centroid bias.

This file exists for one reason: to let any reader of the README run

    pytest tests/test_bias.py -v

and watch the V1 estimator produce the same hundreds-of-dollars-off
result described in the bug story.

The fixture below is a synthetic Polymarket-style "above-X" gold ladder
calibrated to match the May 29 2026 horizon — the case that prompted
the rewrite. Real captured fixtures from the production scrape would
work too, but synthetic is more honest: anyone can read the data, see
why the bug exists, and verify the math without understanding our
ingest schema.

We pin THREE things:

1. The synthetic ladder, plugged into V1 (money-centroid), produces a
   p50 well below the synthetic spot — the bias.
2. The same ladder, plugged into V2 (direction-aware F mapping using
   this library's PAV + Hermite primitives), produces a p50 close to
   spot — sanity restored.
3. The numerical magnitude of the V1 bias is at least the size we
   reported in the README ($600+), so the README claim stays accurate
   if anyone tweaks the fixture.
"""

from __future__ import annotations

import numpy as np

from prediction_commodity_distribution.bias_reproduction import Market, money_centroid_p50
from prediction_commodity_distribution.dedup import dedup_average
from prediction_commodity_distribution.hermite import (
    fritsch_carlson_slopes,
    hermite_eval,
)
from prediction_commodity_distribution.invert import invert_decreasing
from prediction_commodity_distribution.isotonic import pool_adjacent_violators


# Synthetic May-29-2026 horizon. Spot = $4580 (close to the actual May 4
# observation). Strike ladder modeled on Polymarket's "Will gold be
# above $X by May 29?" book: many low-strike tautologies (yes_price near
# 1.0), thin coverage in the middle, sparse tail. liquidity_usd is the
# reported notional for each market.
SYNTHETIC_SPOT_USD = 4580.0

SYNTHETIC_LADDER: tuple[Market, ...] = (
    # Low-strike tautologies: gold is already at $4580, so "above $X"
    # for X << 4580 trades at near-1.0. These are the markets that
    # silently dominate the V1 weighted mean.
    Market("above", 1000.0, 0.995, 50_000.0),
    Market("above", 2000.0, 0.992, 80_000.0),
    Market("above", 3000.0, 0.985, 120_000.0),
    Market("above", 3500.0, 0.965, 90_000.0),
    Market("above", 4000.0, 0.880, 200_000.0),
    # Mid-range: market is genuinely uncertain near spot
    Market("above", 4400.0, 0.620, 350_000.0),
    Market("above", 4500.0, 0.520, 400_000.0),
    Market("above", 4600.0, 0.420, 380_000.0),
    Market("above", 4700.0, 0.330, 250_000.0),
    Market("above", 4800.0, 0.230, 180_000.0),
    # Upper tail: thin liquidity, low yes_price
    Market("above", 5000.0, 0.110, 90_000.0),
    Market("above", 5500.0, 0.040, 40_000.0),
    Market("above", 6000.0, 0.018, 20_000.0),
)


def test_v1_money_centroid_underestimates_versus_spot() -> None:
    """V1 money-centroid p50 should land hundreds of dollars below
    spot — the bug. Pinning the bug here so any "refactor" of
    bias_reproduction.py that accidentally fixes the math fails this
    test loudly.
    """
    p50_v1 = money_centroid_p50(SYNTHETIC_LADDER)
    drift = SYNTHETIC_SPOT_USD - p50_v1
    assert drift > 600.0, (
        f"Expected V1 estimator to under-shoot spot by >$600 (the "
        f"reproducible-bug threshold the README claims). Got drift = "
        f"${drift:.2f}, V1 p50 = ${p50_v1:.2f} vs spot = "
        f"${SYNTHETIC_SPOT_USD:.2f}."
    )


def test_v1_money_centroid_drift_is_at_least_readme_magnitude() -> None:
    """The README claims '$628 below spot' as the actual reproduced
    drift on the production May 29 case. Synthetic data should be
    calibrated so the same magnitude or larger holds — otherwise the
    README claim isn't backed by a runnable test.
    """
    p50_v1 = money_centroid_p50(SYNTHETIC_LADDER)
    drift = SYNTHETIC_SPOT_USD - p50_v1
    assert drift >= 600.0, (
        f"V1 drift below the README's reported $628. Got ${drift:.2f}."
    )


def _v2_direction_aware_p50(
    markets: tuple[Market, ...], spot_usd: float
) -> float:
    """V2 direction-aware estimator built from this library's primitives.

    Steps:

    1. Map each above-X market to a CDF point: F(strike) = 1 - yes_price.
       (For an above-X market, yes_price = P(price > strike), so the
       CDF F(strike) = P(price <= strike) = 1 - yes_price.)
    2. Sort by strike, dedup average y at duplicate strikes.
    3. Enforce monotonicity (PAV) over the F values.
    4. Hermite-cubic interpolation between PAV plateaus.
    5. Invert at F = 0.5 to recover the median strike.

    The lib's `weighted_pool_adjacent_violators` is wired for the touch-
    CDF representation (monotone-decreasing y). We use the unweighted
    `pool_adjacent_violators` (monotone-increasing y) since F = 1 -
    yes_price is non-decreasing in strike for above-X markets — the
    bias fix doesn't depend on liquidity weighting at this layer.
    """
    above_only = [m for m in markets if m.direction == "above"]
    if not above_only:
        raise ValueError("synthetic ladder must contain above-X markets")

    above_only.sort(key=lambda m: m.strike)

    # 1. Map each market to a (strike, F) point. F(strike) = 1 - yp.
    pts = [(m.strike, 1.0 - m.yes_price) for m in above_only]

    # 2. Dedup duplicate strikes (defensive — none in this fixture).
    pts = dedup_average(pts)

    # 3. PAV → monotone non-decreasing F.
    pts = pool_adjacent_violators(pts)

    # Split into parallel arrays for the cubic step.
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # 4. Fit Fritsch-Carlson Hermite slopes.
    slopes = fritsch_carlson_slopes(xs, ys)

    # 5. Invert at F = 0.5 by working on the decreasing complement.
    #    `invert_decreasing` returns (x, in_range) for monotone-
    #    decreasing curves, so feed it (xs, 1-F, -slopes) and target 0.5.
    ys_complement = 1.0 - ys
    slopes_complement = -slopes
    p50_strike, _in_range = invert_decreasing(
        xs, ys_complement, slopes_complement, 0.5
    )
    return float(p50_strike)


def test_v2_direction_aware_lands_near_spot() -> None:
    """V2 direction-aware estimator on the same ladder should produce
    a p50 within the synthetic market's noise floor of spot — call it
    ±$300, generous enough to absorb the deliberate yes_price kinks in
    the fixture without being so loose it would pass for V1 too.
    """
    p50_v2 = _v2_direction_aware_p50(SYNTHETIC_LADDER, SYNTHETIC_SPOT_USD)
    drift = abs(p50_v2 - SYNTHETIC_SPOT_USD)
    assert drift < 300.0, (
        f"V2 direction-aware p50 drifted ${drift:.2f} from spot — "
        f"got ${p50_v2:.2f} vs spot ${SYNTHETIC_SPOT_USD:.2f}. The "
        f"fix should land much closer than this."
    )


def test_v2_strictly_better_than_v1() -> None:
    """The whole point of the rewrite. V2 should be at least 5x closer
    to spot than V1 on this fixture — otherwise the lib's claim that
    direction-aware mapping fixes the bug is unsupported.
    """
    p50_v1 = money_centroid_p50(SYNTHETIC_LADDER)
    p50_v2 = _v2_direction_aware_p50(SYNTHETIC_LADDER, SYNTHETIC_SPOT_USD)
    drift_v1 = abs(p50_v1 - SYNTHETIC_SPOT_USD)
    drift_v2 = abs(p50_v2 - SYNTHETIC_SPOT_USD)
    assert drift_v2 * 5 < drift_v1, (
        f"V2 should be at least 5x closer to spot than V1. Got "
        f"V1 drift = ${drift_v1:.2f}, V2 drift = ${drift_v2:.2f}."
    )
