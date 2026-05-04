"""V1 (broken) money-centroid implementation, kept as a frozen module
for reproducibility of the bias that prompted the rewrite.

This module is NOT part of the public API. Do not import it from
``__init__``. It exists so anyone reading the README's bug story can run
``pytest tests/test_bias.py`` and see the bias in action — the V1 math
applied to the same input shape the production direction-aware path
consumes, producing a p50 hundreds of dollars off both spot and
futures.

Why keep V1 in the repo at all?
-------------------------------
1. Reproducibility. The README claim ("caught a $628 mistake") is only
   credible if any reader can clone, install, and reproduce.
2. Regression guard. ``test_bias.py`` asserts the V1 code STILL produces
   the bias — if anyone refactors this module to "fix" it later, the
   test fails loudly. This is the inverse of a normal regression test:
   we are pinning the broken behavior so the "before" half of the story
   stays demonstrably-true.
3. Documentation. The math is explained in code comments where it lives
   rather than buried in commit history.

Bug summary
-----------
The V1 estimator treats each prediction-market row as a vote for its
strike, weighted by ``yes_price * liquidity_usd``. For an above-X market
chain — most of Polymarket's gold book — this concentrates weight at
LOW strikes where ``yes_price`` is near 1.0 because the market is
asking a near-tautology ("will gold be above $1,000 by May?" trades at
~99¢ when spot is $4,580). Those high yes-prices dominate the weighted
mean and drag the estimated p50 hundreds of dollars below where the
underlying actually is.

The fix (in ``cdf_view`` / ``isotonic`` / ``hermite``) treats each
market as a CDF point, not a vote: F(X) = 1 - yes_price for above-X,
yes_price for below-X. See README "Why this matters" for the full
direction-aware F mapping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """One prediction-market row.

    Attributes:
        direction: "above" | "below" | "touch_upside" | "touch_downside"
            | "range". Determines how yes_price relates to strike. The
            V1 estimator below ignores this field — that is the bug.
        strike: USD strike of the market (e.g. 4500.0 for "above $4500").
        yes_price: Latest yes-side trade price, in [0, 1].
        liquidity_usd: Reported liquidity in USD. Used as a confidence
            weight by both V1 and V2.
    """

    direction: str
    strike: float
    yes_price: float
    liquidity_usd: float


def money_centroid_p50(markets: Iterable[Market]) -> float:
    """V1 (broken) point-estimator.

    Treats ``yes_price * liquidity_usd`` as a money weight on each
    market's strike, returns the weighted mean strike. Reads intuitive
    ("strikes that have lots of conviction-money are where the market
    thinks gold will be"); fundamentally misuses ``yes_price``, which
    is a probability, not a money weight.

    See module docstring for the full explanation. Test
    ``test_bias.py::test_money_centroid_p50_under_spot`` reproduces
    the May 29 case where this returned $3,952 against a spot of
    $4,580.

    Returns:
        The weighted mean strike, in USD. NaN-equivalent: returns 0.0
        if all weights collapse (no eligible rows). Production-V1 also
        emitted 0.0 in this case; preserving the behavior so the test
        can pin it.
    """
    rows = list(markets)
    if not rows:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for m in rows:
        weight = m.yes_price * m.liquidity_usd
        weighted_sum += m.strike * weight
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight
