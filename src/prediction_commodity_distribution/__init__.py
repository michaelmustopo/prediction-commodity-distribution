"""prediction-commodity-distribution — pure-numpy CDF math for prediction markets.

The math layer that powers goldprice.dev's `/v1/prediction-market-view`
endpoint (and, by extension, the cone chart hero) under the Tidore
umbrella. Generic across any prediction-market touch-CDF input shape;
operates on abstract `(x, y, weight)` tuples — no schema coupling, no
DB, no I/O.

**Math attribution**: Pool-Adjacent-Violators (PAV) isotonic regression
+ Fritsch-Butland 1984 weighted-harmonic-mean Hermite cubic
(scipy.PchipInterpolator's exact algorithm) + Cleve Moler 3-point
endpoint formula (Fritsch-Carlson 1980 §4 shape correction) + linear
tail extrapolation. Output is byte-equivalent to
`scipy.interpolate.PchipInterpolator` for any quant verifier — locked
by `tests/test_scipy_parity.py`.

**License**: Apache 2.0.

**Public surface**:

    from prediction_commodity_distribution import (
        pool_adjacent_violators,
        weighted_pool_adjacent_violators,
        fritsch_carlson_slopes,
        hermite_eval,
        invert_percentile,
        invert_decreasing,
        dedup_average,
    )

Or via submodules:

    from prediction_commodity_distribution.isotonic import pool_adjacent_violators
    from prediction_commodity_distribution.hermite import fritsch_carlson_slopes
    from prediction_commodity_distribution.invert import invert_percentile
    from prediction_commodity_distribution.dedup import dedup_average
"""

from __future__ import annotations

from .dedup import dedup_average
from .hermite import fritsch_carlson_slopes, hermite_eval
from .invert import invert_decreasing, invert_percentile
from .isotonic import pool_adjacent_violators, weighted_pool_adjacent_violators

__version__ = "0.1.1"

__all__ = [
    "dedup_average",
    "fritsch_carlson_slopes",
    "hermite_eval",
    "invert_decreasing",
    "invert_percentile",
    "pool_adjacent_violators",
    "weighted_pool_adjacent_violators",
]
