# prediction-commodity-distribution

Pure-numpy CDF math for prediction-market touch curves. Powers the cone chart at [goldprice.dev](https://goldprice.dev) and (eventually) every commodity vertical on the [Tidore](https://tidore.co) umbrella.

This library is what runs in production. There is no internal fork.

## What it does

Given a list of prediction-market touch points — e.g. Polymarket / Kalshi questions like "Will gold touch $5,000 before Dec 31?" with associated `yes_price` + liquidity — this library produces a smooth monotone CDF and inverts it to extract percentile bands (p10/p25/p50/p75/p90).

The math:

1. **Pool-Adjacent-Violators (PAV) isotonic regression** — enforces monotonicity on noisy market-implied probabilities. Weighted variant pools toward high-conviction (high `yes_price × liquidity`) strikes.
2. **Fritsch-Butland 1984 weighted-harmonic-mean Hermite cubic** — `scipy.PchipInterpolator`'s exact algorithm. Smooths the isotonic step function into a continuously-differentiable curve.
3. **Cleve Moler 3-point endpoint formula** with Fritsch-Carlson 1980 §4 sign + shape correction — handles the boundary slopes scipy uses.
4. **Linear-tail extrapolation + cubic-bisection inversion** — finds strikes at target probability levels, flagging which are inside the observed range vs extrapolated.

Output is **byte-equivalent** to `scipy.interpolate.PchipInterpolator` for any quant who wants to verify the cone against a reference. Locked by [`tests/test_scipy_parity.py`](tests/test_scipy_parity.py) — interior slopes, endpoint slopes, and curve evaluation all match within `1e-12` on uniform spacing, non-uniform spacing, and realistic gold-CDF input.

## Install

```bash
pip install prediction-commodity-distribution
```

## Usage

```python
import numpy as np
from prediction_commodity_distribution import (
    pool_adjacent_violators,
    weighted_pool_adjacent_violators,
    fritsch_carlson_slopes,
    hermite_eval,
    invert_percentile,
    invert_decreasing,
    dedup_average,
)

# 1. Start with raw market touch points: list of (strike, yes_price)
raw = [
    (4500.0, 0.85),
    (4750.0, 0.65),
    (5000.0, 0.45),
    (5250.0, 0.30),
    (5500.0, 0.18),
]

# 2. Average any same-strike collisions, then enforce monotonicity.
points = pool_adjacent_violators(dedup_average(raw))

# 3. Build the smooth Hermite cubic.
xs = np.array([x for x, _ in points])
ys = np.array([y for _, y in points])
slopes = fritsch_carlson_slopes(xs, ys)

# 4. Evaluate the curve at any strike inside the observed range.
prob_at_4900 = hermite_eval(xs, ys, slopes, 4900.0)

# 5. Invert to find strikes at target probability levels.
#    invert_percentile  → for monotone-increasing CDF (settlement-style)
#    invert_decreasing  → for monotone-decreasing touch probability
strike_p25, was_extrapolated = invert_decreasing(xs, ys, slopes, 0.25)
```

For weighted-PAV (conviction-weighted, where strikes with more capital pull harder during pooling):

```python
weighted_points = [
    (4500.0, 0.85, 12_000.0),  # (strike, yes_price, conviction_weight)
    (4750.0, 0.65, 25_000.0),
    (5000.0, 0.45, 8_000.0),
]
pooled = weighted_pool_adjacent_violators(weighted_points)
```

## Public API

| Function | Module | Purpose |
|---|---|---|
| `pool_adjacent_violators(points)` | `isotonic` | Enforce non-decreasing y on (x, y) tuples |
| `weighted_pool_adjacent_violators(points)` | `isotonic` | Enforce decreasing-monotone on (x, y, w); high-w dominates pooling |
| `fritsch_carlson_slopes(xs, ys)` | `hermite` | Hermite tangent slopes — scipy.PchipInterpolator's exact algorithm (Fritsch-Butland 1984 weighted harmonic mean + Cleve Moler 3-point endpoints with sign/shape correction) |
| `hermite_eval(xs, ys, slopes, x)` | `hermite` | Evaluate cubic at x (caller pre-brackets) |
| `invert_percentile(xs, ys, slopes, target)` | `invert` | Find x for f(x) = target on monotone-increasing curve |
| `invert_decreasing(xs, ys, slopes, target)` | `invert` | Same on monotone-decreasing curve |
| `dedup_average(points)` | `dedup` | Collapse same-x collisions via average(y) |

All accept Python list / numpy array inputs as documented in each module's docstring. Output is `(x, y)` tuples or numpy arrays depending on the function.

## Math attribution

- **Pool-Adjacent-Violators**: Brunk 1955; Ayer et al. 1955. Canonical O(n) isotonic regression.
- **Fritsch-Carlson 1980** (SIAM J. Numer. Anal. 17(2):238-246): the broader monotone-Hermite framework PCHIP fits under, and the source of the §4 sign + shape correction applied to endpoint slopes.
- **Fritsch-Butland 1984**: weighted harmonic-mean slope choice for interior knots. The canonical PCHIP variant scipy adopted.
- **Cleve Moler** (*Numerical Computing with MATLAB* §3.6, `pchiptx.m`): one-sided 3-point endpoint formula scipy uses.
- **scipy parity**: this library produces output **byte-equivalent** to `scipy.interpolate.PchipInterpolator` on the same input. Locked by `tests/test_scipy_parity.py`.

We follow scipy's exact algorithm (weighted harmonic mean for interior + 3-point endpoints) rather than a simpler arithmetic mean or linear-secant fallback because (a) it removes "why does our cone differ from PCHIP?" friction with quant readers, and (b) the weighted variant produces more conservative slopes on non-uniform knot spacing (less overshoot risk on steep CDF transitions).

## Where this is used

- **goldprice.dev** — the cone chart hero on `/data/gold` and the `/v1/prediction-market-view` endpoint.
- **Tidore (umbrella, in progress)** — same library will power silverprice.dev / copperprice.dev / oilprice.dev / etc as those verticals ship. First commodity (gold) paid the math cost; subsequent ones inherit it for free.

## Scope discipline

This library is intentionally narrow. It contains:

- ✅ Pure-functional CDF math operating on abstract tuples
- ✅ No I/O, no DB, no network calls
- ✅ Single dependency: `numpy`

It does NOT contain:

- ❌ Polymarket / Kalshi specific scrapers (those live in production code)
- ❌ Database schemas or ORM models
- ❌ Pricing data sources or spot-price fetchers
- ❌ HTTP clients, REST handlers, or routing
- ❌ Confidence labelling business logic

If you need any of the above, look at the integration code in [goldprice-dev](https://github.com/michaelmustopo/goldprice-dev) or build your own data layer around this math.

## Development

```bash
git clone https://github.com/michaelmustopo/prediction-commodity-distribution.git
cd prediction-commodity-distribution
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
pyright
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contributing

Issues + PRs welcome. Scope is bounded (math only — see "Scope discipline" above); contributions that add integrations or non-math features will be politely declined to keep maintenance overhead bounded.

For substantive math additions or alternative slope-choice strategies (Akima 1970, Steffen 1990, etc), open an issue first to discuss before sending a PR.
