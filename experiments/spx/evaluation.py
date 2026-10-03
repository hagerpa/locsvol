"""Evaluation of the calibrated models (Section 3.4 of the paper).

Every model is priced on independent sets of paths, the replications, at the quotes and at a
comparison grid of 81 equally spaced log strikes per quoted maturity, between the smallest and the
largest quoted one. From the prices pooled over the replications:

- the implied volatilities on the comparison grid are compared with the target: a point belongs
  to the left wing if k <= 0.4 k_min(T), to the right wing if k >= 0.4 k_max(T) and to the central
  region otherwise; within each region every maturity has the same weight, and the root mean
  squared difference is reported in volatility basis points (bp);
- the bid-ask loss (sum_q w_q dist(p_q, [bid_q, ask_q])^2 / v_q^2)^(1/2) in bp, over all quotes
  with their weights and over the liquid quotes with theirs;
- the shares of liquid quotes inside their bid-ask interval, outside by less than two standard
  errors of the pooled price (within noise), and outside by more.

Standard errors come from the paired delete-one jackknife over the replications: each statistic
is recomputed from the prices pooled over all replications but one, deleting the same replication
for every model.
"""

import jax.numpy as jnp
import numpy as np
from settings import COMPARISON_STRIKES, WING

from locsvol.calibration.gauss_newton import interval_residuals
from locsvol.pricing import implied_volatility


def comparison_grid(quotes):
    """Maturities and log strikes of the comparison grid."""
    t, k = np.asarray(quotes.maturities), np.asarray(quotes.log_strikes)
    maturities, strikes = [], []
    for maturity in np.unique(t):
        rows = k[t == maturity]
        maturities.extend([maturity] * COMPARISON_STRIKES)
        strikes.extend(np.linspace(rows.min(), rows.max(), COMPARISON_STRIKES))
    return jnp.asarray(maturities), jnp.asarray(strikes)


def region_weights(maturities, log_strikes):
    """Weights (3, points) of the left wing, the central region and the right wing."""
    t, k = np.asarray(maturities), np.asarray(log_strikes)
    masks = np.zeros((3, k.size), dtype=bool)
    for maturity in np.unique(t):
        rows = t == maturity
        left, right = min(float(k[rows].min()), 0.0), max(float(k[rows].max()), 0.0)
        masks[0] |= rows & (k < 0) & (-k >= WING * -left)
        masks[2] |= rows & (k > 0) & (k >= WING * right)
    masks[1] = ~(masks[0] | masks[2])
    weights = np.zeros(masks.shape)
    for index, mask in enumerate(masks):
        maturities_in_region = np.unique(t[mask])
        for maturity in maturities_in_region:
            rows = mask & (t == maturity)
            weights[index, rows] = 1 / (maturities_in_region.size * rows.sum())
    return jnp.asarray(weights)


class Comparison:
    """The quotes, the comparison grid and the target's implied volatilities on it."""

    def __init__(self, quotes, surface):
        self.quotes = quotes
        self.count = quotes.maturities.size
        self.grid = comparison_grid(quotes)
        self.target = surface.implied_volatility(*self.grid)
        self.regions = region_weights(*self.grid)
        self.maturities = jnp.concatenate((quotes.maturities, self.grid[0]))
        self.log_strikes = jnp.concatenate((quotes.log_strikes, self.grid[1]))
        self.is_call = jnp.ones(self.maturities.size, dtype=bool)

    def statistics(self, prices):
        """Left, central and right RMSE and the two bid-ask losses, in bp; prices (..., all)."""
        count, quotes = self.count, self.quotes
        sigma = implied_volatility(prices[..., count:], *self.grid, self.is_call[count:])
        error = sigma - self.target
        regions = jnp.stack(
            [jnp.sqrt(jnp.sum(jnp.where(w > 0, error**2 * w, 0), axis=-1)) for w in self.regions],
            axis=-1,
        )
        losses = [
            jnp.linalg.norm(
                interval_residuals(prices[..., :count], quotes.bid, quotes.ask, w, quotes.vega),
                axis=-1,
            )[..., None]
            for w in (quotes.weights, quotes.liquid_weights)
        ]
        return 1e4 * jnp.concatenate((regions, *losses), axis=-1)

    def positions(self, replication_prices):
        """Shares of liquid quotes inside, within noise of and clearly outside their intervals."""
        prices = jnp.asarray(replication_prices)[..., : self.count]
        pooled = jnp.mean(prices, axis=0)
        error = jnp.std(prices, axis=0, ddof=1) / jnp.sqrt(prices.shape[0])
        quotes = self.quotes
        gap = jnp.maximum(quotes.bid - pooled, 0.0) + jnp.maximum(pooled - quotes.ask, 0.0)
        inside = quotes.liquid & (gap == 0)
        clear = quotes.liquid & (gap > 2 * error)
        near = quotes.liquid & ~inside & ~clear
        shares = [jnp.sum(mask, axis=-1) for mask in (inside, near, clear)]
        return jnp.stack(shares, axis=-1) / jnp.sum(quotes.liquid)


def jackknife(replication_prices, statistic):
    """The statistic of the pooled prices and its paired delete-one jackknife standard error.

    replication_prices has the replications on its first axis.
    """
    prices = jnp.asarray(replication_prices)
    count = prices.shape[0]
    pooled = jnp.mean(prices, axis=0)
    deleted = (jnp.sum(prices, axis=0) - prices) / (count - 1)
    estimate = statistic(pooled)
    values = jnp.stack([statistic(row) for row in deleted])
    spread = jnp.sum((values - jnp.mean(values, axis=0)) ** 2, axis=0)
    return estimate, jnp.sqrt((count - 1) / count * spread), values
