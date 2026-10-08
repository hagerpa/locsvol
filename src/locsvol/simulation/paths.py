"""Simulated paths of the variance factor, with the increments that drive the spot."""

from dataclasses import dataclass

import jax

from locsvol.simulation.euler import log_spot
from locsvol.simulation.random import brownian_increments


@dataclass(frozen=True)
class Paths:
    """The variance factor Y at the nodes (paths, nodes) and the spot's increments (paths, steps).

    The paths of the spot follow for every leverage factor from log_spot.
    """

    times: jax.Array
    variance: jax.Array
    spot_increments: jax.Array

    def log_spot(self, leverage):
        """X = log(S/F) at the nodes for the leverage factor, by the Euler scheme."""
        return log_spot(leverage, self.times, self.variance, self.spot_increments)


def simulate(backbone, times, stream, paths, sobol=True, bridge=True, antithetic=True, segments=()):
    """Paths of the backbone's variance factor on the grid times; see brownian_increments."""
    increments = brownian_increments(
        times, stream, paths, backbone.factors, sobol, bridge, antithetic, segments=segments
    )
    return Paths(times, backbone.variance(times, increments), increments[:, :, 0])
