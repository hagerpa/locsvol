"""Calibration of local stochastic volatility models by fixed-point iteration.

The subpackages are targets, backbones, simulation, leverage, regression, calibration and
pricing; save and load write and read the objects of the package. Importing locsvol enables
64-bit floating point in JAX.
"""

import jax

jax.config.update("jax_enable_x64", True)

from locsvol import (  # noqa: E402
    backbones,
    calibration,
    leverage,
    pricing,
    regression,
    simulation,
    targets,
)
from locsvol.io import load, save  # noqa: E402

__all__ = [
    "backbones",
    "calibration",
    "leverage",
    "load",
    "pricing",
    "regression",
    "save",
    "simulation",
    "targets",
]
