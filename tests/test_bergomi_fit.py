import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401
from locsvol.backbones.bergomi import Bergomi, flat_forward_variance
from locsvol.calibration.bergomi_fit import (
    PARAMETERS,
    SCALES,
    bergomi_prices,
    correlation_minors,
    displaced,
    fit_bergomi,
    fit_bergomi_interior_point,
)
from locsvol.calibration.gauss_newton import interval_residuals
from locsvol.simulation.random import brownian_increments

TIMES = jnp.linspace(0.0, 1.0, 13)
TEMPLATE = Bergomi(1.0, 0.5, -0.7, -0.4, forward_variance=flat_forward_variance(0.04))


def increments(count=256):
    return [brownian_increments(TIMES, {"seed": 5, "replication": r}, count, 3) for r in range(2)]


def test_correlation_minors_decide_positive_definiteness():
    rng = np.random.default_rng(0)
    for a, b, c in rng.uniform(-1, 1, (200, 3)):
        matrix = np.array([[1, a, b], [a, 1, c], [b, c, 1]])
        minors = correlation_minors(Bergomi(1.0, 0.5, a, b, rho_z=c))
        assert (np.linalg.eigvalsh(matrix).min() > 0) == bool(jnp.all(minors > 0))


def test_displaced_parameters():
    backbone = displaced(
        TEMPLATE, jnp.array([0.2, -0.4, 0.1, 0.0]), jnp.asarray(SCALES), PARAMETERS
    )
    p = [backbone.nu, backbone.theta, backbone.rho1, backbone.rho2]
    np.testing.assert_allclose(p, [1.2, 0.3, -0.65, -0.4], atol=1e-15)


def test_price_derivatives_match_finite_differences():
    maturities, k = jnp.array([0.5, 1.0, 1.0]), jnp.array([0.0, -0.1, 0.1])
    call, x, paths = (
        jnp.array([True, False, True]),
        jnp.array([0.1, -0.1, 0.05, 0.02]),
        increments(64),
    )

    def prices(x):
        return bergomi_prices(TEMPLATE, x, TIMES, paths, maturities, k, call)

    value, jacobian = prices(x)
    for i, h in enumerate(1e-6 * jnp.eye(4)):
        central = (prices(x + h)[0] - prices(x - h)[0]) / 2e-6
        np.testing.assert_allclose(jacobian[:, i], central, atol=1e-7)
    assert value.shape == (3,)


def test_fit_decreases_the_loss_within_the_bounds():
    paths = increments()
    maturities = jnp.repeat(jnp.array([0.25, 0.5, 1.0]), 5)
    k, call = jnp.tile(jnp.linspace(-0.2, 0.1, 5), 3), jnp.ones(15, bool)
    quoted, _ = bergomi_prices(
        TEMPLATE, jnp.array([0.5, -0.2, -0.2, 0.2]), TIMES, paths, maturities, k, call
    )
    weights, vegas = jnp.full(15, 1 / 15), jnp.full(15, 0.1)
    fitted = fit_bergomi(
        TEMPLATE, TIMES, paths, maturities, k, call, quoted, quoted, weights, vegas, (0.25, 0.25)
    )

    def loss(backbone):
        prices, _ = bergomi_prices(backbone, jnp.zeros(4), TIMES, paths, maturities, k, call)
        r = interval_residuals(prices, quoted, quoted, weights, vegas)
        return float(r @ r)

    losses = [loss(b) for b in (TEMPLATE, *fitted)]
    assert losses[2] < losses[1] < losses[0]
    previous = jnp.zeros(4)
    for backbone in fitted:
        x = jnp.array(
            [
                (getattr(backbone, n) - getattr(TEMPLATE, n)) / s
                for n, s in zip(PARAMETERS, SCALES, strict=True)
            ]
        )
        assert float(jnp.linalg.norm(x - previous)) <= 0.25 * (1 + 1e-7)
        assert bool(jnp.all(correlation_minors(backbone) >= 1e-4))
        previous = x


def test_interior_point_fit_recovers_the_model_that_generated_the_quotes():
    paths = increments(64)
    maturities = jnp.repeat(jnp.array([0.25, 0.5, 1.0]), 5)
    k, call = jnp.tile(jnp.linspace(-0.2, 0.1, 5), 3), jnp.ones(15, bool)
    truth = jnp.array([0.3, -0.2, -0.2, 0.2])
    quoted, _ = bergomi_prices(TEMPLATE, truth, TIMES, paths, maturities, k, call)
    weights, vegas = jnp.full(15, 1 / 15), jnp.full(15, 0.1)
    fitted = fit_bergomi_interior_point(
        TEMPLATE, TIMES, paths, maturities, k, call, quoted, quoted, weights, vegas
    )
    expected = displaced(TEMPLATE, truth, jnp.asarray(SCALES), PARAMETERS)
    found = [getattr(fitted, n) for n in PARAMETERS]
    np.testing.assert_allclose(found, [getattr(expected, n) for n in PARAMETERS], atol=1e-4)


def test_uncertified_step_is_taken_at_a_looser_tolerance(monkeypatch):
    from locsvol.calibration import bergomi_fit
    from locsvol.errors import ConvergenceError

    original, tolerances = bergomi_fit.slsqp, []

    def certified_only_when_loose(loss, margins, jacobian, solver):
        tolerances.append(solver.tolerance)
        if solver.tolerance < 1e-9:
            raise ConvergenceError(
                "SLSQP: status 8, Positive directional derivative for linesearch"
            )
        return original(loss, margins, jacobian, solver)

    monkeypatch.setattr(bergomi_fit, "slsqp", certified_only_when_loose)
    paths = increments(64)
    maturities, k, call = jnp.array([0.5, 1.0, 1.0]), jnp.array([0.0, -0.1, 0.1]), jnp.ones(3, bool)
    quoted, _ = bergomi_prices(
        TEMPLATE, jnp.array([0.3, 0.0, 0.0, 0.0]), TIMES, paths, maturities, k, call
    )
    weights, vegas = jnp.full(3, 1 / 3), jnp.full(3, 0.1)
    fitted = fit_bergomi(
        TEMPLATE, TIMES, paths, maturities, k, call, quoted, quoted, weights, vegas, (0.25,)
    )
    assert len(fitted) == 1 and tolerances == [1e-10, 1e-8]
