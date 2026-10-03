"""The fixed-point iteration (Algorithm 2.1 of the paper).

A leverage factor L calibrates the model dS_t = L(t, S_t) sqrt(Y_t) S_t dW_t to the target local
volatility sigma_D when

    L(t, s)^2 E[Y_t | S_t = s] = sigma_D(t, s)^2.

Starting from L_0, the iteration maps L_i to L_{i+1}(t, s)^2 = sigma_D(t, s)^2 / E_i[Y_t | S_t = s],
where E_i is the expectation in the model with leverage factor L_i. One iteration

1. simulates the log spot X = log S on the time grid with leverage factor L_i; the paths of the
   variance factor Y do not depend on L and are simulated once;
2. fits the Gamma regression of Y_t on the coordinates of (t, X_t) with the offset
   log(sigma_D^2 / L_i^2), so that E_i[Y_t | X_t = x] is approximated by
   sigma_D(t, x)^2 / L_i(t, x)^2 exp(b(t, x)^T alpha_i);
3. sets L_{i+1} = L_i exp(-b^T alpha_i / 2), that is, beta_{i+1} = beta_i - alpha_i / 2 for the
   leverage factor L_beta = L_0 exp(b^T beta).

The regression uses every k-th time node of every path: path m starts at node floor(k v_m) with
the van der Corput sequence v_m in base two, so that every node is used equally often. Each row
carries k times the trapezoidal weight of its time.
"""

import jax.numpy as jnp
import numpy as np

from locsvol.simulation.grid import trapezoid_weights


def van_der_corput(count):
    """The first count points of the van der Corput sequence in base two."""
    index, value, scale = np.arange(count), np.zeros(count), 0.5
    while index.any():
        value += scale * (index & 1)
        index >>= 1
        scale *= 0.5
    return value


def every_kth_node(paths, nodes, k):
    """Mask (paths, nodes) of the rows used by the regression."""
    first = np.floor(k * van_der_corput(paths)).astype(int)
    return np.arange(nodes)[None, :] % k == first[:, None]


def fixed_point(leverage, target, paths, regression, iterations, k=4):
    """The leverage factors L_1, ..., L_iterations, starting from L_0 = leverage.

    leverage is a parametrized leverage factor L_beta, target the local volatility sigma_D,
    paths the simulated variance factor with its Brownian increments, and regression for instance
    a GammaRegression on a Chebyshev basis.
    """
    rows = every_kth_node(*paths.variance.shape, k)
    t = jnp.broadcast_to(paths.times, rows.shape)[rows]
    y = paths.variance[rows]
    w = k * jnp.broadcast_to(trapezoid_weights(paths.times), rows.shape)[rows]
    alpha, iterates = None, []
    for _ in range(iterations):
        x = paths.log_spot(leverage)[rows]
        offset = 2 * jnp.log(target(t, x)) - 2 * jnp.log(leverage(t, x))
        alpha = regression.fit(leverage.coordinates(t, x), y, w, offset, start=alpha)
        leverage = leverage.shifted(-alpha / 2)
        iterates.append(leverage)
    return iterates
