"""Starting values of the regression with log link of gamma.py, for every basis.

Step 1 of the Gamma regression starts from the ridge regression of log y - o on b(x),

    alpha_0 = H^{-1} r,   H = G + P,   G = sum_n w_n b(x_n) b(x_n)^T / sum_n w_n,
    r = sum_n w_n (log y_n - o_n) b(x_n) / sum_n w_n,

with the penalty matrix P of the fit (diagonal lambda for a ridge penalty), and continues in the
coordinates theta = H^{1/2} alpha. The Chebyshev basis sums G from product moments; every other
basis sums it from its features b(x_n), block by block. H gets the smallest shift that makes its
eigenvalues at least machine precision times its largest one.

The other starting rules need neither G nor log y:

- zeros: alpha_0 = 0, that is the mean exp(o);
- mean: alpha_0 = log(ybar) e_k, for a basis whose k-th function is the constant 1;
- partition: alpha_0 = log(ybar) c, for a basis that represents the constant 1 by coefficients c
  (its property unit), as B-splines do by c = (1, ..., 1),

with the weighted mean ybar = sum_n w_n y_n / sum_n w_n of the responses, so that the start is
the constant mean ybar when o = 0. From these starts the fit runs L-BFGS-B in alpha itself, that
is with H^{1/2} = I, and takes no gradient steps.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

RULES = ("ridge", "zeros", "mean", "partition")


@jax.jit
def conditioned(gram, right, penalty):
    """alpha_0 = H^{-1} r, H^{1/2} and H^{-1/2} for H = G + P, P = diag(penalty) for a vector."""
    matrix = penalty if penalty.ndim == 2 else jnp.diag(penalty)
    eigenvalues, vectors = jnp.linalg.eigh(0.5 * (gram + gram.T) + matrix)
    floor = jnp.finfo(gram.dtype).eps * jnp.maximum(1.0, jnp.max(jnp.abs(eigenvalues)))
    eigenvalues = eigenvalues + jnp.maximum(0.0, floor - jnp.min(eigenvalues))
    root = (vectors * jnp.sqrt(eigenvalues)) @ vectors.T
    inverse_root = (vectors / jnp.sqrt(eigenvalues)) @ vectors.T
    return (vectors / eigenvalues) @ (vectors.T @ right), root, inverse_root


@partial(jax.jit, static_argnames="basis")
def dense_ridge_start(rows, penalty, basis):
    """Step 1 with G and r summed from the features of any basis; rows as in gamma.blocks."""

    def add(total, block):
        x, y, o, w = block
        active = w != 0
        features = basis(jnp.where(active[:, None], x, 0.0))
        z = jnp.log(jnp.where(active, y, 1.0)) - jnp.where(active, o, 0.0)
        return (
            total[0] + features.T @ (w[:, None] * features),
            total[1] + features.T @ (w * z),
        ), None

    zero = (jnp.zeros((basis.size, basis.size)), jnp.zeros(basis.size))
    (gram, right), _ = lax.scan(add, zero, rows)
    mass = jnp.sum(rows.weights)
    return conditioned(gram / mass, right / mass, penalty)


def rule_start(rule, rows, basis, intercept=None):
    """alpha_0 of the rule 'zeros', 'mean' (with the index of the constant function) or
    'partition'; see the module docstring."""
    if rule == "zeros":
        return jnp.zeros(basis.size)
    if rule == "mean":
        if intercept is None:
            raise ValueError("the rule 'mean' needs the index of the constant basis function")
        unit = jnp.zeros(basis.size).at[intercept].set(1.0)
    elif rule == "partition":
        unit = basis.unit
    else:
        raise ValueError(f"the starting rule must be one of {RULES}")
    log_mean = jnp.log(jnp.sum(rows.weights * rows.y) / jnp.sum(rows.weights))
    return jnp.where(unit != 0, log_mean * unit, 0.0)
