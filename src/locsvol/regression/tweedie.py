"""The Tweedie losses of a mean with log link.

A Tweedie distribution with power p has variance proportional to mu^p (Jorgensen, The Theory of
Dispersion Models, 1997): p = 1 gives the Poisson, 1 < p < 2 the compound Poisson-Gamma, p = 2
the Gamma and p = 3 the inverse Gaussian law. Half its deviance at the mean mu = exp(eta) is, up
to terms in y alone, the loss

    l_p(y, eta) = exp(eta) - y eta                                          (p = 1),
    l_p(y, eta) = eta + y exp(-eta)                                         (p = 2),
    l_p(y, eta) = exp((2 - p) eta) / (2 - p) - y exp((1 - p) eta) / (1 - p)  (otherwise),

which is minimal at exp(eta) = y and convex in eta for 1 <= p <= 2. Responses y >= 0 are
admitted for p < 2 and y > 0 otherwise. A regression with log link minimizes the weighted mean
of l_p; without a penalty its fitted mean solves sum_n w_n (y_n - mu_n) mu_n^{1-p} b(x_n) = 0.
"""

import jax.numpy as jnp


def tweedie_loss(y, eta, power):
    """l_p(y, eta) for the power p >= 1, a Python number."""
    if power == 2:
        return eta + y * jnp.exp(-eta)
    if power == 1:
        return jnp.exp(eta) - y * eta
    return -y * jnp.exp((1 - power) * eta) / (1 - power) + jnp.exp((2 - power) * eta) / (2 - power)
