"""Raw SVI slices (Gatheral, Global Derivatives conference, 2004; Gatheral and Jacquier,
Quant. Finance 14, 2014).

A raw SVI slice with parameters p = (a, b, rho, m, sigma), b > 0, |rho| < 1, sigma > 0, is the
total variance smile

    w(k) = a + b (rho (k - m) + r),    r = sqrt((k - m)^2 + sigma^2),

with derivatives w'(k) = b (rho + (k - m) / r) and w''(k) = b sigma^2 / r^3, minimum
a + b sigma sqrt(1 - rho^2), and slopes b (1 - rho) and b (1 + rho) of its left and right wing.
Where rho (k - m) < 0 the two terms of rho (k - m) + r cancel, and the slice is evaluated in the
equivalent form r (s^2 + (1 - rho^2) n^2) / (1 - rho n) with n = (k - m) / r and s = sigma / r.

The slice of an SSVI surface with ATM total variance theta, correlation rho and phi = phi(theta)
is the raw SVI slice (Gatheral and Jacquier, Lemma 3.1)

    a = theta (1 - rho^2)/2,    b = theta phi/2,    m = -rho/phi,    sigma = sqrt(1 - rho^2)/phi.
"""

import jax.numpy as jnp


def svi(parameters, log_moneyness):
    """w(k) of the raw SVI slice with parameters (a, b, rho, m, sigma)."""
    a, b, rho, m, sigma = parameters
    q = log_moneyness - m
    r = jnp.sqrt(q**2 + sigma**2)
    n, s = q / r, sigma / r
    rationalized = r * (s**2 + (1.0 - rho) * (1.0 + rho) * n**2) / (1.0 - rho * n)
    return a + b * jnp.where(rho * q >= 0.0, r + rho * q, rationalized)


def svi_derivatives(parameters, log_moneyness):
    """w'(k) and w''(k) of the raw SVI slice."""
    _, b, rho, m, sigma = parameters
    q = log_moneyness - m
    r = jnp.sqrt(q**2 + sigma**2)
    return b * (rho + q / r), b * sigma**2 / r**3


def svi_minimum(parameters):
    """min_k w(k) = a + b sigma sqrt(1 - rho^2)."""
    a, b, rho, _, sigma = parameters
    return a + b * sigma * jnp.sqrt((1.0 - rho) * (1.0 + rho))


def svi_wings(parameters):
    """The slopes b (1 - rho) and b (1 + rho) of the left and right wing."""
    _, b, rho, _, _ = parameters
    return b * (1.0 - rho), b * (1.0 + rho)


def svi_valid(parameters):
    """Whether b > 0, |rho| < 1, sigma > 0 and the slice's minimum is positive."""
    _, b, rho, _, sigma = parameters
    return (b > 0.0) & (jnp.abs(rho) < 1.0) & (sigma > 0.0) & (svi_minimum(parameters) > 0.0)


def svi_from_ssvi(theta, rho, phi):
    """The raw SVI parameters (a, b, rho, m, sigma) of an SSVI slice."""
    complement = (1.0 - rho) * (1.0 + rho)
    return jnp.stack(
        (
            0.5 * theta * complement,
            0.5 * theta * phi,
            rho,
            -rho / phi,
            jnp.sqrt(complement) / phi,
        )
    )
