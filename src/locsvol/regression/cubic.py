"""Cubic spline interpolation with the not-a-knot end condition and flat extrapolation.

On nodes x_0 < ... < x_{n-1} with values v_i, the interpolant is a cubic on each interval, twice
continuously differentiable, and its third derivative is continuous at x_1 and x_{n-2}
(de Boor, A Practical Guide to Splines, Ch. IV). Outside [x_0, x_{n-1}] it is constant. The
coefficients of interval i are stored in descending powers of x - x_i. Vectors are padded to a
common capacity; the first count nodes are used.
"""

import jax
import jax.numpy as jnp


@jax.jit
def not_a_knot(nodes, values, count):
    """Coefficients (capacity - 1, 4) of the not-a-knot cubic through the first count nodes."""
    n = nodes.size
    active = jnp.arange(n - 1) < count - 1
    h = jnp.where(active, jnp.diff(nodes), 1.0)
    slope = jnp.where(active, jnp.diff(values) / h, 0.0)
    rows = jnp.arange(1, n - 1)
    interior = rows < count - 1
    matrix = jnp.eye(n, dtype=nodes.dtype)
    matrix = matrix.at[rows, rows - 1].set(jnp.where(interior, h[:-1], 0.0))
    matrix = matrix.at[rows, rows].set(jnp.where(interior, 2 * (h[:-1] + h[1:]), 1.0))
    matrix = matrix.at[rows, rows + 1].set(jnp.where(interior, h[1:], 0.0))
    matrix = matrix.at[0].set(0.0).at[0, :3].set(jnp.array([-h[1], h[0] + h[1], -h[0]]))
    last = count - 1
    matrix = matrix.at[last].set(0.0)
    matrix = matrix.at[last, last - 2].set(-h[last - 1])
    matrix = matrix.at[last, last - 1].set(h[last - 2] + h[last - 1])
    matrix = matrix.at[last, last].set(-h[last - 2])
    rhs = (
        jnp.zeros(n, dtype=values.dtype).at[rows].set(jnp.where(interior, 3 * jnp.diff(slope), 0.0))
    )
    second_half = jnp.linalg.solve(matrix, rhs)
    coefficients = jnp.stack(
        (
            jnp.diff(second_half) / (3 * h),
            second_half[:-1],
            slope - h * (2 * second_half[:-1] + second_half[1:]) / 3,
            values[:-1],
        ),
        axis=1,
    )
    return jnp.where(active[:, None], coefficients, 0.0)


def cubic(coefficients, dx):
    """Horner evaluation of descending coefficients (..., 4) at dx."""
    return ((coefficients[..., 0] * dx + coefficients[..., 1]) * dx + coefficients[..., 2]) * dx + (
        coefficients[..., 3]
    )


@jax.jit
def flat_cubic(nodes, values, coefficients, count, query):
    """The interpolant at the queries, equal to the end values outside the nodes."""
    bounded = jnp.clip(query, nodes[0], nodes[count - 1])
    index = jnp.clip(jnp.searchsorted(nodes, bounded, side="right") - 1, 0, count - 2)
    result = cubic(coefficients[index], bounded - nodes[index])
    result = jnp.where(
        query <= nodes[0],
        values[0],
        jnp.where(query >= nodes[count - 1], values[count - 1], result),
    )
    return jnp.where(jnp.isfinite(query), result, jnp.nan)


@jax.jit
def cubic_minimum(nodes, coefficients, count):
    """The minimum of the interpolant over [x_0, x_{count-1}], from the interval ends and the real
    stationary points inside the intervals."""
    active = jnp.arange(nodes.size - 1) < count - 1
    h = jnp.where(active, jnp.diff(nodes), 1.0)
    scale = jnp.max(jnp.abs(coefficients[:, :3]), axis=1)
    derivative = coefficients[:, :3] / jnp.where(scale > 0, scale, 1.0)[:, None]
    a, b, c = 3 * derivative[:, 0], 2 * derivative[:, 1], derivative[:, 2]
    discriminant = b**2 - 4 * a * c
    q = -0.5 * (b + jnp.copysign(jnp.sqrt(jnp.maximum(discriminant, 0.0)), b))
    first = q / jnp.where(a != 0, a, 1.0)
    second = c / jnp.where(q != 0, q, 1.0)
    linear = -c / jnp.where(b != 0, b, 1.0)
    roots = jnp.stack((jnp.where(a != 0, first, linear), second), axis=1)
    real = jnp.stack(
        (
            (a != 0) & (discriminant >= 0) | ((a == 0) & (b != 0)),
            (a != 0) & (discriminant >= 0) & (q != 0),
        ),
        axis=1,
    )
    inside = real & jnp.isfinite(roots) & (roots > 0) & (roots < h[:, None])
    candidates = cubic(coefficients[:, None, :], jnp.where(inside, roots, 0.0))
    ends = jnp.minimum(coefficients[:, 3], cubic(coefficients, h))
    result = jnp.minimum(ends, jnp.min(jnp.where(inside, candidates, jnp.inf), axis=1))
    return jnp.min(jnp.where(active, result, jnp.inf))
