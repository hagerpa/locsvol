"""Time grids for the Euler scheme.

The paper's grid is graded near zero, where the short maturities are resolved: t_k = T_h (k/K)^p
for k = 0, ..., K, followed by equal steps up to the horizon. The maturities of the quotes are
inserted as nodes, so that the paths are observed exactly there.
"""

import jax.numpy as jnp
import numpy as np


def graded_times(horizon, head_horizon, head_steps, power, tail_step):
    """t_k = head_horizon (k / head_steps)^power, then steps of tail_step, ending at horizon."""
    head = head_horizon * (np.arange(head_steps + 1) / head_steps) ** power
    count = int(np.floor((horizon - head_horizon) / tail_step - 1e-9))
    tail = head_horizon + tail_step * np.arange(1, count + 1)
    return jnp.asarray(np.concatenate((head, tail, [horizon])))


def with_dates(times, dates):
    """The grid with the dates inserted as nodes."""
    return jnp.asarray(np.unique(np.concatenate((np.asarray(times), np.asarray(dates)))))


def node_indices(times, dates):
    """Indices of the dates among the nodes of a grid that contains them."""
    return jnp.asarray(np.searchsorted(np.asarray(times), np.asarray(dates)), dtype=jnp.int32)


def trapezoid_weights(times):
    """Weights w_j with sum_j w_j f(t_j) ~ (1/t_n) int_0^{t_n} f(t) dt by the trapezoidal rule."""
    dt = jnp.diff(times)
    return jnp.concatenate((dt[:1], dt[:-1] + dt[1:], dt[-1:])) / (2.0 * times[-1])
