"""A positive conditional mean by a neural network.

The network with hidden widths n_1, ..., n_L maps the standardized coordinates
z = (x - x_mean) / x_scale to

    h(x) = softplus(W_{L+1}^T a_L + c_{L+1}),   a_l = tanh(W_l^T a_{l-1} + c_l),   a_0 = z,

with softplus(v) = log(1 + e^v) > 0, and the fitted mean is m(x) = s h(x), with the mean s of the
responses as scale, or m(x, o) = exp(o) h(x) with a log offset o. The responses, not their
logarithms, are fitted by least squares with weight decay,

    J = sum_n (m(x_n, o_n) - y_n)^2 / (2 N s^2) + alpha |W|^2 / (2 N),

where |W|^2 sums the squared weights W_l but no biases c_l. All observations carry the same
weight.

Training uses Adam (Kingma and Ba, ICLR 2015) on minibatches: every epoch draws a random
permutation of the observations and steps through it in batches. Every check_every epochs, J is
evaluated on all observations and the parameters are kept if J falls below the smallest value J*
so far by more than tolerance * max(1, J*); training stops after patience epochs without such a
decrease or after max_epochs, and returns the kept parameters. The hidden weights start uniform
on [-r, r] with r = sqrt(6 / (n_in + n_out)) (Glorot and Bengio, AISTATS 2010) and the biases at
zero; the output weights start at zero and the output bias at log(e - 1), so that h = 1 and m = s
at the start.

x_mean and x_scale are the population means and standard deviations of the observed coordinates;
a coordinate whose variance is zero up to rounding keeps the scale one. The fitted mean is
floored at a small positive number.
"""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from locsvol.errors import ConvergenceError
from locsvol.simulation.random import stream_key


class Parameters(NamedTuple):
    """Weights W_l (n_{l-1}, n_l) and biases c_l (n_l,) of the layers, the last of width one."""

    weights: tuple
    biases: tuple


class Adam(NamedTuple):
    """Parameters, first and second moment estimates, step count, and whether all are finite."""

    parameters: Parameters
    first: Parameters
    second: Parameters
    step: jax.Array
    finite: jax.Array


@partial(jax.jit, static_argnames=("widths", "epochs"))
def initialize(key, widths, epochs):
    """Initial parameters for the layer widths (d, n_1, ..., n_L, 1) and one permutation key
    per epoch."""
    init_key, shuffle_key = jax.random.split(key)
    layer_keys = jax.random.split(init_key, len(widths) - 2)
    weights, biases = [], []
    for i, (left, right) in enumerate(zip(widths[:-2], widths[1:-1], strict=True)):
        bound = jnp.sqrt(6.0 / (left + right))
        weights.append(jax.random.uniform(layer_keys[i], (left, right), jnp.float64, -bound, bound))
        biases.append(jnp.zeros(right))
    weights.append(jnp.zeros((widths[-2], 1)))
    biases.append(jnp.full((1,), jnp.log(jnp.expm1(1.0))))
    epoch_keys = jax.random.key_data(jax.random.split(shuffle_key, epochs))
    return Parameters(tuple(weights), tuple(biases)), epoch_keys


@jax.jit
def standardize(x):
    """(x - c) / s with the column means c and standard deviations s, and c and s; s is one
    where the variance vanishes."""
    n = x.shape[0]
    mean = jnp.mean(x, axis=0)
    centered = x - mean
    variance = (jnp.sum(centered**2, axis=0) - jnp.sum(centered, axis=0) ** 2 / n) / n
    eps = jnp.finfo(jnp.float64).eps
    constant = variance <= n * eps * variance + (n * mean * eps) ** 2
    scale = jnp.where(constant, 1.0, jnp.sqrt(variance))
    return centered / scale, mean, scale


def forward(parameters, z):
    """h at standardized coordinates z of shape (..., d), shape (...)."""
    for w, c in zip(parameters.weights[:-1], parameters.biases[:-1], strict=True):
        z = jnp.tanh(z @ w + c)
    return jnp.logaddexp(0.0, (z @ parameters.weights[-1] + parameters.biases[-1])[..., 0])


def _errors(parameters, z, y, offset):
    prediction = forward(parameters, z)
    if offset is not None:
        prediction = jnp.exp(offset) * prediction
    return prediction - y


def _decay(parameters, alpha, count):
    return 0.5 * alpha * sum(jnp.sum(w * w) for w in parameters.weights) / count


def _batch_loss(parameters, z, y, mask, alpha, count, offset):
    errors = _errors(parameters, z, y, offset)
    return 0.5 * jnp.sum(jnp.where(mask, errors**2, 0.0)) / jnp.sum(mask) + _decay(
        parameters, alpha, count
    )


def _adam_step(state, gradients, learning_rate, beta1, beta2, epsilon):
    first = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state.first, gradients)
    second = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g * g, state.second, gradients)
    step = state.step + 1
    parameters = jax.tree.map(
        lambda p, m, v: (
            p
            - learning_rate * (m / (1 - beta1**step)) / (jnp.sqrt(v / (1 - beta2**step)) + epsilon)
        ),
        state.parameters,
        first,
        second,
    )
    leaves = jax.tree.leaves((parameters, first, second, gradients))
    finite = state.finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in leaves]))
    return Adam(parameters, first, second, step, finite)


def _batches(key_data, count, batch_size):
    """Row indices (batches, batch_size) of a random permutation and the mask of real rows."""
    key = jax.random.wrap_key_data(key_data, impl="threefry2x32")
    permutation = jax.random.permutation(key, count)
    padding = (-count) % batch_size
    indices = jnp.concatenate((permutation, jnp.full((padding,), permutation[0])))
    mask = jnp.arange(count + padding) < count
    return indices.reshape(-1, batch_size), mask.reshape(-1, batch_size)


@partial(jax.jit, static_argnames="batch_size")
def train(state, z, y, keys, alpha, learning_rate, beta1, beta2, epsilon, batch_size, offset):
    """Adam steps over the epochs of the permutation keys (epochs, 2)."""

    def epoch(state, key):
        indices, masks = _batches(key, y.size, batch_size)

        def batch(state, rows):
            index, mask = rows
            loss, gradients = jax.value_and_grad(_batch_loss)(
                state.parameters,
                z[index],
                y[index],
                mask,
                alpha,
                y.size,
                None if offset is None else offset[index],
            )
            state = _adam_step(state, gradients, learning_rate, beta1, beta2, epsilon)
            return state._replace(finite=state.finite & jnp.isfinite(loss)), None

        return jax.lax.scan(batch, state, (indices, masks))[0], None

    return jax.lax.scan(epoch, state, keys)[0]


@partial(jax.jit, static_argnames="block_rows")
def objective(parameters, z, y, alpha, block_rows, offset):
    """J on all observations, summed over blocks of rows."""
    size = min(block_rows, y.size)
    positions = jnp.arange(((y.size + size - 1) // size) * size).reshape(-1, size)

    def add(total, index):
        valid = index < y.size
        rows = jnp.minimum(index, y.size - 1)
        errors = _errors(parameters, z[rows], y[rows], None if offset is None else offset[rows])
        return total + jnp.sum(jnp.where(valid, errors**2, 0.0)), None

    squared = jax.lax.scan(add, jnp.asarray(0.0), positions)[0]
    return 0.5 * squared / y.size + _decay(parameters, alpha, y.size)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["parameters", "center", "scale", "response_scale", "floor"],
    meta_fields=["residual"],
)
@dataclass(frozen=True)
class Network:
    """The fitted mean max(s h(x), floor), or max(exp(o) h(x), floor) when residual."""

    parameters: Parameters
    center: jax.Array
    scale: jax.Array
    response_scale: jax.Array
    floor: jax.Array
    residual: bool = False

    def unfloored(self, x, offset=None):
        """s h(x), or exp(o) h(x), before the floor."""
        if self.residual != (offset is not None):
            raise ValueError("a residual network needs its offset, a direct one none")
        x = jnp.asarray(x, dtype=jnp.float64)
        baseline = self.response_scale if offset is None else jnp.exp(jnp.asarray(offset))
        values = baseline * forward(self.parameters, (x - self.center) / self.scale)
        return jnp.where(jnp.isfinite(baseline) & (baseline > 0), values, jnp.nan)

    def __call__(self, x, offset=None):
        mean = self.unfloored(x, offset)
        return jnp.where(jnp.isfinite(mean), jnp.maximum(mean, self.floor), jnp.nan)


@dataclass(frozen=True)
class PositiveMeanNetwork:
    """The neural network regression of the module docstring.

    stream is a dictionary with an integer 'seed' that fixes the initial parameters and the
    permutations; distinct streams give independent fits.
    """

    hidden_widths: tuple[int, ...]
    alpha: float = 1e-4
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    batch_size: int = 200
    max_epochs: int = 200
    check_every: int = 1
    tolerance: float = 1e-4
    patience: int = 10
    floor: float = float(np.finfo(np.float64).tiny)
    block_rows: int = 32768
    stream: dict | None = None

    def fit(self, x, y, weights=None, offset=None, stream=None):
        """The fitted mean on coordinates x (n, d), responses y > 0 of equal weights, with log
        offsets o for a residual fit; stream replaces the regression's stream."""
        x, y = jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)
        if weights is not None:
            w = jnp.asarray(weights)
            if not bool(jnp.all(w > 0) & jnp.all(w == w[0])):
                raise ValueError("the network fits observations of equal positive weight")
        stream = self.stream if stream is None else stream
        if stream is None:
            raise ValueError("the network needs a random stream")
        z, center, scale = standardize(x)
        response_scale = jnp.mean(y)
        scaled = y / response_scale
        if offset is not None:
            offset = jnp.broadcast_to(jnp.asarray(offset, dtype=jnp.float64), y.shape)
            scaled_offset = offset - jnp.log(response_scale)
        else:
            scaled_offset = None
        widths = (x.shape[1], *self.hidden_widths, 1)
        initial, keys = initialize(stream_key(stream), widths, self.max_epochs)
        zeros = jax.tree.map(jnp.zeros_like, initial)
        state = Adam(initial, zeros, zeros, jnp.asarray(0), jnp.asarray(True))

        def loss(parameters):
            return float(
                objective(parameters, z, scaled, self.alpha, self.block_rows, scaled_offset)
            )

        best, best_parameters = loss(initial), initial
        finite = isfinite(best)
        completed = unchanged = 0
        while finite and completed < self.max_epochs:
            following = min(completed + self.check_every, self.max_epochs)
            state = train(
                state,
                z,
                scaled,
                keys[completed:following],
                self.alpha,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                self.batch_size,
                scaled_offset,
            )
            finite = bool(state.finite)
            if not finite:
                break
            current = loss(state.parameters)
            finite = isfinite(current)
            if not finite:
                break
            if best - current > self.tolerance * max(1.0, best):
                best, best_parameters, unchanged = current, state.parameters, 0
            else:
                unchanged += following - completed
            completed = following
            if unchanged >= self.patience:
                break
        fitted = Network(
            best_parameters,
            center,
            scale,
            response_scale,
            jnp.asarray(self.floor, dtype=jnp.float64),
            offset is not None,
        )
        mean = fitted.unfloored(x, offset)
        if not finite or not bool(jnp.all(jnp.isfinite(mean) & (mean >= 0))):
            raise ConvergenceError("network training produced values that are not finite")
        return fitted
