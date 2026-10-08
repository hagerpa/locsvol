"""Brownian increments on a time grid.

The increments of all factors on a grid 0 = t_0 < t_1 < ... < t_n are made from standard
normals Z, which are either independent pseudo-random numbers or scrambled Sobol points mapped by
the inverse normal distribution function (randomized quasi-Monte Carlo). The normals become
increments chronologically, dW_j = sqrt(t_{j+1} - t_j) Z_j, or by a Brownian bridge: the first
normal of a path sets W(t_n), and each following one sets W at the node nearest the midpoint of
the widest remaining interval, given its two ends. The bridge lets the leading Sobol coordinates
carry the coarse shape of the paths (Glasserman, Monte Carlo Methods in Financial Engineering,
Sec. 3.1 and 5.2). Antithetic pairs append the negated increments.

With segments, the grid is cut at the given times and every segment gets its own Sobol points
(an independent scramble) and its own bridge, and the paths of every segment after the first are
paired with those of the preceding ones by a random permutation; antithetic pairs stay paired.
A single set of Sobol points spends its leading coordinates on the coarse shape of the whole
horizon, so on a long grid the first days are driven by deep coordinates and lose the quality of
the points there; segments of about a year give every part of the grid leading coordinates.

A stream is a dictionary with an integer "seed" and any further entries. Its SHA-256 digest
seeds the generator: the same dictionary always gives the same numbers, and different
dictionaries give independent ones.
"""

import heapq
import json
from functools import partial
from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri
from scipy.stats import qmc


def stream_digest(stream):
    """SHA-256 digest of the stream's description."""
    text = json.dumps(stream, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode()).digest()


@jax.jit
def inverse_normal(points):
    """Phi^{-1}(u) for 0 < u < 1, and nan otherwise."""
    valid = jnp.isfinite(points) & (points > 0) & (points < 1)
    return jnp.where(valid, ndtri(jnp.where(valid, points, 0.5)), jnp.nan)


def sobol_normals(stream, paths, steps, factors, bits=64):
    """Normals (paths, steps, factors) from scrambled Sobol points; paths is a power of two."""
    sampler = qmc.Sobol(
        d=steps * factors,
        scramble=True,
        bits=bits,
        rng=np.random.default_rng(int.from_bytes(stream_digest(stream), "little")),
    )
    points = sampler.random_base2(paths.bit_length() - 1)
    return inverse_normal(jnp.asarray(points)).reshape(paths, steps, factors)


@partial(jax.jit, static_argnames=("paths", "steps", "factors"))
def _threefry_normals(key, paths, steps, factors):
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(paths, dtype=jnp.uint32))
    return jax.vmap(lambda k: jax.random.normal(k, (steps, factors), dtype=jnp.float64))(keys)


def stream_key(stream):
    """The Threefry key of a stream: its seed, folded with the words of its digest."""
    key = jax.random.key(stream["seed"], impl="threefry2x32")
    for word in np.frombuffer(stream_digest(stream), dtype="<u4"):
        key = jax.random.fold_in(key, int(word))
    return key


def pseudo_random_normals(stream, paths, steps, factors):
    """Independent normals (paths, steps, factors); path i depends only on the stream and i."""
    return _threefry_normals(stream_key(stream), paths, steps, factors)


def bridge_schedule(times):
    """Order of the Brownian bridge: for each normal after the first, its node and the two
    nodes it lies between."""
    t = np.asarray(times)
    intervals = [(-(t[-1] - t[0]), 0, t.size - 1)] if t.size > 2 else []
    nodes, lefts, rights = [], [], []
    while intervals:
        _, left, right = heapq.heappop(intervals)
        midpoint = (t[left] + t[right]) / 2
        position = int(np.searchsorted(t, midpoint))
        candidates = [i for i in (position - 1, position) if left < i < right]
        node = min(candidates, key=lambda i: (abs(t[i] - midpoint), i))
        nodes.append(node)
        lefts.append(left)
        rights.append(right)
        if node - left > 1:
            heapq.heappush(intervals, (-(t[node] - t[left]), left, node))
        if right - node > 1:
            heapq.heappush(intervals, (-(t[right] - t[node]), node, right))
    return tuple(jnp.asarray(v, dtype=jnp.int32) for v in (nodes, lefts, rights))


@jax.jit
def bridge_increments(normals, times, schedule):
    """Increments (paths, steps, factors) of the Brownian bridge driven by the normals."""
    nodes, lefts, rights = schedule
    paths = jnp.zeros((normals.shape[0], times.size, normals.shape[2]), dtype=normals.dtype)
    paths = paths.at[:, -1, :].set(jnp.sqrt(times[-1]) * normals[:, 0, :])

    def insert(index, values):
        node, left, right = nodes[index], lefts[index], rights[index]
        a, b = times[node] - times[left], times[right] - times[node]
        value = (b * values[:, left, :] + a * values[:, right, :]) / (a + b)
        value += jnp.sqrt(a * b / (a + b)) * normals[:, index + 1, :]
        return values.at[:, node, :].set(value)

    if nodes.size:
        paths = jax.lax.fori_loop(0, nodes.size, insert, paths)
    return jnp.diff(paths, axis=1)


@jax.jit
def chronological_increments(normals, times):
    """Increments sqrt(t_{j+1} - t_j) Z_j."""
    return normals * jnp.sqrt(jnp.diff(times))[None, :, None]


def segment_nodes(times, segments):
    """Indices of the nodes nearest to the segment boundaries, with the grid's first and last."""
    t = np.asarray(times)
    cuts = sorted({int(np.argmin(np.abs(t - float(b)))) for b in segments} - {0, t.size - 1})
    return [0, *cuts, t.size - 1]


def brownian_increments(
    times,
    stream,
    paths,
    factors,
    sobol=True,
    bridge=True,
    antithetic=True,
    bits=64,
    segments=(),
):
    """Independent Brownian increments (paths, steps, factors) on the grid times.

    With antithetic=True, paths counts both halves of the pairs. The first factor drives the
    spot in every model of this package. segments are the times at which the grid is cut into
    segments with their own Sobol points (module docstring).
    """
    times = jnp.asarray(times, dtype=jnp.float64)
    if len(segments):
        cuts = segment_nodes(times, segments)
        pieces = []
        for n, (i, j) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
            piece = brownian_increments(
                times[i : j + 1] - times[i],
                dict(stream, segment=n),
                paths,
                factors,
                sobol,
                bridge,
                antithetic,
                bits,
            )
            if n > 0:
                count = paths // 2 if antithetic else paths
                order = jax.random.permutation(stream_key(dict(stream, segment=n, pairing=True)), count)
                piece = piece[jnp.concatenate((order, order + count)) if antithetic else order]
            pieces.append(piece)
        return jnp.concatenate(pieces, axis=1)
    count, steps = paths // 2 if antithetic else paths, times.size - 1
    if sobol:
        normals = sobol_normals(stream, count, steps, factors, bits)
    else:
        normals = pseudo_random_normals(stream, count, steps, factors)
    if bridge:
        increments = bridge_increments(normals, times, bridge_schedule(times))
    else:
        increments = chronological_increments(normals, times)
    return jnp.concatenate((increments, -increments), axis=0) if antithetic else increments
