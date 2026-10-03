"""Functions of (t, x) inside compiled code.

Leverage factors and targets are callables (t, x) -> value. Compiled functions such as the Euler
scheme take them as arguments, which requires them to be JAX pytrees. The dataclasses of this
package are; a plain Python function is wrapped by jax.tree_util.Partial, which holds it as a
static part.
"""

import functools
import types

import jax


def compiled(function):
    """The callable as a pytree for compiled code; a callable already wrapped stays as it is, so
    that rebuilding a dataclass keeps its structure and compiled code is not traced again."""
    if isinstance(function, jax.tree_util.Partial):
        return function
    if isinstance(function, types.FunctionType | types.BuiltinFunctionType | functools.partial):
        return jax.tree_util.Partial(function)
    return function
