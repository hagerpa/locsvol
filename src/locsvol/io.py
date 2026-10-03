"""Saving and loading the objects of the package: leverage factors, targets, backbones, charts.

save(path, obj) writes an object of the package, or a list, tuple or dictionary of them, into
one .npz file: its structure (classes, fields, numbers and strings) as JSON text and its arrays
as arrays. load(path) rebuilds the object.
A plain Python function inside an object, for instance a target given as a lambda, cannot be
saved.
"""

import importlib
import json
from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
import numpy as np


def _encode(value, arrays):
    if isinstance(value, jax.tree_util.Partial):
        raise TypeError("a plain Python function cannot be saved")
    if is_dataclass(value):
        kind = type(value)
        return dict(
            type=f"{kind.__module__}:{kind.__qualname__}",
            fields={f.name: _encode(getattr(value, f.name), arrays) for f in fields(value)},
        )
    if isinstance(value, jax.Array | np.ndarray | np.generic):
        key = f"array_{len(arrays)}"
        arrays[key] = np.asarray(value)
        return dict(array=key)
    if isinstance(value, tuple | list):
        return {type(value).__name__: [_encode(item, arrays) for item in value]}
    if isinstance(value, dict):
        return dict(dict={str(key): _encode(item, arrays) for key, item in value.items()})
    if value is None or isinstance(value, bool | int | float | str):
        return dict(value=value)
    raise TypeError(f"cannot save a {type(value).__name__}")


def _decode(node, arrays):
    if "type" in node:
        module, name = node["type"].split(":")
        kind = importlib.import_module(module)
        for part in name.split("."):
            kind = getattr(kind, part)
        return kind(**{key: _decode(item, arrays) for key, item in node["fields"].items()})
    if "array" in node:
        return jnp.asarray(arrays[node["array"]])
    if "tuple" in node:
        return tuple(_decode(item, arrays) for item in node["tuple"])
    if "list" in node:
        return [_decode(item, arrays) for item in node["list"]]
    if "dict" in node:
        return {key: _decode(item, arrays) for key, item in node["dict"].items()}
    return node["value"]


def save(path, value):
    """Write the object into the .npz file path."""
    arrays = {}
    structure = json.dumps(_encode(value, arrays))
    np.savez(path, structure=np.asarray(structure), **arrays)


def load(path):
    """The object saved in the .npz file path."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "structure"}
        return _decode(json.loads(str(data["structure"])), arrays)
