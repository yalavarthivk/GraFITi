r"""Utility functions."""

__all__ = [
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
    "initialize_from",
    "initialize_from_config",
    "is_dunder",
    "pad",
    # Classes
]

import logging
from collections.abc import Iterable, Mapping
from copy import deepcopy
from functools import partial, wraps
from importlib import import_module
from types import ModuleType
from typing import Any, TypeVar

import torch
from torch import Tensor, jit, nn

from linodenet.config import conf

__logger__ = logging.getLogger(__name__)

ObjectType = TypeVar("ObjectType")
r"""Generic type hint for instances."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""


@jit.script
def pad(
    x: Tensor,
    value: float,
    pad_width: int,
    dim: int = -1,
    prepend: bool = False,
) -> Tensor:
    r"""Pad a tensor with a constant value along a given dimension."""
    shape = list(x.shape)
    shape[dim] = pad_width
    z = torch.full(shape, value, dtype=x.dtype, device=x.device)

    if prepend:
        return torch.cat((z, x), dim=dim)
    return torch.cat((x, z), dim=dim)


def deep_dict_update(d: dict, new: Mapping, inplace: bool = False) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new: Mapping
    inplace: bool = False
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            d[key] = new[key]
    return d


def deep_keyval_update(d: dict, **new_kv: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kv: Mapping
    """
    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_keyval_update(d.get(key, {}), **new_kv)
        elif key in new_kv:
            d[key] = new_kv[key]
    return d


def autojit(base_class: type[nnModuleType]) -> type[nnModuleType]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule():
            ...

        model = jit.script(MyModule())

    and

    .. code-block:: python


        class MyModule():
            ...

        model = MyModule()

    are (roughly?) equivalent

    Parameters
    ----------
    base_class: type[nn.Module]

    Returns
    -------
    type
    """
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[misc, valid-type]
        r"""A simple Wrapper."""

        # noinspection PyArgumentList
        def __new__(cls, *args: Any, **kwargs: Any) -> nnModuleType:  # type: ignore[misc]
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: nnModuleType = base_class(*args, **kwargs)

            if conf.autojit:
                scripted: nnModuleType = jit.script(instance)
                return scripted
            return instance

    assert issubclass(WrappedClass, base_class)
    return WrappedClass


def flatten(inputs: Tensor | Iterable[Tensor]) -> Tensor:
    r"""Flattens element of general Hilbert space.

    Parameters
    ----------
    inputs: Tensor

    Returns
    -------
    Tensor
    """
    if isinstance(inputs, Tensor):
        return torch.flatten(inputs)
    if isinstance(inputs, Iterable):
        return torch.cat([flatten(x) for x in inputs])
    raise ValueError(f"{inputs=} not understood")


def initialize_from(
    lookup_table: ModuleType | dict[str, type[ObjectType]],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectType:
    r"""Lookup class/function from dictionary and initialize it.

    Roughly equivalent to:

    .. code-block:: python

        obj = lookup_table[__name__]
        if isclass(obj):
            return obj(**kwargs)
        return partial(obj, **kwargs)

    Parameters
    ----------
    lookup_table: dict[str, Callable]
    __name__: str
        The name of the class/function
    kwargs: Any
        Optional arguments to initialize class/function

    Returns
    -------
    Callable
        The initialized class/function
    """
    if isinstance(lookup_table, ModuleType):
        assert hasattr(lookup_table, __name__)
        obj: type[ObjectType] = getattr(lookup_table, __name__)
    else:
        obj = lookup_table[__name__]

    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        return obj(**kwargs)
    # if it is function, fix kwargs
    return partial(obj, **kwargs)  # type: ignore[return-value]


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize a class from a dictionary.

    Parameters
    ----------
    config: dict[str, Any]

    Returns
    -------
    object
    """
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    obj = cls(**opts)
    assert isinstance(obj, nn.Module)
    return obj


def is_dunder(name: str) -> bool:
    r"""Check if name is a dunder method.

    Parameters
    ----------
    name: str

    Returns
    -------
    bool
    """
    return name.startswith("__") and name.endswith("__")
