r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    "conf",
    # Sub-Modules
    "config",
    "initializations",
    "models",
    "projections",
    "regularizations",
    "utils",
]

import logging
import sys
from importlib import metadata
from types import ModuleType

# version check
if sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 required")

# pylint: disable=wrong-import-position

from linodenet import (
    config,
    initializations,
    models,
    projections,
    regularizations,
    utils,
)
from linodenet.config import conf

# pylint: enable=wrong-import-position


__logger__ = logging.getLogger(__name__)
__version__ = metadata.version(__package__)
r"""The version number of the `linodenet` package."""


# Recursively clean up namespaces to only show what the user should see.
def _clean_namespace(module: ModuleType) -> None:
    r"""Recursively cleans up the namespace.

    Sets `obj.__module__` equal to `obj.__package__` for all objects listed in
    `package.__all__` that are originating from private submodules (`package/_module.py`).

    Parameters
    ----------
    module: ModuleType
    """
    __logger__.info("Cleaning module=%s", module)
    variables = vars(module)

    def is_private(s: str) -> bool:
        return s.startswith("_") and not s.startswith("__")

    def get_module(obj_ref: object) -> str:
        return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]

    assert hasattr(module, "__name__"), f"{module=} has no __name__ ?!?!"
    assert hasattr(module, "__package__"), f"{module=} has no __package__ ?!?!"
    assert hasattr(module, "__all__"), f"{module=} has no __all__!"
    assert module.__name__ == module.__package__, f"{module=} is not a package!"

    maxlen = max((len(key) for key in variables))

    def _format(key: str) -> str:
        return key.ljust(maxlen)

    for key in list(variables):
        key_repr = _format(key)
        obj = variables[key]
        # ignore _clean_namespace and ModuleType
        if key in ("ModuleType", "_clean_namespace"):
            __logger__.debug("key=%s  skipped! - protected object!", key_repr)
            continue
        # ignore dunder keys
        if key.startswith("__") and key.endswith("__"):
            __logger__.debug("key=%s  skipped! - dunder object!", key_repr)
            continue
        # special treatment for ModuleTypes
        if isinstance(obj, ModuleType):
            if obj.__package__ is None:
                __logger__.debug(
                    "key=%s  skipped! Module with no __package__!", key_repr
                )
                continue
            # subpackage!
            if obj.__package__.rsplit(".", maxsplit=1)[0] == module.__name__:
                __logger__.debug("key=%s  recursion!", key_repr)
                _clean_namespace(obj)
            # submodule!
            elif obj.__package__ == module.__name__:
                __logger__.debug("key=%s  skipped! Sub-Module!", key_repr)
                continue
            # 3rd party!
            else:
                __logger__.debug("key=%s  skipped! 3rd party Module!", key_repr)
                continue
        # key is found:
        if key in module.__all__:
            # set __module__ attribute to __package__ for functions/classes
            # originating from private modules.
            if isinstance(obj, type) or callable(obj):
                mod = get_module(obj)
                if is_private(mod):
                    __logger__.debug(
                        "key=%s  changed {obj.__module__=} to {module.__package__}!",
                        key_repr,
                    )
                    obj.__module__ = str(module.__package__)
        else:
            # kill the object
            delattr(module, key)
            __logger__.debug("key=%s  killed!", key_repr)
    # Post Loop - clean up the rest
    for key in ("ModuleType", "_clean_namespace"):
        if key in variables:
            key_repr = _format(key)
            delattr(module, key)
            __logger__.debug("key=%s  killed!", key_repr)


# recursively clean namespace from self.
_clean_namespace(__import__(__name__))
