r"""LinODE-Net Configuration.

# TODO: There must be a better way to handle global config
"""

__all__ = [
    # Constants
    "conf",
    # Classes
    "Config",
]

import os

os.environ["LINODENET_AUTOJIT"] = "True"
r"""Default value."""

_NAME = __name__
_FILE = __file__


class Config:
    r"""Configuration Interface."""

    # TODO: Should be initialized by a init/toml file.
    _autojit: bool = True
    __name__ = _NAME
    __file__ = _FILE

    @property
    def autojit(self) -> bool:
        r"""Whether to automatically jit-compile the models."""
        return self._autojit

    @autojit.setter
    def autojit(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._autojit = bool(value)
        os.environ["LINODENET_AUTOJIT"] = str(value)


conf: Config = Config()  # = Config(__name__, __doc__)
r"""The unique `~linodenet.config.Config` instance used to configure `linodenet`."""
