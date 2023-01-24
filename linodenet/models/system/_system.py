r"""Models for the latent dynamical system."""

__all__ = [
    # Classes
    "LinODECell",
]

from typing import Any, Final

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import FUNCTIONAL_INITIALIZATIONS
from linodenet.initializations.functional import gaussian
from linodenet.projections import PROJECTIONS
from linodenet.utils import deep_dict_update


class LinODECell(nn.Module):
    r"""Linear System module, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}x_t$.

    .. Signature:: ``[∆t=(...,), x=(..., d)] -> (..., d)]``.

    By default, the Cell is parametrized by

    .. math:: e^{γ⋅A⋅∆t}x

    Attributes
    ----------
    scalar: float
        PARAM - The scalar $γ$ in the parametrization.
    weight: torch.Tensor
        PARAM - The weight matrix $A$ in the parametrization.
    kernel: torch.Tensor
        BUFFER - The parametrized kernel $γ⋅A$. or $ψ(γ⋅A)$ if parametrized.
    scalar_learnable: bool
        PARAM - Whether the scalar $γ$ is learnable or not.

    Parameters
    ----------
    input_size: int
    kernel_initialization: Tensor | Callable[[int], Tensor]
    kernel_parametrization: nn.Module
        The parametrization to apply to the kernel matrix.

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    output_size: int
        The dimensionality of the output space.
    kernel: Tensor
        The system matrix
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "kernel_initialization": "skew-symmetric",
        "kernel_parametrization": None,
        "scalar": 0.0,
        "scalar_learnable": True,
    }

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""

    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    # Buffers
    scalar: Tensor
    r"""PARAM: the scalar applied to the kernel."""

    weight: Tensor
    r"""PARAM: The learnable weight-matrix of the linear ODE component."""

    # Parameters
    kernel: Tensor
    r"""BUFFER: The system matrix of the linear ODE component."""

    def __init__(
        self,
        input_size: int,
        **cfg: Any,
    ):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        self.input_size = input_size
        self.output_size = input_size
        kernel_init = config["kernel_initialization"]
        kernel_parametrization = config["kernel_parametrization"]

        def kernel_initialization_dispatch():
            r"""Dispatch the kernel initialization."""
            if kernel_init is None:
                return lambda: gaussian(input_size)
            if isinstance(kernel_init, str):
                assert (
                    kernel_init in FUNCTIONAL_INITIALIZATIONS
                ), "Unknown initialization!"
                _init = FUNCTIONAL_INITIALIZATIONS[kernel_init]
                return lambda: _init(input_size)
            if callable(kernel_init):
                assert Tensor(kernel_init(input_size)).shape == (
                    input_size,
                    input_size,
                )
                return lambda: Tensor(kernel_init(input_size))
            if isinstance(kernel_init, Tensor):
                tensor = kernel_init
                assert tensor.shape == (
                    input_size,
                    input_size,
                ), f"Kernel has bad shape! {tensor.shape} but should be {(input_size, input_size)}"
                return lambda: tensor

            tensor = Tensor(kernel_init)
            assert tensor.shape == (
                input_size,
                input_size,
            ), f"Kernel has bad shape! {tensor.shape} but should be {(input_size, input_size)}"
            return lambda: tensor

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_parametrization_dispatch():
            r"""Dispatch the kernel parametrization."""
            if kernel_parametrization is None:
                _kernel_parametrization = PROJECTIONS["identity"]
            elif kernel_parametrization in PROJECTIONS:
                _kernel_parametrization = PROJECTIONS[kernel_parametrization]
            elif callable(kernel_parametrization):
                _kernel_parametrization = kernel_parametrization
            else:
                raise NotImplementedError(f"{kernel_parametrization=} unknown")
            return _kernel_parametrization

        self._kernel_initialization = kernel_initialization_dispatch()
        self._kernel_parametrization = kernel_parametrization_dispatch()

        self.scalar_learnable = config["scalar_learnable"]
        self.scalar = nn.Parameter(
            torch.tensor(config["scalar"]), requires_grad=self.scalar_learnable
        )
        self.weight = nn.Parameter(self._kernel_initialization())
        with torch.no_grad():
            parametrized_kernel = self.kernel_parametrization(self.weight)
            self.register_buffer("kernel", parametrized_kernel, persistent=False)

    def kernel_initialization(self) -> Tensor:
        r"""Draw an initial kernel matrix (random or static)."""
        return self._kernel_initialization()

    @jit.export
    def kernel_parametrization(self, w: Tensor) -> Tensor:
        r"""Parametrize the Kernel, e.g. by projecting onto skew-symmetric matrices."""
        return self._kernel_parametrization(w)

    @jit.export
    def forward(self, dt: Tensor, x0: Tensor) -> Tensor:
        r"""Signature: ``[(...,), (..., d)] -> (..., d)]``.

        Parameters
        ----------
        dt: Tensor, shape=(...,)
            The time difference $t_1 - t_0$ between $x_0$ and $x̂$.
        x0:  Tensor, shape=(...,DIM)
            Time observed value at $t_0$.

        Returns
        -------
        xhat:  Tensor, shape=(...,DIM)
            The predicted value at $t_1$
        """
        self.kernel = self.scalar * self.kernel_parametrization(self.weight)
        Adt = torch.einsum("kl, ... -> ...kl", self.kernel, dt)
        expAdt = torch.linalg.matrix_exp(Adt)
        xhat = torch.einsum("...kl, ...l -> ...k", expAdt, x0)
        return xhat
