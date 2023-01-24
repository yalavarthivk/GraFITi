#!/usr/bin/env python
r"""Test the iResNet components.

1. is the LinearContraction layer really a linear contraction?
2. is the iResNet really invertible via fixed-point iteration?
"""

import logging
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor

from linodenet.models import LinearContraction, iResNetBlock
from tsdm.linalg import scaled_norm
from tsdm.viz import visualize_distribution

__logger__ = logging.getLogger(__name__)  # noqa: E402


PATH = Path(__file__)
TEST_DIR = PATH.parent / "test_results" / PATH.stem
TEST_DIR.mkdir(parents=True, exist_ok=True)


def test_LinearContraction(
    num_sample: Optional[int] = None,
    dim_inputs: Optional[int] = None,
    dim_output: Optional[int] = None,
    make_plot: bool = False,
) -> None:
    r"""Test empirically if the LinearContraction really is a linear contraction.

    Parameters
    ----------
    num_sample: Optional[int] = None
        default: sample randomly from [1000, 2000, ..., 5000]
    dim_inputs: Optional[int] = None
        default: sample randomly from [2, 4, 8, , .., 128]
    dim_output: Optional[int] = None
            default: sample randomly from [2, 4, 8, , .., 128]
    make_plot: bool
    """
    __logger__.info(">>> Testing LinearContraction <<<")
    num_sample = num_sample or random.choice([1000 * k for k in range(1, 6)])
    dim_inputs = dim_inputs or random.choice([2**k for k in range(2, 8)])
    dim_output = dim_output or random.choice([2**k for k in range(2, 8)])
    extra_stats = {
        "Samples": f"{num_sample}",
        "Dim-in": f"{dim_inputs}",
        "Dim-out": f"{dim_output}",
    }
    __logger__.info("Configuration: %s", extra_stats)

    x = torch.randn(num_sample, dim_inputs)
    y = torch.randn(num_sample, dim_inputs)
    distances: Tensor = torch.cdist(x, y)

    model = LinearContraction(dim_inputs, dim_output)
    xhat = model(x)
    yhat = model(y)
    latent_distances: Tensor = torch.cdist(xhat, yhat)

    # Test whether contraction property holds
    assert torch.all(latent_distances <= distances)
    __logger__.info("LinearContraction passes test ✔ ")

    if not make_plot:
        return

    __logger__.info("LinearContraction generating figure")
    scaling_factor = (latent_distances / distances).flatten()

    # TODO: Switch to PGF backend with matplotlib 3.5 release
    fig, ax = plt.subplots(figsize=(5.5, 3.4), tight_layout=True)
    ax.set_title(r"LinearContraction -- Scaling Factor Distribution")
    ax.set_xlabel(
        r"$s(x, y) = \frac{\|\phi(x)-\phi(y)\|}{\|x-y\|}$ where "
        r"$x_i, y_j \overset{\text{i.i.d}}{\sim} \mathcal N(0, 1)$"
    )
    ax.set_ylabel(r"density $p(s \mid x, y)$")

    visualize_distribution(scaling_factor, ax=ax, extra_stats=extra_stats)

    fig.savefig(TEST_DIR / "LinearContraction_ScalingFactor.pdf")
    __logger__.info("LinearContraction all done")


@pytest.mark.flaky(reruns=3)
def test_iResNetBlock(
    num_sample: Optional[int] = None,
    dim_inputs: Optional[int] = None,
    dim_output: Optional[int] = None,
    maxiter: int = 20,
    make_plot: bool = False,
    quantiles: tuple[float, ...] = (0.5, 0.68, 0.95, 0.997),
    targets: tuple[float, ...] = (0.005, 0.005, 0.01, 0.01),
) -> None:
    r"""Test empirically whether the iResNetBlock is indeed invertible.

    Parameters
    ----------
    num_sample: Optional[int] = None
        default: sample randomly from [1000, 2000, ..., 10000]
    dim_inputs: Optional[int] = None
        default: sample randomly from [2, 4, 8, , .., 128]
    dim_output: Optional[int] = None
        default: sample randomly from [2, 4, 8, , .., 128]
    make_plot: bool
    quantiles: tuple[float, ...]
        The quantiles of the error distribution
    targets: tuple[float, ...]
        The target values for the quantiles of the error distribution
    """
    __logger__.info(">>> Testing iResNetBlock <<<")
    num_sample = num_sample or random.choice([1000 * k for k in range(1, 11)])
    dim_inputs = dim_inputs or random.choice([2**k for k in range(2, 8)])
    dim_output = dim_output or random.choice([2**k for k in range(2, 8)])
    extra_stats = {
        "Samples": f"{num_sample}",
        "Dim-in": f"{dim_inputs}",
        "Dim-out": f"{dim_output}",
        "maxiter": f"{maxiter}",
    }
    QUANTILES = torch.tensor(quantiles)
    TARGETS = torch.tensor(targets)

    __logger__.info("Configuration: %s", extra_stats)
    __logger__.info("QUANTILES: %s", QUANTILES)
    __logger__.info("TARGETS  : %s", TARGETS)

    with torch.no_grad():
        model = iResNetBlock(dim_inputs, hidden_size=dim_output, maxiter=maxiter)
        x = torch.randn(num_sample, dim_inputs)
        y = torch.randn(num_sample, dim_inputs)
        fx = model(x)
        xhat = model.inverse(fx)
        ify = model.inverse(y)
        yhat = model(ify)

    # Test if ϕ⁻¹∘ϕ=id, i.e. the right inverse is working
    forward_inverse_error = scaled_norm(x - xhat, axis=-1)
    forward_inverse_quantiles = torch.quantile(forward_inverse_error, QUANTILES)
    assert forward_inverse_error.shape == (num_sample,)
    assert (forward_inverse_quantiles <= TARGETS).all(), f"{forward_inverse_quantiles=}"
    __logger__.info("iResNetBlock satisfies ϕ⁻¹∘ϕ≈id ✔ ")
    __logger__.info("Quantiles: %s", forward_inverse_quantiles)

    # Test if ϕ∘ϕ⁻¹=id, i.e. the right inverse is working
    inverse_forward_error = scaled_norm(y - yhat, axis=-1)
    inverse_forward_quantiles = torch.quantile(forward_inverse_error, QUANTILES)
    assert inverse_forward_error.shape == (num_sample,)
    assert (inverse_forward_quantiles <= TARGETS).all(), f"{inverse_forward_quantiles=}"
    __logger__.info("iResNetBlock satisfies ϕ∘ϕ⁻¹≈id ✔ ")
    __logger__.info("Quantiles: %s", inverse_forward_quantiles)

    # Test if ϕ≠id, i.e. the forward map is different from the identity
    forward_difference = scaled_norm(x - fx, axis=-1)
    forward_quantiles = torch.quantile(forward_difference, 1 - QUANTILES)
    assert forward_difference.shape == (num_sample,)
    assert (forward_quantiles >= TARGETS).all(), f"{forward_quantiles}"
    __logger__.info("iResNetBlock satisfies ϕ≉id ✔ ")
    __logger__.info("Quantiles: %s", forward_quantiles)

    # Test if ϕ⁻¹≠id, i.e. the inverse map is different from an identity
    inverse_difference = scaled_norm(y - ify, axis=-1)
    inverse_quantiles = torch.quantile(inverse_difference, 1 - QUANTILES)
    assert inverse_difference.shape == (num_sample,)
    assert (inverse_quantiles >= TARGETS).all(), f"{inverse_quantiles}"
    __logger__.info("iResNetBlock satisfies ϕ⁻¹≉id ✔ ")
    __logger__.info("Quantiles: %s", inverse_quantiles)

    if not make_plot:
        return

    __logger__.info("iResNetBlock generating figure")
    fig, ax = plt.subplots(
        ncols=2, nrows=2, figsize=(8, 5), tight_layout=True, sharex="row", sharey="row"
    )

    visualize_distribution(forward_inverse_error, ax=ax[0, 0], extra_stats=extra_stats)
    visualize_distribution(inverse_forward_error, ax=ax[0, 1], extra_stats=extra_stats)
    visualize_distribution(forward_difference, ax=ax[1, 0], extra_stats=extra_stats)
    visualize_distribution(inverse_difference, ax=ax[1, 1], extra_stats=extra_stats)

    ax[0, 0].set_xlabel(
        r"$r_\text{left}(x) = \|x - \phi^{-1}(\phi(x))\|$  where $x_i \sim \mathcal N(0,1)$"
    )
    ax[0, 0].set_ylabel(r"density $p(r_\text{left} \mid x)$")
    ax[0, 1].set_xlabel(
        r"$r_\text{right}(y) = \|y - \phi(\phi^{-1}(y))\|$ where $y_j \sim \mathcal N(0,1)$"
    )
    ax[0, 1].set_ylabel(r"density $p(r_\text{right}\mid y)$")

    ax[1, 0].set_xlabel(
        r"$d_\text{left}(x) = \|x - \phi(x)\|$ where $x_i \sim \mathcal N(0,1)$"
    )
    ax[1, 0].set_ylabel(r"density $p(d_\text{left} \mid x)$")
    ax[1, 1].set_xlabel(
        r"$d_\text{right}(y) = \|y - \phi^{-1}(y)\|$ where $y_j \sim \mathcal N(0,1)$"
    )
    ax[1, 1].set_ylabel(r"density $p(d_\text{right} \mid y)$")
    fig.suptitle(TEST_DIR / "iResNetBlock -- Inversion Property", fontsize=16)
    fig.savefig(TEST_DIR / "iResNetBlock_inversion.pdf")
    __logger__.info("iResNetBlock all done")


def __main__():
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    __logger__.info("Testing LinearContraction started!")
    test_LinearContraction(make_plot=True)
    __logger__.info("Testing LinearContraction finished!")

    __logger__.info("Testing iResNetBlock started!")
    test_iResNetBlock(make_plot=True)
    __logger__.info("Testing iResNetBlock finished!")


if __name__ == "__main__":
    __main__()
