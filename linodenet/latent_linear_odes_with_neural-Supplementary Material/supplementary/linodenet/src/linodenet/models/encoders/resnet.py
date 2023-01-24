r"""Residual Network Implementation.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
    "ResNetBlock",
]

from collections.abc import Iterable
from math import sqrt
from typing import Any, Optional, cast

import torch
from torch import Tensor, jit, nn
from torch.nn.functional import dropout

from linodenet.models.encoders.ft_transformer import (
    get_activation_fn,
    get_nonglu_activation_fn,
)
from linodenet.utils import (
    ReverseDense,
    ReZeroCell,
    deep_dict_update,
    initialize_from_config,
)


class _ResNet(nn.Module):
    r"""Residual Network."""

    def __init__(
        self,
        *,
        d_numerical: int,
        categories: Optional[list[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(
                            d, d_hidden * (2 if activation.endswith("glu") else 1)
                        ),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x_num: Tensor
        x_cat: Optional[Tensor]

        Returns
        -------
        Tensor
        """
        tensors = []
        if x_num is not None:
            tensors.append(x_num)
        if x_cat is not None:
            assert self.category_embeddings is not None, "No category embeddings!"
            assert self.category_offsets is not None, "No category offsets!"

            tensors.append(
                self.category_embeddings(
                    x_cat + self.category_offsets[None]  # type: ignore[index]
                ).view(x_cat.size(0), -1)
            )
        x = torch.cat(tensors, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = cast(dict[str, nn.Module], layer)
            z = x
            z = layer["norm"](z)
            z = layer["linear0"](z)
            z = self.main_activation(z)

            if self.hidden_dropout:
                z = dropout(z, self.hidden_dropout, self.training)

            z = layer["linear1"](z)

            if self.residual_dropout:
                z = dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        return x


class ResNetBlock(nn.Sequential):
    r"""Pre-activation ResNet block.

    References
    ----------
    - | Identity Mappings in Deep Residual Networks
      | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
      | European Conference on Computer Vision 2016
      | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_layers": 2,
        "layer": ReverseDense.HP,
        "layer_cfg": {},
        "rezero": True,
    }

    def __init__(self, *modules: nn.Module, **cfg: Any) -> None:

        config = deep_dict_update(self.HP, cfg)

        assert config["input_size"] is not None, "input_size is required!"

        layer = config["layer"]
        if layer["__name__"] == "Linear":
            layer["in_features"] = config["input_size"]
            layer["out_features"] = config["input_size"]
        if layer["__name__"] == "BatchNorm1d":
            layer["num_features"] = config["input_size"]
        else:
            layer["input_size"] = config["input_size"]
            layer["output_size"] = config["input_size"]

        layers: list[nn.Module] = list(modules)

        for _ in range(config["num_layers"]):
            module = initialize_from_config(config["layer"])
            # self.add_module(f"subblock{k}", module)
            layers.append(module)

        if config["rezero"]:
            layers.append(ReZeroCell())

        # self.subblocks = nn.Sequential(subblocks)
        super().__init__(*layers)


class ResNet(nn.ModuleList):
    r"""A ResNet model."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_blocks": 5,
        "block": ResNetBlock.HP,
    }

    def __init__(
        self, modules: Optional[Iterable[nn.Module]] = None, **cfg: Any
    ) -> None:
        config = deep_dict_update(self.HP, cfg)

        assert config["input_size"] is not None, "input_size is required!"

        # pass the input_size to the subblocks
        block = config["block"]
        if "input_size" in block:
            block["input_size"] = config["input_size"]

        blocks: list[nn.Module] = [] if modules is None else list(modules)

        for _ in range(config["num_blocks"]):
            module = initialize_from_config(config["block"])
            # self.add_module(f"block{k}", module)
            blocks.append(module)

        super().__init__(blocks)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass."""
        for block in self:
            x = x + block(x)
        return x
