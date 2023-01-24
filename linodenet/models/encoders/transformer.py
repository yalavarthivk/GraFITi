r"""Transformer based Encoder models."""

__all__ = [
    # Classes
    "TransformerEncoder",
    "Transformer",
]

from typing import Optional

from torch import Tensor, nn
from torch.nn import TransformerEncoder as _TransformerEncoder

from linodenet.utils import autojit, deep_dict_update, initialize_from

TransformerEncoder = autojit(_TransformerEncoder)
r"""TransformerEncoder: Transformer based Encoder model."""


class Transformer(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Parameters
    ----------
    encoder_layer: an instance of the TransformerEncoderLayer() class (required).
    num_layers: the number of sub-encoder-layers in the encoder (required).
    norm: the layer normalization component (optional).

    Examples
    --------
    ..code-block:: python
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        src = torch.rand(10, 32, 512)
        out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "num_layers": 6,
        # the layer normalization component (optional).
        "norm": None,
        "EncoderLayer": {
            "__name__": "TransformerEncoderLayer",
            # the number of expected features in the input (required).
            "d_model": 8,
            # the number of heads in the multi-head-attention models (required).
            "nhead": 8,
            # the dimension of the feedforward network model (default=2048).
            "dim_feedforward": 2048,
            # the dropout value (default=0.1).
            "dropout": 0.1,
            # the activation function of the intermediate layer.
            "activation": "relu",
            # the eps value in layer normalization components (default=1e-5).
            "layer_norm_eps": 1e-5,
            # If True, then the input and output tensors are provided as (batch, seq, feature).
            # Default: False.
            "batch_first": False,
            # if True, layer norm is done prior to attention and feedforward operations.
            # Otherwise it’s done after. Default: False (after).
            "norm_first": False,
        },
    }

    def __init__(self, **cfg):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        self.layers = nn.ModuleList(
            initialize_from(nn, **config["EncoderLayer"])
            for _ in range(config["num_layers"])
        )
        self.num_layers = config["num_layers"]
        self.norm = config["norm"]

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Parameters
        ----------
        src: the sequence to the encoder (required).
        mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        output = src

        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
