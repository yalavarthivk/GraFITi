#!/usr/bin/env python
r"""Test whether `LinODEnet` is forward stable."""

import torch

from linodenet.models import LinODEnet, ResNet, embeddings, filters, system


def _implement_it():
    # Test forward stability.

    N = 1000
    D = 5
    L = 32

    MODEL_CONFIG = {
        "__name__": "LinODEnet",
        "input_size": D,
        "hidden_size": L,
        "embedding_type": "concat",
        "Filter": filters.SequentialFilter.HP,
        "System": system.LinODECell.HP | {"kernel_initialization": "skew-symmetric"},
        "Encoder": ResNet.HP,
        "Decoder": ResNet.HP,
        "Embedding": embeddings.ConcatEmbedding.HP,
    }
    print(MODEL_CONFIG)
    T = torch.randn(N)
    X = torch.randn(N, D)
    model = LinODEnet(D, L)
    model(T, X)


if __name__ == "__main__":
    # main program
    pass
