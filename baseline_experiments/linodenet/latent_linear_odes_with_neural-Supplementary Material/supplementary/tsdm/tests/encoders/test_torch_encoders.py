#!/usr/bin/env python
r"""Test the torch encoders."""

import logging

import numpy as np
import torch

from tsdm.encoders import PositionalEncoder
from tsdm.encoders.torch import PositionalEncoder as PositionalEncoder_Torch
from tsdm.encoders.torch import Time2Vec

__logger__ = logging.getLogger(__name__)


def test_PositionalEncoder() -> None:
    r"""Test the PositionalEncoder class."""
    scale = 1.23
    N = 10
    num_dim = 6
    t = np.sort(np.random.rand(N))
    logger = __logger__.getChild(PositionalEncoder.__name__)
    logger.info("Start Testing")

    try:
        encoder = PositionalEncoder(num_dim, scale)
        encoder.fit(None)
        logger.info("Initialization")
    except Exception as E:
        logger.error("Initialization")
        raise RuntimeError from E

    try:
        y = encoder.encode(t)
        logger.info("Forward")
    except Exception as E:
        logger.error("Forward")
        raise RuntimeError from E

    try:
        t_inverse = encoder.decode(y)
        logger.info("Inverse")
    except Exception as E:
        logger.error("Inverse")
        raise RuntimeError("Failed to run PositionalEncoder inverse") from E
    else:
        assert np.allclose(t_inverse, t), "inverse failed"

    logger.info("Finished Testing")


def test_PositionalEncoder_Torch() -> None:
    r"""Test the PositionalEncoder class."""
    scale = 1.23
    N = 10
    num_dim = 6
    t = torch.rand(N).sort().values
    logger = __logger__.getChild(PositionalEncoder_Torch.__name__)
    logger.info("Start Testing")

    try:
        encoder = PositionalEncoder_Torch(num_dim, scale=scale)
        logger.info("Initialization")
    except Exception as E:
        logger.error("Initialization")
        raise RuntimeError from E

    try:
        y = encoder(t)
        logger.info("Forward")
    except Exception as E:
        logger.error("Forward")
        raise RuntimeError from E

    try:
        t_inverse = encoder.inverse(y)
        logger.info("Inverse")
    except Exception as E:
        logger.error("Inverse")
        raise RuntimeError("Failed to run PositionalEncoder inverse") from E
    else:
        assert torch.allclose(t_inverse, t), "inverse failed"

    logger.info("Finished Testing")


def test_Time2Vec() -> None:
    r"""Test the Time2Vec class."""
    N = 10
    num_dim = 6
    t = torch.rand(N).sort().values

    logger = __logger__.getChild(Time2Vec.__name__)
    logger.info("Start Testing")

    try:
        encoder = Time2Vec(num_dim, "sin")
        logger.info("Initialization")
    except Exception as E:
        logger.error("Initialization")
        raise RuntimeError("Failed to initialize Time2Vec") from E

    try:
        y = encoder(t)
        logger.info("Forward")
    except Exception as E:
        logger.error("Forward")
        raise RuntimeError("Failed to run Time2Vec") from E

    try:
        t_inverse = encoder.inverse(y)
        logger.info("Inverse")
    except Exception as E:
        logger.error("Inverse")
        raise RuntimeError("Failed to run Time2Vec inverse") from E
    else:
        assert torch.allclose(t_inverse, t), "inverse failed"

    logger.info("Finished Testing")


def __main__():
    logging.basicConfig(level=logging.NOTSET)
    test_PositionalEncoder()
    test_Time2Vec()


if __name__ == "__main__":
    __main__()
