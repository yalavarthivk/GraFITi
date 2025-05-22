#!/usr/bin/env python
r"""Test converters to masked format etc."""

import logging

from pandas import NA, DataFrame, testing

from tsdm.encoders.functional import make_masked_format

__logger__ = logging.getLogger(__name__)


def test_make_masked_format():
    r"""Using example taken from Figure 2 in [1].

    References
    ----------
    1. Recurrent Neural Networks for Multivariate Time Series with Missing Values
       Che et al., Nature 2017
    """
    __logger__.info("Testing %s started!", make_masked_format.__name__)

    x = [[47, 49, NA, 40, NA, 43, 55], [NA, 15, 14, NA, NA, NA, 15]]
    t = [0, 0.1, 0.6, 1.6, 2.2, 2.5, 3.1]

    d = [[0.0, 0.1, 0.5, 1.5, 0.6, 0.9, 0.6], [0.0, 0.1, 0.5, 1.0, 1.6, 1.9, 2.5]]

    m = [[1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0, 1]]

    data = DataFrame(x, columns=t).T
    mask = DataFrame(m, columns=t).T
    diff = DataFrame(d, columns=t).T

    x, m, d = make_masked_format(data)
    testing.assert_frame_equal(x, data, check_dtype=False)
    testing.assert_frame_equal(m, mask, check_dtype=False)
    testing.assert_frame_equal(d, diff, check_dtype=False)
    __logger__.info("Testing %s finished!", make_masked_format.__name__)


def __main__():
    logging.basicConfig(level=logging.INFO)
    test_make_masked_format()


if __name__ == "__main__":
    __main__()
