#!/usr/bin/env python
r"""Test converters to masked format etc."""

import logging

from pandas import Series, date_range, testing

from tsdm.encoders import DateTimeEncoder

__logger__ = logging.getLogger(__name__)


def test_datetime_encoder() -> None:
    r"""Test whether the encoder is reversible."""
    __logger__.info("Testing %s started!", DateTimeEncoder.__name__)

    # test Index
    time = date_range("2020-01-01", "2021-01-01", freq="1d")
    encoder = DateTimeEncoder()
    encoder.fit(time)
    encoded = encoder.encode(time)
    decoded = encoder.decode(encoded)
    testing.assert_index_equal(time, decoded)

    # test Series
    time = Series(time)
    encoder = DateTimeEncoder()
    encoder.fit(time)
    encoded = encoder.encode(time)
    decoded = encoder.decode(encoded)
    testing.assert_series_equal(time, decoded)
    __logger__.info("Testing %s finished!", DateTimeEncoder.__name__)


def __main__() -> None:
    logging.basicConfig(level=logging.INFO)
    test_datetime_encoder()


if __name__ == "__main__":
    __main__()
