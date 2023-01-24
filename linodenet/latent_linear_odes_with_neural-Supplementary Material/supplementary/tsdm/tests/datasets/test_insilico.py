#!/usr/bin/env python
r"""Testing of Electricity dataset, as a token for the whole BaseDataset architecture."""

import logging
from copy import copy

from pandas import DataFrame

from tsdm.datasets import BaseDataset, InSilicoData
from tsdm.utils.decorators import timefun

__logger__ = logging.getLogger(__name__)


def test_caching():
    r"""Check if dataset caching works (should be way faster)."""
    # NOTE: this test must be executed first!!!

    ds = InSilicoData()
    __logger__.info("Testing caching of dataset %s", ds.__class__.__name__)
    _, pre_cache_time = timefun(lambda: ds.dataset, append=True)()
    _, post_cache_time = timefun(lambda: ds.dataset, append=True)()

    __logger__.info("%f, %f", pre_cache_time, post_cache_time)

    assert (
        100 * post_cache_time <= pre_cache_time
    ), f"{post_cache_time=}, {pre_cache_time=}"

    __logger__.info(
        "%s passes caching test \N{HEAVY CHECK MARK}", ds.__class__.__name__
    )


def test_attributes():
    r"""Test if all attributes are present."""
    ds = InSilicoData()
    base_attrs = copy(set(dir(ds)))
    attrs = {
        "BASE_URL",
        "RAWDATA_DIR",
        "DATASET_DIR",
        "clean",
        "dataset",
        "dataset_files",
        "download",
        "load",
    }

    assert attrs <= base_attrs, f"{attrs - base_attrs} missing!"
    assert isinstance(ds.dataset, DataFrame)
    assert issubclass(InSilicoData, BaseDataset)
    # assert isinstance(InSilicoData(), BaseDataset)

    assert hasattr(ds, "dataset")
    assert hasattr(ds, "load")
    assert hasattr(ds, "download")
    assert hasattr(ds, "clean")


def __main__():
    logging.basicConfig(level=logging.INFO)
    test_caching()
    test_attributes()


if __name__ == "__main__":
    __main__()
