#!/usr/bin/env python3

import gc
import logging
import os
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import update_wrapper, wraps
from inspect import Parameter, signature
from time import perf_counter_ns
from types import MethodType
from typing import Any, Dict, Final, Optional, TypedDict, TypeVar, Union, overload

import numpy as np
import torch
from torch import Tensor, jit, nn


@dataclass
class Config:
    input_size: int
    output_size: int
    latent_size: int = None

    def __post_init__(self):
        if self.latent_size is None:
            self.latent_size = self.input_size


conf = Config(2, 3)
bar: int = conf.latent_size
