#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""A module containing uninformed search strategies, including grid search,
random search, and quasi-random search.
"""

import pathlib
import sys
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
from oao.utilities import get_grid_parameters


def get_grid_samples(
    bounds: dict, num_samples: Union[int, list], seed: Optional[int] = None
) -> pd.DataFrame:
    """Returns samples at grid points from the bounded parameter space.

    :param bounds: Paramater space boundaries, defined in dictionary as
        key-value pairs.
    :type bounds: dict
    :param num_samples: A list of grid points in each dimension; if an
        integer is provided, equal grid points are assumed for all dimensions.
    :type num_samples: int, list
    :param seed: Unused, defaults to None
    :type seed: int, optional
    :return: Data frame containing points in parameter space, with each
        column corresponding to a parameter and rows corresponding to
        points in parameter space.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(get_grid_parameters(bounds, num_samples))


def get_latin_hypercube_samples(
    bounds: dict, num_samples: int, seed: Optional[int] = None
) -> pd.DataFrame:
    """Returns samples generated using Latin hypercube sampling from bounded
    parameter space.

    :param bounds: Paramater space boundaries, defined in dictionary as
        key-value pairs.
    :type bounds: dict
    :param num_samples: Number of samples to return.
    :type num_samples: int
    :param seed: Random seed specification, defaults to None
    :type seed: int, optional
    :return: Data frame containing points in parameter space, with each
        column corresponding to a parameter and rows corresponding to
        points in parameter space.
    :rtype: pd.DataFrame
    """
    sampler = qmc.LatinHypercube(d=len(bounds))
    samples = sampler.random(num_samples)
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    return pd.DataFrame(samples, columns=list(bounds.keys()))


def get_random_samples(
    bounds: dict, num_samples: int, seed: Optional[int] = None
) -> pd.DataFrame:
    """Returns randomly generated samples from bounded parameter space.

    :param bounds: Paramater space boundaries, defined in dictionary as
        key-value pairs.
    :type bounds: dict
    :param num_samples: Number of samples to return.
    :type num_samples: int
    :param seed: Random seed specification, defaults to None
    :type seed: int, optional
    :return: Data frame containing points in parameter space, with each
        column corresponding to a parameter and rows corresponding to
        points in parameter space.
    :rtype: pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for parameter, bound in bounds.items():
        samples[parameter] = rng.uniform(bound[0], bound[1], num_samples)
    return pd.DataFrame(samples)


def get_sobol_samples(
    bounds: dict, num_samples: int, seed: Optional[int] = None
) -> pd.DataFrame:
    """Returns samples generated using Sobol sampling from bounded parameter space.

    :param bounds: Paramater space boundaries, defined in dictionary as
        key-value pairs.
    :type bounds: dict
    :param num_samples: Number of samples to return.
    :type num_samples: int
    :param seed: Random seed specification, defaults to None
    :type seed: int, optional
    :return: Data frame containing points in parameter space, with each
        column corresponding to a parameter and rows corresponding to
        points in parameter space.
    :rtype: pd.DataFrame
    """
    sampler = qmc.Sobol(len(bounds), seed=seed)
    samples = sampler.random(num_samples)
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    return pd.DataFrame(samples, columns=list(bounds.keys()))
