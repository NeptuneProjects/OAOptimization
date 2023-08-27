# -*- coding: utf-8 -*-

from dataclasses import dataclass
from itertools import product
from typing import Optional, Union

import numpy as np


@dataclass
class SearchParameter:
    name: str
    type: str
    value_type: Optional[str] = None
    bounds: Optional[list[Union[float, int]]] = None
    values: Optional[str] = None
    value: Optional[str] = None
    value_type: Optional[type] = None
    is_ordered: Optional[bool] = None
    # sort_values: Optional[bool] = None
    log_scale: Optional[bool] = False


@dataclass
class SearchParameterBounds:
    name: str
    lower_bound: float
    upper_bound: float
    relative: bool
    min_lower_bound: float = 0.0
    max_upper_bound: float = 0.0


@dataclass
class SearchSpace:
    parameters: list[SearchParameter]

    def __iter__(self) -> SearchParameter:
        return iter(self.__dict__.values())

    def to_dict(self) -> list[dict]:
        """
        Return a list of dictionaries containing the attributes of each
        parameter in the search space.

        :return: List of dictionaries containing the attributes of each
            parameter in the search space.
        :rtype: list[dict]
        """
        return [p.__dict__ for p in self.parameters]


@dataclass
class SearchSpaceBounds:
    bounds: list[SearchParameterBounds]

    def __getitem__(self, index):
        return self.bounds[index]


def get_parameter_grid_array(
    bounds: list[tuple], num_samples: Union[int, list]
) -> np.ndarray:
    """
    Given the bounds of a parameter space, return a flattened grid of parameter
    values.

    :param bounds: List of tuples specifying the bounds of each parameter.
    :type bounds: list[tuple]
    :param num_samples: Number of samples to take along each dimension of the
        grid. If an integer is given, the same number of samples will be taken
        for each dimension. If a list is given, each element corresponds to the
        number of samples to take along each dimension.
    :raises ValueError: Raise an error if a list of num_samples is given but
        the length of the list does not match the number of dimensions in the
        parameter space.
    :return: An (N x D) array of parameter values.
    :rtype: np.ndarray
    """
    d = len(bounds)
    if isinstance(num_samples, int):
        num_samples = [num_samples] * d

    if d != len(num_samples):
        raise ValueError(
            f"Parameter space has {d} dimensions, but grid specifications "
            f"given for only {len(num_samples)} dimensions."
        )

    return np.array(
        list(
            product(*[np.linspace(num=n, *bnd) for bnd, n in zip(bounds, num_samples)])
        )
    )


def get_parameterized_grid(
    search_space: SearchSpace, num_samples: Union[int, list]
) -> list[dict]:
    """
    Given a search space, return a list of parameterized grids.

    :param search_space: Search space to parameterize with grid.
    :type search_space: SearchSpace
    :param num_samples: Number of samples to take along each dimension of the
        grid. If an integer is given, the same number of samples will be taken
        for each dimension. If a list is given, each element corresponds to the
        number of samples to take along each dimension.
    :return: A list of parameterized grids.
    :rtype: list[dict]
    """
    bounds = [tuple(p.bounds) for p in search_space.parameters]
    param_array = get_parameter_grid_array(bounds=bounds, num_samples=num_samples)
    return [
        {p.name: i for i, p in zip(row, search_space.parameters)} for row in param_array
    ]
