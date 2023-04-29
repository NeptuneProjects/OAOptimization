#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
from typing import Union

import numpy as np


def get_grid_parameters(
    bounds: list[tuple], num_samples: Union[int, list]
) -> np.ndarray:
    d = len(bounds)
    if isinstance(num_samples, int):
        num_samples = [num_samples for _ in bounds]

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


if __name__ == "__main__":
    bounds = [(0, 1), (20, 30)]
    n = [20, 40]
    print(get_grid_parameters(bounds, n))
