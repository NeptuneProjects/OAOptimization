#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""Module with package utilities."""

import json
from pathlib import PurePath
from typing import Union

from ax.core.observation import ObservationFeatures
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.ax_client import ObjectiveProperties
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
import numpy as np
from sklearn.model_selection import ParameterGrid


class ConfigEncoder(json.JSONEncoder):
    """#TODO:_summary_

    :param json: _description_
    :type json: _type_
    """
    def default(self, o):
        """#TODO:_summary_

        :param o: _description_
        :type o: _type_
        :return: _description_
        :rtype: _type_
        """
        if isinstance(o, (ObjectiveProperties)):
            return {"__type": "ObjectiveProperties", "kwargs": o.__dict__}
        elif isinstance(o, (GenerationStrategy)):
            return object_to_json(o)
        else:
            return json.JSONEncoder.encode(self, o)


class ConfigDecoder(json.JSONDecoder):
    """#TODO:_summary_

    :param json: _description_
    :type json: _type_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """#TODO:_summary_

        :param obj: _description_
        :type obj: _type_
        :return: _description_
        :rtype: _type_
        """
        if "__type" not in obj:
            return obj
        obj_type = obj["__type"]
        if obj_type == "ObjectiveProperties":
            return ObjectiveProperties(**obj["kwargs"])
        elif obj_type == "GenerationStrategy":
            return object_from_json(obj)
        else:
            return obj


def save_config(path: Union[str, PurePath], config: dict):
    """#TODO:_summary_

    :param path: _description_
    :type path: Union[str, PurePath]
    :param config: _description_
    :type config: dict
    """
    with open(path, "w") as fp:
        json.dump(config, fp, cls=ConfigEncoder, indent=4)


def load_config(path: Union[str, PurePath]) -> dict:
    """#TODO:_summary_

    :param path: _description_
    :type path: Union[str, PurePath]
    :return: _description_
    :rtype: dict
    """
    with open(path, "r") as fp:
        return json.load(fp, cls=ConfigDecoder)


def get_grid_parameters(bounds: dict, num_samples: Union[int, list]) -> list:
    """Returns parameters at grid points from the bounded parameter space.

    :param bounds: Paramater space boundaries, defined in dictionary as
        key-value pairs.
    :type bounds: dict
    :param num_samples: A list of grid points in each dimension; if an
        integer is provided, equal grid points are assumed for all dimensions.
    :type num_samples: int, list
    :param seed: Unused, defaults to None
    :type seed: int, optional
    :return: List of dictionaries containing points in parameter space,
        with each dictionary containing key-value pairs of parameter name
        and value.
    :rtype: list
    """
    d = len(bounds)
    if isinstance(num_samples, int):
        num_samples = [num_samples for _ in bounds.items()]

    if d != len(num_samples):
        raise ValueError(
            f"Parameter space has {d} dimensions, but grid specifications "
            f"given for only {len(num_samples)} dimensions."
        )

    param_grid = {}
    for (name, bnd), n in zip(bounds.items(), num_samples):
        param_grid[name] = np.linspace(bnd[0], bnd[1], num=n)

    return list(ParameterGrid(param_grid))


def get_test_features(bounds: dict, num_samples: Union[int, list]) -> list:
    """#TODO:_summary_

    :param bounds: _description_
    :type bounds: dict
    :param num_samples: _description_
    :type num_samples: Union[int, list]
    :return: _description_
    :rtype: list
    """
    grid_parameters = get_grid_parameters(bounds, num_samples)
    return [ObservationFeatures(p) for p in grid_parameters]
