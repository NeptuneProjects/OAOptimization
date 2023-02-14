#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""Module contains objective functions for testing optimization."""

from ax.utils.measurement import synthetic_functions as synth
from botorch.test_functions.synthetic import Griewank
import numpy as np

# from tritonoa.kraken import run_kraken
# from tritonoa.sp import beamformer


def evaluate_branin(parameters: dict) -> dict:
    """#TODO:_summary_

    :param parameters: _description_
    :type parameters: dict
    :return: _description_
    :rtype: dict
    """
    x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
    return {"branin": (synth.branin(x), 0.0)}


def evaluate_hartmann6(parameters: dict) -> dict:
    """#TODO:_summary_

    :param parameters: _description_
    :type parameters: dict
    :return: _description_
    :rtype: dict
    """
    x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
    return {"hartmann6": (synth.hartmann6(x), 0.0)}


def evaluate_griewank(parameters: dict) -> dict:
    x = np.array(parameters.get("x1"))[..., None]
    griewank = synth.from_botorch(Griewank(dim=1))
    return {"griewank": (griewank(x), 0.0)}


# def evaluate_kraken(parameters: dict) -> dict:
#     """#TODO:_summary_

#     :param parameters: _description_
#     :type parameters: dict
#     :return: _description_
#     :rtype: dict
#     """
#     K = parameters.pop("K")
#     p_rep = run_kraken(parameters)
#     objective_raw = beamformer(K, p_rep, atype="bartlett").item()
#     return {"bartlett": (objective_raw, 0.0)}
