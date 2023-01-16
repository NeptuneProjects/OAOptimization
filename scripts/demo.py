#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

import argparse
import pathlib
import sys

from ax.utils.measurement.synthetic_functions import branin
import numpy as np

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
from oao.handler import Handler

K = None


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
    return {"branin": (branin(x), 0.0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reads optimization configuration files and submits for execution."
    )
    parser.add_argument(
        "-source", type=str, help="Path to source file.", default="config.json"
    )
    parser.add_argument(
        "-destination", type=str, help="Path to destination file.", default="."
    )
    args = parser.parse_args()

    Handler(args).run()
