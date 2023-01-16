#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""A module containing constants and values used throughout the package."""

import logging

LOG_FORMAT = logging.Formatter(
    fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
UNINFORMED_STRATEGIES = ["grid", "lhs", "random", "sobol"]
