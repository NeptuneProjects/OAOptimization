# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""oao (Ocean Acoustics Optimization) is a Python package for estimating
acoustic parameters in the ocean using uninformed search methods and 
Bayesian optimization.
"""
__version__ = "0.0.3"

import logging
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
import oao.common

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ax_logger = logging.getLogger("ax")
while ax_logger.hasHandlers():
    ax_logger.removeHandler(ax_logger.handlers[0])
ax_logger.setLevel(logging.CRITICAL)

sh = logging.StreamHandler()
sh.setLevel(logging.CRITICAL)
sh.setFormatter(oao.common.LOG_FORMAT)

logger.addHandler(sh)
ax_logger.addHandler(sh)
