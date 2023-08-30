#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
__version__ = "0.2.0"

from ax.storage.botorch_modular_registry import ACQUISITION_FUNCTION_REGISTRY
from ax.storage.botorch_modular_registry import REVERSE_ACQUISITION_FUNCTION_REGISTRY
from botorch.acquisition import (
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    ProbabilityOfImprovement,
)

ACQUISITION_FUNCTION_REGISTRY.update(
    {
        ProbabilityOfImprovement: "ProbabilityOfImprovement",
        qProbabilityOfImprovement: "qProbabilityOfImprovement",
        qUpperConfidenceBound: "qUpperConfidenceBound",
    }
)
REVERSE_ACQUISITION_FUNCTION_REGISTRY.update(
    {
        "ProbabilityOfImprovement": ProbabilityOfImprovement,
        "qProbabilityOfImprovement": qProbabilityOfImprovement,
        "qUpperConfidenceBound": qUpperConfidenceBound,
    }
)
