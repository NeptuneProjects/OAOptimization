#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

import pathlib
import sys

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import ObjectiveProperties
import torch

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
from oao.utilities import save_config

# The optimization requires three major components:
# 1. The search (parameter) space
# 2. An objective function
# 3. A search strategy

# LOW-LEVEL CONFIGS ===========================================================
SEED = 2009
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. SEARCH SPACE ==============================================================
parameters = [
    {
        "name": "x1",
        "type": "range",
        "bounds": [-5.0, 10.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "x2",
        "type": "range",
        "bounds": [0.0, 15.0],
        "value_type": "float",
    },
]

# 2. OBJECTIVE FUNCTION ========================================================
obj_func_parameters = {
    # "K": K, # Covariance matrix of measured data
    # "env_parameters": {} # Kraken parameters will go here
}

# 3. SEARCH STRATEGY ===========================================================
# Uninformed Search Configurations
num_trials = 10
strategy = "lhs"

# Bayesian Search Configurations
NUM_WARMUP = 7
num_trials = 20
strategy = {
    "loop_type": "sequential",
    "generation_strategy": GenerationStrategy(
        [
            GenerationStep(
                model=Models.SOBOL,
                num_trials=NUM_WARMUP,
                max_parallelism=NUM_WARMUP,
                model_kwargs={"seed": SEED},
            ),
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,
                max_parallelism=None,
                model_kwargs={"torch_device": DEVICE},
            ),
        ]
    )
}


# HIGH-LEVEL CONFIGS ===========================================================
experiment_name = "branin_test_experiment"
experiment_kwargs = {
    "name": experiment_name,
    "parameters": parameters,
    "objectives": {"branin": ObjectiveProperties(minimize=True)},
}

# Dictionary for controlling evaluation of GP and Acq Function during training
# evaluation_config = {"num_samples": 3}
evaluation_config = None

config = {
    "experiment_kwargs": experiment_kwargs,
    "obj_func_parameters": obj_func_parameters,
    "num_trials": num_trials,
    "seed": SEED,
    "strategy": strategy,
    "evaluation_config": evaluation_config,
}

if __name__ == "__main__":
    save_config("scripts/config.json", config)
