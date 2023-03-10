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
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import ObjectiveProperties
from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement, qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import torch

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
from oao.utilities import save_config

# The optimization requires three major components:
# 1. The search (parameter) space
# 2. An objective function
# 3. A search strategy

# LOW-LEVEL CONFIGS ===========================================================
SEED = 2009
N_RESTARTS = 40
N_SAMPLES = 1024
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)


# 1. SEARCH SPACE ==============================================================
parameters = [
    {
        "name": "x1",
        "type": "range",
        "bounds": [-20.0, 20.0],
        "value_type": "float",
        "log_scale": False,
    },
    # {
    #     "name": "x2",
    #     "type": "range",
    #     "bounds": [0.0, 15.0],
    #     "value_type": "float",
    # },
]

# 2. OBJECTIVE FUNCTION ========================================================
obj_func_parameters = {
    # "K": K, # Covariance matrix of measured data
    # "env_parameters": {} # Kraken parameters will go here
}

# 3. SEARCH STRATEGY ===========================================================
# Uninformed Search Configurations
# NUM_TRIALS = [3, 3]
# strategy = {
#     "loop_type": "sobol",
#     "num_trials": 10
# }

# Bayesian Search Configurations
# Sequential optimization
NUM_WARMUP = 5
NUM_TRIALS = 20
strategy = {
    "loop_type": "sequential",
    "num_trials": NUM_TRIALS,
    "generation_strategy": GenerationStrategy(
        [
            GenerationStep(
                model=Models.SOBOL,
                num_trials=NUM_WARMUP,
                max_parallelism=NUM_WARMUP,
                model_kwargs={"seed": SEED},
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=NUM_TRIALS - NUM_WARMUP,
                max_parallelism=None,
                model_kwargs={
                    "surrogate": Surrogate(
                        botorch_model_class=FixedNoiseGP,
                        mll_class=ExactMarginalLogLikelihood,
                    ),
                    "botorch_acqf_class": qExpectedImprovement,
                },
                model_gen_kwargs={
                    "model_gen_options": {
                        "optimizer_kwargs": {
                            "num_restarts": N_RESTARTS,
                            "raw_samples": N_SAMPLES,
                        }
                    }
                },
                should_deduplicate=False,
            )
        ]
    )
}

# Sequential greedy batch optimization
# NUM_WARMUP = 7
# NUM_TRIALS = 20
# strategy = {
#     "loop_type": "greedy_batch", # <--- This is where to specify loop
#     "num_trials": NUM_TRIALS,
#     "batch_size": 5,
#     "generation_strategy": GenerationStrategy(
#         [
#             GenerationStep(
#                 model=Models.SOBOL,
#                 num_trials=NUM_WARMUP,
#                 max_parallelism=NUM_WARMUP,
#                 # model_kwargs={"seed": SEED},
#             ),
#             GenerationStep(
#                 model=Models.BOTORCH_MODULAR,
#                 num_trials=NUM_TRIALS - NUM_WARMUP,
#                 max_parallelism=5,
#                 model_kwargs={
#                     "surrogate": Surrogate(
#                         botorch_model_class=SingleTaskGP,
#                         mll_class=ExactMarginalLogLikelihood
#                     ),
#                     "botorch_acqf_class": qExpectedImprovement, # qProbabilityOfImprovement | qExpectedImprovement
#                     "torch_device": DEVICE,
#                 },
#                 model_gen_kwargs={
#                     "model_gen_options": {
#                         "optimizer_kwargs": {
#                             "num_restarts": N_RESTARTS,
#                             "raw_samples": N_SAMPLES,
#                         }
#                     }
#                 }
#             ),
#         ]
#     ),
# }

# Batch optimization
# NUM_WARMUP = 7
# NUM_TRIALS = 20
# strategy = {
#     "loop_type": "batch",
#     "num_trials": NUM_TRIALS,
#     "batch_size": 5,
#     "generation_strategy": GenerationStrategy(
#         [
#             GenerationStep(
#                 model=Models.SOBOL,
#                 num_trials=NUM_WARMUP,
#                 max_parallelism=NUM_WARMUP,
#                 model_kwargs={"seed": SEED},
#             ),
#             GenerationStep(
#                 model=Models.BOTORCH_MODULAR,
#                 num_trials=NUM_TRIALS - NUM_WARMUP,
#                 max_parallelism=5,
#                 model_kwargs={
#                     "surrogate": Surrogate(
#                         botorch_model_class=SingleTaskGP,
#                         mll_class=ExactMarginalLogLikelihood
#                     ),
#                     "botorch_acqf_class": qExpectedImprovement,
#                     "torch_device": DEVICE
#                 },
                # model_gen_kwargs={
                #     "model_gen_options": {
                #         "optimizer_kwargs": {
                #             "num_restarts": N_RESTARTS,
                #             "raw_samples": N_SAMPLES
                #         }
                #     }
                # }
#             )
#         ]
#     ),
# }


# HIGH-LEVEL CONFIGS ===========================================================
experiment_name = "griewank_test_experiment"
experiment_kwargs = {
    "name": experiment_name,
    "parameters": parameters,
    "objectives": {"griewank": ObjectiveProperties(minimize=True)},
}

# Dictionary for controlling evaluation of GP and Acq Function during training
# evaluation_config = {"num_test_points": 100}
evaluation_config = None

config = {
    "experiment_kwargs": experiment_kwargs,
    "obj_func_parameters": obj_func_parameters,
    # "num_trials": NUM_TRIALS,
    "seed": SEED,
    "strategy": strategy,
    "evaluation_config": evaluation_config,
}

if __name__ == "__main__":
    save_config("scripts/config.json", config)


# test = config["strategy"]["generation_strategy"]._steps[1].model_kwargs["seed"]
# print(test)

# for step in config["strategy"]["generation_strategy"]._steps:
#     step.model_kwargs["seed"] = 1234

# test = config["strategy"]["generation_strategy"]._steps[1].model_kwargs["seed"]
# print(test)
