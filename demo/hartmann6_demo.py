#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render
from botorch.acquisition import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from oao.objective import NoiselessFormattedObjective
from oao.optimizer import BayesianOptimization, GridSearch
from oao.space import SearchParameter, SearchSpace


class Hartmann6Objective:
    """Custom objective function for the Hartmann6 function."""

    def evaluate(self, parameters) -> dict:
        x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
        return hartmann6(x)


def main():
    # Instantiate the objective function & wrap in a formatting class to
    # properly handle output.
    h6 = Hartmann6Objective()
    objective = NoiselessFormattedObjective(h6, "hartmann6", {"minimize": True})

    # Define the generation strategy.
    gs = GenerationStrategy(
        [
            GenerationStep(
                model=Models.SOBOL,
                num_trials=16,
                max_parallelism=4,
                model_kwargs={"seed": 0},
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=16,
                max_parallelism=4,
                model_kwargs={
                    "surrogate": Surrogate(
                        botorch_model_class=SingleTaskGP,
                        mll_class=ExactMarginalLogLikelihood,
                    ),
                    "botorch_acqf_class": qExpectedImprovement,
                },
                model_gen_kwargs={
                    "model_gen_options": {
                        "optimizer_kwargs": {
                            "num_restarts": 40,
                            "raw_samples": 1024,
                        }
                    }
                },
            ),
        ]
    )

    # Define the search space.
    search_space = [
        {"name": f"x{i + 1}", "type": "range", "bounds": [0.0, 1.0]} for i in range(6)
    ]
    space = SearchSpace([SearchParameter(**d) for d in search_space])

    # Instantiate and run the optimizers.
    opt_bo = BayesianOptimization(
        objective=objective,
        search_space=space,
        strategy=gs,
    )
    opt_bo.run(name="demo_bo")

    opt_gs = GridSearch(
        objective=objective,
        search_space=space,
        num_trials=4,
    )
    opt_gs.run(name="demo_gs")

    # Save the results to a JSON file.
    opt_bo.client.save_to_json_file("demo/experiment_bo.json")
    opt_gs.client.save_to_json_file("demo/experiment_gs.json")

    # Load the results from the JSON file and render the optimization trace.
    restored_client_bo = AxClient.load_from_json_file("demo/experiment_bo.json")
    restored_client_gs = AxClient.load_from_json_file("demo/experiment_gs.json")
    render(restored_client_bo.get_optimization_trace(objective_optimum=hartmann6.fmin))
    render(restored_client_gs.get_optimization_trace(objective_optimum=hartmann6.fmin))


if __name__ == "__main__":
    main()