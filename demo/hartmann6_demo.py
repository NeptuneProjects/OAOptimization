#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import time

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from oao.objective import NoiselessFormattedObjective
from oao.optimizer import BayesianOptimization, GridSearch, QuasiRandom
from oao.results import get_results
from oao.space import SearchParameter, SearchSpace
from oao.strategy import GridStrategy


class Hartmann6Objective:
    """Custom objective function for the Hartmann6 function."""

    def evaluate(self, parameters) -> dict:
        x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
        time.sleep(1.0)
        return hartmann6(x)


def main():
    # Instantiate the objective function & wrap in a formatting class to
    # properly handle output.
    h6 = Hartmann6Objective()
    objective = NoiselessFormattedObjective(h6, "hartmann6", {"minimize": True})

    # Define the search space.
    search_space = [
        {"name": f"x{i + 1}", "type": "range", "bounds": [-2.0, 2.0]} for i in range(6)
    ]
    space = SearchSpace([SearchParameter(**d) for d in search_space])

    ## Bayesian Optimization ===========================================
    # Define the generation strategy.
    gs = GenerationStrategy(
        [
            GenerationStep(
                model=Models.SOBOL,
                num_trials=64,
                max_parallelism=16,
                model_kwargs={"seed": 0},
            ),
            GenerationStep(
                model=Models.GPEI,
                num_trials=16,
                max_parallelism=4,
            ),
        ]
    )
    # Instantiate and run the optimizers.
    opt_bo = BayesianOptimization(
        objective=objective,
        search_space=space,
        strategy=gs,
    )
    opt_bo.run(name="demo_bo")

    ## Grid Search =====================================================
    # Define the generation strategy.
    gs = GridStrategy(num_trials=2, max_parallelism=4)
    # Instantiate and run the optimizers.
    opt_gs = GridSearch(
        objective=objective,
        search_space=space,
        strategy=gs,
    )
    opt_gs.run(name="demo_gs")

    ## Quasi-Random Search =============================================
    # Define the generation strategy.
    gs = GenerationStrategy(
        [
            GenerationStep(
                model=Models.SOBOL,
                num_trials=64,
                max_parallelism=16,
                model_kwargs={"seed": 0},
            ),
        ]
    )
    # Instantiate and run the optimizers.
    opt_qr = QuasiRandom(
        objective=objective,
        search_space=space,
        strategy=gs,
    )
    opt_qr.run(name="demo_bo")

    ## Save the results to CSV files. ==================================
    get_results(
        opt_bo.client,
        times=opt_bo.batch_execution_times,
        minimize=objective.properties.minimize,
    ).to_csv("demo/results_bo.csv")

    get_results(
        opt_gs.client,
        times=opt_gs.batch_execution_times,
        minimize=objective.properties.minimize,
    ).to_csv("demo/results_gs.csv")

    get_results(
        opt_qr.client,
        times=opt_qr.batch_execution_times,
        minimize=objective.properties.minimize,
    ).to_csv("demo/results_qr.csv")

    # Save the clients to JSON files.
    opt_bo.client.save_to_json_file("demo/experiment_bo.json")
    opt_gs.client.save_to_json_file("demo/experiment_gs.json")
    opt_qr.client.save_to_json_file("demo/experiment_qr.json")

    # Load the results from the JSON file and render the optimization trace.
    restored_client_bo = AxClient.load_from_json_file("demo/experiment_bo.json")
    restored_client_gs = AxClient.load_from_json_file("demo/experiment_gs.json")
    restored_client_qr = AxClient.load_from_json_file("demo/experiment_qr.json")
    render(restored_client_bo.get_optimization_trace(objective_optimum=hartmann6.fmin))
    render(restored_client_gs.get_optimization_trace(objective_optimum=hartmann6.fmin))
    render(restored_client_qr.get_optimization_trace(objective_optimum=hartmann6.fmin))


if __name__ == "__main__":
    main()
