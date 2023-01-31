#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""This module contains optimization classes comprising the core 
machinery of the optimization package. The `Optimizer` class is inherited
by `BayesianOptimizer`, which contains the Bayesian optimization loop, 
and `UninformedOptimization`, which contains a loop for implementing
various types of non-Bayesian search.
"""

import logging
import pathlib
import sys
from typing import Optional, Union
import warnings

from ax.service.ax_client import AxClient
import numpy as np
import torch

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

from oao.common import UNINFORMED_STRATEGIES
from oao.optim import uninformed
from oao.utilities import get_test_features

logger = logging.getLogger(__name__)


class Optimizer:
    """Base class for optimization strategies.

    :return: _description_
    :rtype: _type_
    """

    verbose_logging = True

    def __init__(
        self,
        objective,
        strategy: Union[int, dict],
        obj_func_parameters: Optional[dict] = {},
    ):
        self.objective = objective
        self.obj_func_parameters = obj_func_parameters
        self.strategy = strategy

    @staticmethod
    def get_bounds_from_parameters(parameters: dict) -> dict:
        """Extracts the bounds of the parameter space from a `parameters`
        dictionary.

        :param parameters: Search space parameters.
        :type parameters: dict
        :return: Boundary of the search space.
        :rtype: dict
        """
        return {d["name"]: d["bounds"] for d in parameters}

    def initialize_run(
        self,
        experiment_kwargs,
        evaluation_config=None,
        seed: int=None,
        *args,
        **kwargs,
    ):
        """#TODO:_summary_

        :param experiment_kwargs: _description_
        :type experiment_kwargs: _type_
        :param num_trials: _description_
        :type num_trials: _type_
        :param evaluation_config: _description_, defaults to None
        :type evaluation_config: _type_, optional
        :param seed: _description_, defaults to None
        :type seed: _type_, optional
        """
        self.experiment_kwargs = experiment_kwargs
        self.seed = seed
        torch.manual_seed(self.seed)
        self.num_trials = self.strategy["num_trials"]
        if self.strategy["loop_type"] in UNINFORMED_STRATEGIES:
            self.generation_strategy = None
        else:
            self.generation_strategy = self.strategy["generation_strategy"]

        self.ax_client = AxClient(
            generation_strategy=self.generation_strategy,
            verbose_logging=self.verbose_logging,
        )
        self.ax_client.create_experiment(**experiment_kwargs)

        self.evaluation_config = evaluation_config
        self.bounds = self.get_bounds_from_parameters(
            experiment_kwargs.get("parameters")
        )


class BayesianOptimizer(Optimizer):
    """#TODO:_summary_

    :param Optimizer: _description_
    :type Optimizer: _type_
    """

    def __init__(
        self,
        objective,
        strategy: dict,
        obj_func_parameters: Optional[dict] = {},
    ):
        super().__init__(objective, strategy, obj_func_parameters)

    def run(
        self,
        experiment_kwargs,
        evaluation_config=None,
        seed=None,
        *args,
        **kwargs,
    ):
        """#TODO:_summary_

        :param experiment_kwargs: _description_
        :type experiment_kwargs: _type_
        :param num_trials: _description_
        :type num_trials: _type_
        :param evaluation_config: _description_, defaults to None
        :type evaluation_config: _type_, optional
        :param seed: _description_, defaults to None
        :type seed: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.initialize_run(experiment_kwargs, evaluation_config, seed, *args, **kwargs)
            self._run_loop()
        return self.ax_client.get_trials_data_frame()

    def _run_loop(self):
        if self.evaluation_config is not None:
            self.test_features = get_test_features(self.bounds, 10)

        if self.strategy["loop_type"] == "sequential":
            return self._run_sequential_loop()
        elif self.strategy["loop_type"] == "batch":
            return self._run_batch_loop()
        elif self.strategy["loop_type"] == "greedy_batch":
            return self._run_greedybatch_loop()

    def _run_batch_loop(self):
        # Batch optimization loop
        num_warmup = self.ax_client.generation_strategy._steps[0].num_trials
        for _ in range(num_warmup):
            params, trial_index = self.ax_client.get_next_trial()
            raw_data = self.objective(params | self.obj_func_parameters)
            self.ax_client.complete_trial(trial_index, raw_data)

        # Run optimization
        # logger.debug("Running optimization loop.")
        num_optim = self.num_trials - num_warmup
        num_batches = np.ceil(num_optim / self.strategy["batch_size"]).astype(int)

        optim_num = 0
        for batch_num in range(num_batches):
            batch_size = (
                self.strategy["batch_size"]
                if batch_num < num_batches - 1
                else num_optim - batch_num * self.strategy["batch_size"]
            )
            # TODO: Investigate why GP isn't updating.
            gen_run = self.ax_client._gen_new_generator_run(n=batch_size)
            for arm in gen_run.arms:
                if optim_num == num_optim:
                    break
                params, trial_index = self.ax_client.attach_trial(arm.parameters)
                results = self.objective(params | self.obj_func_parameters)
                self.ax_client.complete_trial(trial_index, results)
                optim_num += 1

    def _run_greedybatch_loop(self):
        # Run warmup
        num_warmup = self.ax_client.generation_strategy._steps[0].num_trials
        for _ in range(num_warmup):
            params, trial_index = self.ax_client.get_next_trial()
            raw_data = self.objective(params | self.obj_func_parameters)
            self.ax_client.complete_trial(trial_index, raw_data)

        # Run optimization
        num_optim = self.num_trials - num_warmup
        num_batches = np.ceil(num_optim / self.strategy["batch_size"]).astype(int)

        for batch_num in range(num_batches):
            batch_size = (
                self.strategy["batch_size"]
                if batch_num < num_batches - 1
                else num_optim - batch_num * self.strategy["batch_size"]
            )

            trials_dict, _ = self.ax_client.get_next_trials(max_trials=batch_size)

            results = {
                trial_index: self.objective(params | self.obj_func_parameters)
                for trial_index, params in trials_dict.items()
            }

            [
                self.ax_client.complete_trial(trial_index, results.get(trial_index))
                for trial_index in results
            ]

    def _run_sequential_loop(self):
        for _ in range(self.num_trials):
            params, trial_index = self.ax_client.get_next_trial()
            results = self.objective(params | self.obj_func_parameters)
            self.ax_client.complete_trial(trial_index, results)

            # Evaluate acquisition function and GP model
            if (self.ax_client.generation_strategy.current_step.index != 1) and (
                self.evaluation_config is not None
            ):
                df = self.evaluate_model_and_acquisition(trial_index)

    def evaluate_model_and_acquisition(self, trial_index):
        """#TODO:_summary_

        :param trial_index: _description_
        :type trial_index: _type_
        """
        model = self.ax_client.generation_strategy.model
        y_test = model.predict(self.test_features)
        alpha = model.evaluate_acquisition_function(self.test_features)
        # TODO: tabularize eval results by trial index
        return


class UninformedOptimizer(Optimizer):
    """#TODO:_summary_

    :param Optimizer: _description_
    :type Optimizer: _type_
    """

    def __init__(
        self,
        objective,
        strategy: Union[str, dict],
        obj_func_parameters: Optional[dict] = {},
    ):
        super().__init__(objective, strategy, obj_func_parameters)

    def run(self, experiment_kwargs, seed=None, *args, **kwargs):
        """#TODO:_summary_

        :param experiment_kwargs: _description_
        :type experiment_kwargs: _type_
        :param num_trials: _description_
        :type num_trials: _type_
        :param seed: _description_, defaults to None
        :type seed: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        self.initialize_run(experiment_kwargs, seed, *args, **kwargs)
        self.search_strategy = self._get_search_strategy()
        self._run_loop()
        return self.ax_client.get_trials_data_frame()

    def _get_search_strategy(self):
        if self.strategy["loop_type"] == "grid":
            logger.info("Generating samples using grid sampling.")
            return uninformed.get_grid_samples
        elif self.strategy["loop_type"] == "lhs":
            logger.info("Generating samples using Latin hypercube sampling.")
            return uninformed.get_latin_hypercube_samples
        elif self.strategy["loop_type"] == "random":
            logger.info("Generating samples using random search.")
            return uninformed.get_random_samples
        elif self.strategy["loop_type"] == "sobol":
            logger.info("Generating samples using Sobol sequence sampling.")
            return uninformed.get_sobol_samples
        else:
            return ValueError(
                f"Wrong loop type ({self.strategy['loop_type']}) specified."
            )

    def _run_loop(self):
        df = self.search_strategy(self.bounds, self.num_trials, self.seed)
        for parameters in df.to_dict("records"):
            params, trial_index = self.ax_client.attach_trial(parameters)
            self.ax_client.complete_trial(
                trial_index, self.objective(params | self.obj_func_parameters)
            )
