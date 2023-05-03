#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import time
from typing import Optional, Union
import warnings

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
import numpy as np

from oao.objective import Objective
from oao.space import SearchSpace, get_parameterized_grid


class Optimizer(ABC):
    """Base class for optimization strategies."""

    @abstractmethod
    def run(self, name: str = None) -> None:
        """Run the optimization strategy."""
        pass

    @abstractmethod
    def _create_experiment(self, name: str) -> None:
        """Create an Ax experiment."""
        pass


class BayesianOptimization(Optimizer):
    def __init__(
        self,
        objective: Objective,
        search_space: SearchSpace,
        strategy: GenerationStrategy,
        random_seed: Optional[int] = None,
        monitor: Optional[callable] = None,
    ) -> None:
        """
        Initialize Bayesian optimization strategy.

        :param objective: Objective function for optimization; must be callable.
        :type objective: Objective
        :param search_space: Search space definition.
        :type search_space: SearchSpace
        :param strategy: Specify the generation strategy for optimization.
        :type strategy: GenerationStrategy
        :param random_seed: Set the random seed, defaults to None
        :type random_seed: Optional[int], optional
        :param monitor: Callable to log metrics, defaults to None
        :type monitor: Optional[callable], optional
        """
        self.objective = objective
        self.search_space = search_space
        self.strategy = strategy
        self.random_seed = random_seed
        self.monitor = monitor
        self.client = AxClient(generation_strategy=strategy, random_seed=random_seed)
        self.batch_execution_times = []

    @staticmethod
    def get_batch_size(
        batch_number: int, num_batches: int, batch_size: int, num_trials: int
    ) -> int:
        """
        Check the number of trials already completed given the current
        batch index and ensure that the last batch does not exceed the
        maximum number of trials.

        :param batch_number: Index of the current batch.
        :type batch_number: int
        :param num_batches: Total number of batches.
        :type num_batches: int
        :param batch_size: Number of trials within each batch.
        :type batch_size: int
        :param num_trials: Total number of trials in optimization run.
        :type num_trials: int
        :return: Number of trials for the current batch.
        :rtype: int
        """
        if batch_number < num_batches - 1:
            return batch_size
        return num_trials - batch_number * batch_size

    @staticmethod
    def get_num_batches(num_trials: int, batch_size: int) -> int:
        """
        Calculate the number of batches given the total number of trials
        and batch size.

        :param num_trials: Total number of trials in optimization run.
        :type num_trials: int
        :param batch_size: Number of trials within each batch.
        :type batch_size: int
        :return: Number of batches.
        :rtype: int
        """
        return np.ceil(num_trials / batch_size).astype(int)

    def run(self, name: str = None) -> None:
        """Runs the optimization using provided configurations."""
        self._create_experiment(name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._run_steps()

    def _create_experiment(self, name: str) -> None:
        """
        Create an Ax experiment.

        :param name: Name of the experiment.
        :type name: str
        """
        self.client.create_experiment(
            name=name,
            parameters=self.search_space.to_dict(),
            objectives={self.objective.name: self.objective.properties},
        )

    def _run_loop(self, step: GenerationStep) -> None:
        """
        Run a single step of the generation strategy.

        :param step: Contains optimization loop specification.
        :type step: GenerationStep
        """
        self.num_batches = self.get_num_batches(
            num_trials=step.num_trials, batch_size=step.max_parallelism
        )

        for batch_number in range(self.num_batches):
            t0 = time.time()

            # Determine batch size.
            batch_size = self.get_batch_size(
                batch_number=batch_number,
                num_batches=self.num_batches,
                batch_size=step.max_parallelism,
                num_trials=step.num_trials,
            )

            # Use sampler or acquisition function to generate trials.
            trials, _ = self.client.get_next_trials(max_trials=batch_size)

            # Evaluate trials.
            results = {
                trial_index: self.objective(parameters)
                for trial_index, parameters in trials.items()
            }

            # Mark trials as complete and update model.
            [
                self.client.complete_trial(trial_index, raw_data=result)
                for trial_index, result in results.items()
            ]

            # Log batch execution time.
            self.batch_execution_times.extend([time.time() - t0] * batch_size)

            # Log metrics.
            if self.monitor:
                self.monitor(self.client)

    def _run_steps(self) -> None:
        """Run the steps of the generation strategy."""
        [self._run_loop(step) for step in self.strategy._steps]


class GridSearch(Optimizer):
    def __init__(
        self,
        objective: Objective,
        search_space: SearchSpace,
        strategy: Union[int, list[int]],
        monitor: Optional[callable] = None,
    ) -> None:
        """
        Initialize grid search strategy.

        :param objective: Objective function for optimization; must be callable.
        :type objective: Objective
        :param search_space: Search space definition.
        :type search_space: SearchSpace
        :param num_trials: Number of trials to run in each dimension. If an
            integer, the same number of points is sampled in each dimension of the
            search space. If a list, each element corresponds to the number of
            points to sample in each dimension of the search space.
        :type num_trials: Union[int, list[int]]
        :param monitor: Callable to log metrics, defaults to None
        :type monitor: Optional[callable], optional
        """
        self.objective = objective
        self.search_space = search_space
        self.num_trials = strategy  # <- This is for compatability with hydra-zen
        self.monitor = monitor
        self.client = AxClient()
        self.batch_execution_times = []

    def run(self, name: str = None) -> None:
        """Run grid search using provided configurations."""
        self._create_experiment(name)
        self._run_loop()

    def _create_experiment(self, name: str) -> None:
        """
        Create an Ax experiment.

        :param name: Name of the experiment.
        :type name: str
        """
        self.client.create_experiment(
            name=name,
            parameters=self.search_space.to_dict(),
            objectives={self.objective.name: self.objective.properties},
        )

    def _run_loop(self) -> None:
        """Run the grid search."""
        for parameters in get_parameterized_grid(
            search_space=self.search_space, num_samples=self.num_trials
        ):
            t0 = time.time()
            # Attach new trial from grid parameterization.
            params, trial_index = self.client.attach_trial(parameters=parameters)

            # Evaluate trial.
            self.client.complete_trial(
                trial_index=trial_index,
                raw_data=self.objective(params),
            )

            # Log batch execution time.
            self.batch_execution_times.append(time.time() - t0)

            # Log metrics.
            if self.monitor:
                self.monitor(self.client)
