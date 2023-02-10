#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""Module facilitates loading of configuration files, submission of 
optimization jobs, results handling, and logging.
"""

import logging
import pathlib
import sys

from ax.storage.json_store.save import save_experiment

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
import oao.common
from oao.optim.optimizer import logger, BayesianOptimizer, UninformedOptimizer

root_logger = logging.getLogger(__name__)
mod_logger = logger
ax_logger = logging.getLogger("ax")


class Handler:
    """#TODO:_summary_"""

    def __init__(self, config, destination, objective):
        self.config = config
        self.destination = pathlib.Path(destination)
        self.objective = objective
        self._init_logging()

    def _init_logging(self):
        fh = logging.FileHandler(self.destination / "optim.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(oao.common.LOG_FORMAT)

        root_logger.addHandler(fh)
        mod_logger.addHandler(fh)
        ax_logger.addHandler(fh)

    def _get_optimizer(self):

        if self.config["strategy"]["loop_type"] in oao.common.UNINFORMED_STRATEGIES:
            root_logger.info("Uninformed search strategy selected.")
            return UninformedOptimizer
        else:
            root_logger.info("Bayesian search strategy selected.")
            return BayesianOptimizer

    def run(self):
        """#TODO:_summary_"""
        root_logger.info("Starting optimization.")

        Optimizer = self._get_optimizer()
        opt = Optimizer(
            self.objective, self.config["strategy"], self.config["obj_func_parameters"]
        )
        if self.config["evaluation_config"] is not None:
            self.config["evaluation_config"].update({"path": self.destination})
        opt.run(
            self.config["experiment_kwargs"],
            self.config["evaluation_config"],
            self.config["seed"],
        )
        opt.ax_client.save_to_json_file(str(self.destination / "results.json"))
        # save_experiment(
        #     opt.ax_client.experiment, str(self.destination / "results.json")
        # )
        root_logger.info(
            f"Experiment saved to {str(self.destination / 'results.json')}"
        )
        return opt.ax_client
