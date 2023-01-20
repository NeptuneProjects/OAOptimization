#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

import argparse
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
from oao.handler import Handler
from oao.optim.objective import evaluate_branin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reads optimization configuration files and submits for execution."
    )
    parser.add_argument(
        "-source", type=str, help="Path to source file.", default="scripts/config.json"
    )
    parser.add_argument(
        "-destination", type=str, help="Path to destination file.", default="scripts/"
    )
    args = parser.parse_args()

    df = Handler(
        pathlib.Path(args.source), pathlib.Path(args.destination), evaluate_branin
    ).run()
    print(df)
