#!/usr/bin/env python3

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())
import unittest

import numpy as np

from oao.optim import uninformed

# TODO: Build class with factory functions


class TestInputs(unittest.TestCase):
    bounds = {"x1": [0, 1], "x2": [0, 1], "x3": [0, 1], "x4": [0, 1]}
    seed = 2009

    # Test inputs for uninformed.get_grid_samples

    def test_grid_inputs_equal_size(self):
        num_samples = 10
        df = uninformed.get_grid_samples(self.bounds, num_samples)
        self.assertEqual(len(df), num_samples ** len(self.bounds))

    def test_grid_inputs_diff_size(self):
        num_samples = [10, 20, 30, 40]
        df = uninformed.get_grid_samples(self.bounds, num_samples)
        self.assertEqual(len(df), np.prod(num_samples))

    def test_grid_inputs_dim_check(self):
        num_samples = [10, 20, 30]
        self.assertRaises(
            ValueError, uninformed.get_grid_samples, self.bounds, num_samples
        )

    def test_grid_col_keys(self):
        num_samples = 2
        df = uninformed.get_grid_samples(self.bounds, num_samples)
        [
            self.assertEqual(list(self.bounds.keys())[i], col)
            for i, col in enumerate(df.columns)
        ]

    # Test inputs for uninformed.get_latin_hypercube_samples

    def test_lhs_inputs(self):
        num_samples = 10
        df = uninformed.get_latin_hypercube_samples(self.bounds, num_samples, self.seed)
        self.assertEqual(len(df), num_samples)

    def test_lhs_col_keys(self):
        num_samples = 10
        df = uninformed.get_latin_hypercube_samples(self.bounds, num_samples, self.seed)
        [
            self.assertEqual(list(self.bounds.keys())[i], col)
            for i, col in enumerate(df.columns)
        ]

    # Test inputs for uninformed.get_random_samples

    def test_rand_inputs(self):
        num_samples = 10
        df = uninformed.get_random_samples(self.bounds, num_samples, self.seed)
        self.assertEqual(len(df), num_samples)

    def test_rand_col_keys(self):
        num_samples = 10
        df = uninformed.get_random_samples(self.bounds, num_samples, self.seed)
        [
            self.assertEqual(list(self.bounds.keys())[i], col)
            for i, col in enumerate(df.columns)
        ]

    # Test inputs for uninformed.get_sobol_samples

    def test_sobol_inputs(self):
        num_samples = 8
        df = uninformed.get_sobol_samples(self.bounds, num_samples, self.seed)
        self.assertEqual(len(df), num_samples)

    def test_lhs_col_keys(self):
        num_samples = 8
        df = uninformed.get_sobol_samples(self.bounds, num_samples, self.seed)
        [
            self.assertEqual(list(self.bounds.keys())[i], col)
            for i, col in enumerate(df.columns)
        ]


if __name__ == "__main__":
    unittest.main()
