# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for NLopt optimizers."""

import unittest
from test import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_machine_learning.optimizers.nlopts import CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES
from qiskit_machine_learning.utils import algorithm_globals


class TestNLoptOptimizer(QiskitAlgorithmsTestCase):
    """Test cases for NLoptOptimizer and its derived classes."""

    def setUp(self):
        """Set up optimizers for testing."""
        super().setUp()
        algorithm_globals.random_seed = 50
        self.max_evals = 200
        self.bounds = [(-5, 5), (-5, 5)]

    def test_optimizer_support(self):
        """Test the support levels of NLopt optimizers."""
        try:
            optimizers = [
                CRS(max_evals=self.max_evals),
                DIRECT_L(max_evals=self.max_evals),
                DIRECT_L_RAND(max_evals=self.max_evals),
                ESCH(max_evals=self.max_evals),
                ISRES(max_evals=self.max_evals),
            ]
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

        for optimizer in optimizers:
            support_levels = optimizer.get_support_level()
            self.assertIn("gradient", support_levels)
            self.assertIn("bounds", support_levels)
            self.assertIn("initial_point", support_levels)

    def test_optimizer_settings(self):
        """Test the optimizer settings."""
        try:
            optimizers = [
                CRS(max_evals=self.max_evals),
                DIRECT_L(max_evals=self.max_evals),
                DIRECT_L_RAND(max_evals=self.max_evals),
                ESCH(max_evals=self.max_evals),
                ISRES(max_evals=self.max_evals),
            ]
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

        for optimizer in optimizers:
            settings = optimizer.settings
            self.assertEqual(settings["max_evals"], self.max_evals)

    def test_minimize_simple_quadratic(self):
        """Test optimizers on a simple quadratic function."""

        def quadratic_function(params):
            """Test function."""
            return np.sum((params - 4) ** 2)

        initial_point = np.array([0.0, 0.0])

        try:
            optimizers = [
                CRS(max_evals=self.max_evals),
                DIRECT_L(max_evals=self.max_evals),
                DIRECT_L_RAND(max_evals=self.max_evals),
                ESCH(max_evals=self.max_evals),
                ISRES(max_evals=self.max_evals),
            ]
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

        for optimizer in optimizers:
            result = optimizer.minimize(
                fun=quadratic_function, x0=initial_point, bounds=self.bounds
            )
            self.assertTrue(np.allclose(result.x, np.array([4.0, 4.0]), atol=1e-2))

    def test_minimize_rosenbrock(self):
        """Test optimizers on the Rosenbrock function."""

        def rosenbrock_function(params):
            """Test function."""
            return sum(100.0 * (params[1:] - params[:-1] ** 2.0) ** 2.0 + (1 - params[:-1]) ** 2.0)

        initial_point = np.array([1.5, 1.5])

        try:
            optimizers = [
                CRS(max_evals=self.max_evals),
                DIRECT_L(max_evals=self.max_evals),
                DIRECT_L_RAND(max_evals=self.max_evals),
                ESCH(max_evals=self.max_evals),
                ISRES(max_evals=self.max_evals),
            ]
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

        for optimizer in optimizers:
            result = optimizer.minimize(
                fun=rosenbrock_function, x0=initial_point, bounds=self.bounds
            )
            self.assertLess(result.fun, 1e-4)
            self.assertLess(result.nfev, self.max_evals)

    def test_minimize_with_invalid_bounds(self):
        """Test optimizers with invalid bounds."""

        def quadratic_function(params):
            """Test function."""
            return np.sum((params - 2) ** 2)

        initial_point = np.array([1.0, 1.0])
        invalid_bounds = [(None, 5), (None, 5)]  # Invalid bounds with None

        try:
            optimizers = [
                CRS(max_evals=self.max_evals),
                DIRECT_L(max_evals=self.max_evals),
                DIRECT_L_RAND(max_evals=self.max_evals),
                ESCH(max_evals=self.max_evals),
                ISRES(max_evals=self.max_evals),
            ]
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

        for optimizer in optimizers:
            with self.assertRaises(ValueError):
                optimizer.minimize(fun=quadratic_function, x0=initial_point, bounds=invalid_bounds)

    def test_minimize_with_clamped_initial_point(self):
        """Test optimizers with an initial point that is clamped to bounds."""

        def quadratic_function(params):
            """Test function."""
            return np.sum((params - 2) ** 2)

        initial_point = np.array([10.0, -10.0])

        try:
            optimizers = [
                CRS(max_evals=self.max_evals),
                DIRECT_L(max_evals=self.max_evals),
                DIRECT_L_RAND(max_evals=self.max_evals),
                ESCH(max_evals=self.max_evals),
                ISRES(max_evals=self.max_evals),
            ]
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

        for optimizer in optimizers:
            result = optimizer.minimize(
                fun=quadratic_function, x0=initial_point, bounds=self.bounds
            )
            self.assertTrue(np.allclose(result.x, np.array([5.0, 0.0]), atol=1e-2))


if __name__ == "__main__":
    unittest.main()
