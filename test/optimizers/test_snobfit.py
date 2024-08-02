# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for the SNOBFIT optimizer."""

import unittest
from test import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_machine_learning.optimizers.snobfit import SNOBFIT
from qiskit_machine_learning.utils import algorithm_globals


class TestSNOBFITOptimizer(QiskitAlgorithmsTestCase):
    """Test cases for the SNOBFIT optimizer."""

    def setUp(self):
        """Set up the optimizer for testing."""
        super().setUp()
        algorithm_globals.random_seed = 50
        try:
            self.optimizer = SNOBFIT(maxiter=200, maxfail=5, verbose=True)

        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

    def test_optimizer_support(self):
        """Test the optimizer support levels."""
        support_levels = self.optimizer.get_support_level()
        self.assertIn("gradient", support_levels)
        self.assertIn("bounds", support_levels)
        self.assertIn("initial_point", support_levels)

    def test_minimize_simple_quadratic(self):
        """Test the optimizer on a simple quadratic function."""

        def quadratic_function(params):
            """Test function."""
            return np.sum((params - 4) ** 2)

        initial_point = np.array([0.0, 0.0])
        bounds = [(0, 10), (0, 10)]
        result = self.optimizer.minimize(fun=quadratic_function, x0=initial_point, bounds=bounds)
        self.assertTrue(np.allclose(result.x, np.array([4.0, 4.0]), atol=1e-2))

    def test_minimize_rosenbrock(self):
        """Test the optimizer on the Rosenbrock function."""

        def rosenbrock_function(params):
            """Test function."""
            return sum(100.0 * (params[1:] - params[:-1] ** 2.0) ** 2.0 + (1 - params[:-1]) ** 2.0)

        initial_point = np.array([1.5, 1.5])
        bounds = [(-5, 5), (-5, 5)]
        result = self.optimizer.minimize(fun=rosenbrock_function, x0=initial_point, bounds=bounds)
        self.assertLess(result.fun, 1e-4)
        self.assertLess(result.nfev, 200)

    def test_optimizer_settings(self):
        """Test the optimizer settings."""
        settings = self.optimizer.settings
        self.assertEqual(settings["maxiter"], 200)
        self.assertEqual(settings["maxfail"], 5)
        self.assertIsInstance(settings["maxmp"], int)
        self.assertTrue(settings["verbose"])

    def test_minimize_with_invalid_bounds(self):
        """Test the optimizer with invalid bounds."""

        def quadratic_function(params):
            """Test function."""
            return np.sum((params - 2) ** 2)

        initial_point = np.array([1.0, 1.0])
        bounds = [(None, 5), (None, 5)]  # Invalid bounds with None
        with self.assertRaises(ValueError):
            self.optimizer.minimize(fun=quadratic_function, x0=initial_point, bounds=bounds)

    def test_minimize_with_clamped_initial_point(self):
        """Test the optimizer with an initial point that is clamped to bounds."""

        def quadratic_function(params):
            """Test function."""
            return np.sum((params - 2) ** 2)

        initial_point = np.array([10.0, -10.0])
        bounds = [(0, 5), (0, 5)]
        result = self.optimizer.minimize(fun=quadratic_function, x0=initial_point, bounds=bounds)
        self.assertTrue(np.allclose(result.x, np.array([5.0, 0.0]), atol=1e-2))


if __name__ == "__main__":
    unittest.main()
