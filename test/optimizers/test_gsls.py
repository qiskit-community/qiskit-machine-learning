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

"""TestGSLS"""

import unittest
from test import QiskitAlgorithmsTestCase
import numpy as np

from qiskit_machine_learning.optimizers.gsls import GSLS
from qiskit_machine_learning.optimizers.optimizer import OptimizerResult
from qiskit_machine_learning.utils import algorithm_globals


class TestGSLS(QiskitAlgorithmsTestCase):
    """TestGSLS"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        self.optimizer = GSLS(maxiter=200, sampling_radius=0.1, initial_step_size=0.01)

    def test_minimize_rosenbrock(self):
        """Tests minimize."""

        def rosenbrock(x):
            """Defines the calculation strategy."""
            return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

        starting_point = np.array([0.0, 0.0])
        result = self.optimizer.minimize(fun=rosenbrock, x0=starting_point)
        self.assertIsInstance(result, OptimizerResult)
        self.assertLess(result.fun, 1.0)
        self.assertLess(result.nfev, 2000)

    def test_minimize_bounds(self):
        """Testing the minimize bounds."""

        def objective(x):
            """Defines the objective."""
            return np.sum(x**2)

        starting_point = np.array([0.1, 0.1])
        bounds = [(-0.05, 0.05), (-0.05, 0.05)]
        result = self.optimizer.minimize(fun=objective, x0=starting_point, bounds=bounds)
        self.assertIsInstance(result, OptimizerResult)
        self.assertTrue(np.all(result.x >= -0.05) and np.all(result.x <= 0.05))

    def test_settings(self):
        """Testing the settings."""
        settings = self.optimizer.settings
        self.assertEqual(settings["maxiter"], 200)
        self.assertEqual(settings["sampling_radius"], 0.1)
        self.assertEqual(settings["initial_step_size"], 0.01)

    def test_sample_set(self):
        """Testing the sample set."""
        n = 2
        x = np.array([0.0, 0.0])
        num_points = 10
        var_lb = np.array([-1.0, -1.0])
        var_ub = np.array([1.0, 1.0])
        directions, points = self.optimizer.sample_set(n, x, var_lb, var_ub, num_points)
        self.assertEqual(directions.shape, (num_points, n))
        self.assertEqual(points.shape, (num_points, n))
        self.assertTrue(np.all(points >= var_lb) and np.all(points <= var_ub))


if __name__ == "__main__":
    unittest.main()
