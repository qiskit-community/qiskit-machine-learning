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

"""TestIMFIL"""

import unittest
import numpy as np
from qiskit_machine_learning.optimizers.imfil import IMFIL
from qiskit_machine_learning.optimizers.optimizer import OptimizerResult
from qiskit_machine_learning.utils import optionals


class TestIMFIL(unittest.TestCase):
    """TestIMFIL"""

    def setUp(self):
        if not optionals.HAS_SKQUANT:
            self.skipTest("skquant is required for IMFIL optimizer")
        self.optimizer = IMFIL(maxiter=500)

    def test_support_level(self):
        """Test support level."""
        support_levels = self.optimizer.get_support_level()
        self.assertEqual(support_levels["gradient"], 0)
        self.assertEqual(support_levels["bounds"], 2)
        self.assertEqual(support_levels["initial_point"], 2)

    def test_minimize_rosenbrock(self):
        """Testing minimize."""

        def rosenbrock(x):
            """Calculation strategy."""
            return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

        starting_point = np.array([-1.2, 1.0])
        bounds = [(-2.0, 2.0), (-1.0, 3.0)]
        result = self.optimizer.minimize(fun=rosenbrock, x0=starting_point, bounds=bounds)
        self.assertIsInstance(result, OptimizerResult)
        self.assertLess(result.fun, 1e-4)
        self.assertLess(result.nfev, 2000)

    def test_minimize_bounds(self):
        """Testing minimize."""

        def objective(x):
            """Defining the objective"""
            return np.sum(x**2)

        starting_point = np.array([1.0, 1.0])
        bounds = [(-0.5, 0.5), (-0.5, 0.5)]
        result = self.optimizer.minimize(fun=objective, x0=starting_point, bounds=bounds)
        self.assertIsInstance(result, OptimizerResult)
        self.assertTrue(np.all(result.x >= -0.5) and np.all(result.x <= 0.5))

    def test_settings(self):
        """Test settings."""
        settings = self.optimizer.settings
        self.assertEqual(settings["maxiter"], 500)


if __name__ == "__main__":
    unittest.main()
