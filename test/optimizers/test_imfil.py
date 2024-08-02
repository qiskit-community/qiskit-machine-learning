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
from test import QiskitAlgorithmsTestCase
import numpy as np

from qiskit_machine_learning.optimizers.imfil import IMFIL
from qiskit_machine_learning.optimizers.optimizer import OptimizerResult
from qiskit_machine_learning.utils import optionals, algorithm_globals


class TestIMFIL(QiskitAlgorithmsTestCase):
    """TestIMFIL"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        if not optionals.HAS_SKQUANT:
            self.skipTest("skquant is required for IMFIL optimizer")
        self.optimizer = IMFIL(maxiter=500)

    def test_minimize_bounds(self):
        """Testing minimize."""

        def objective(x):
            """Defining the objective"""
            return np.sum(x**2)

        starting_point = np.array([0.5, 0.5])
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
