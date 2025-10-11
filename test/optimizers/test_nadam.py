# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of NAdam optimizer"""

import unittest
import tempfile
import numpy as np
from ddt import ddt

# Import the test case base class
from test.algorithms_test_case import QiskitAlgorithmsTestCase
from qiskit_machine_learning.optimizers import NAdam
from qiskit_machine_learning.utils import algorithm_globals


@ddt
class TestOptimizerNAdam(QiskitAlgorithmsTestCase):
    """Test NAdam optimizer"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 42
        self.quadratic_objective = lambda x: x[0] ** 2 + x[1] ** 2
        self.initial_point = np.array([1.0, 1.0])

    def test_optimizer_minimize(self):
        """Test NAdam optimizer minimize method"""
        nadam = NAdam(maxiter=200, tol=1e-6, lr=1e-1)
        result = nadam.minimize(self.quadratic_objective, self.initial_point)
        self.assertAlmostEqual(result.fun, 0.0, places=6)
        self.assertTrue(np.allclose(result.x, np.zeros_like(self.initial_point), atol=1e-2))

    def test_optimizer_with_noise(self):
        """Test NAdam optimizer with noise factor"""
        nadam = NAdam(maxiter=150, tol=1e-6, lr=1e-1, noise_factor=1e-2)
        result = nadam.minimize(self.quadratic_objective, self.initial_point)
        self.assertAlmostEqual(result.fun, 0.0, places=4)
        self.assertTrue(np.allclose(result.x, np.zeros_like(self.initial_point), atol=1e-2))

    def test_save_load_params(self):
        """Test save and load optimizer parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nadam = NAdam(maxiter=100, tol=1e-6, lr=1e-1, snapshot_dir=tmpdir)
            nadam.minimize(self.quadratic_objective, self.initial_point)
            new_nadam = NAdam(snapshot_dir=tmpdir)
            new_nadam.load_params(tmpdir)

            self.assertTrue(np.allclose(nadam._m, new_nadam._m))
            self.assertTrue(np.allclose(nadam._v, new_nadam._v))
            self.assertEqual(nadam._t, new_nadam._t)

    def test_settings(self):
        """Test settings property"""
        nadam = NAdam(maxiter=100, tol=1e-6, lr=1e-1)
        settings = nadam.settings
        self.assertEqual(settings["maxiter"], 100)
        self.assertEqual(settings["tol"], 1e-6)
        self.assertEqual(settings["lr"], 1e-1)
        self.assertEqual(settings["beta_1"], 0.9)
        self.assertEqual(settings["beta_2"], 0.999)
        self.assertEqual(settings["eps"], 1e-8)
        self.assertEqual(settings["noise_factor"], 1e-8)
        # NAdam does not have amsgrad, so just check key safely
        self.assertIsNone(settings.get("amsgrad"))
        self.assertEqual(settings["snapshot_dir"], None)

    def test_callback(self):
        """Test using the callback."""
        history = {"ite": [], "weights": [], "fvals": []}

        def callback(n_t, weight, fval):
            history["ite"].append(n_t)
            history["weights"].append(weight)
            history["fvals"].append(fval)

        nadam = NAdam(maxiter=100, tol=1e-6, lr=1e-1, callback=callback)
        nadam.minimize(self.quadratic_objective, self.initial_point)

        expected_types = [int, np.ndarray, float]
        for i, (key, values) in enumerate(history.items()):
            self.assertTrue(all(isinstance(value, expected_types[i]) for value in values))
            self.assertEqual(len(history[key]), 100)


if __name__ == "__main__":
    unittest.main()
