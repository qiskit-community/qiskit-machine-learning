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

"""Test of ADAM optimizer"""

import unittest
import tempfile
from test import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt

from qiskit_machine_learning.optimizers import ADAM
from qiskit_machine_learning.utils import algorithm_globals


@ddt
class TestOptimizerADAM(QiskitAlgorithmsTestCase):
    """Test ADAM optimizer"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        self.quadratic_objective = lambda x: x[0] ** 2 + x[1] ** 2
        self.initial_point = np.array([1.0, 1.0])

    def test_optimizer_minimize(self):
        """Test ADAM optimizer minimize method"""
        adam = ADAM(maxiter=150, tol=1e-6, lr=1e-1)
        result = adam.minimize(self.quadratic_objective, self.initial_point)
        self.assertAlmostEqual(result.fun, 0.0, places=6)
        self.assertTrue(np.allclose(result.x, np.zeros_like(self.initial_point), atol=1e-2))

    def test_optimizer_with_noise(self):
        """Test ADAM optimizer with noise factor"""
        adam = ADAM(maxiter=100, tol=1e-6, lr=1e-1, noise_factor=1e-2)
        result = adam.minimize(self.quadratic_objective, self.initial_point)
        self.assertAlmostEqual(result.fun, 0.0, places=4)
        self.assertTrue(np.allclose(result.x, np.zeros_like(self.initial_point), atol=1e-2))

    def test_amsgrad(self):
        """Test ADAM optimizer with AMSGRAD variant"""
        adam = ADAM(maxiter=150, tol=1e-6, lr=1e-1, amsgrad=True)
        result = adam.minimize(self.quadratic_objective, self.initial_point)
        self.assertAlmostEqual(result.fun, 0.0, places=6)
        self.assertTrue(np.allclose(result.x, np.zeros_like(self.initial_point), atol=1e-2))

    def test_save_load_params(self):
        """Test save and load optimizer parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            adam = ADAM(maxiter=100, tol=1e-6, lr=1e-1, snapshot_dir=tmpdir)
            adam.minimize(self.quadratic_objective, self.initial_point)
            new_adam = ADAM(snapshot_dir=tmpdir)
            new_adam.load_params(tmpdir)

            self.assertTrue(np.allclose(adam._m, new_adam._m))
            self.assertTrue(np.allclose(adam._v, new_adam._v))
            self.assertEqual(adam._t, new_adam._t)

    def test_settings(self):
        """Test settings property"""
        adam = ADAM(maxiter=100, tol=1e-6, lr=1e-1)
        settings = adam.settings
        self.assertEqual(settings["maxiter"], 100)
        self.assertEqual(settings["tol"], 1e-6)
        self.assertEqual(settings["lr"], 1e-1)
        self.assertEqual(settings["beta_1"], 0.9)
        self.assertEqual(settings["beta_2"], 0.99)
        self.assertEqual(settings["noise_factor"], 1e-8)
        self.assertEqual(settings["eps"], 1e-10)
        self.assertEqual(settings["amsgrad"], False)
        self.assertEqual(settings["snapshot_dir"], None)

    def test_callback(self):
        """Test using the callback."""

        history = {"ite": [], "weights": [], "fvals": []}

        def callback(n_t, weight, fval):
            history["ite"].append(n_t)
            history["weights"].append(weight)
            history["fvals"].append(fval)

        adam = ADAM(maxiter=100, tol=1e-6, lr=1e-1, callback=callback)
        adam.minimize(self.quadratic_objective, self.initial_point)

        expected_types = [int, np.ndarray, float]
        for i, (key, values) in enumerate(history.items()):
            self.assertTrue(all(isinstance(value, expected_types[i]) for value in values))
            self.assertEqual(len(history[key]), 100)


if __name__ == "__main__":
    unittest.main()
