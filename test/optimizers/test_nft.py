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

"""Unit tests for the Nakanishi-Fujii-Todo (NFT) optimizer."""

import unittest
from test import QiskitAlgorithmsTestCase
import numpy as np

from qiskit_machine_learning.optimizers.nft import NFT
from qiskit_machine_learning.utils import algorithm_globals


class TestNFTOptimizer(QiskitAlgorithmsTestCase):
    """Test cases for the NFT optimizer."""

    def setUp(self):
        """Set up the optimizer for testing."""
        super().setUp()
        algorithm_globals.random_seed = 50
        self.optimizer = NFT(maxiter=400, maxfev=1000)

    def test_optimizer_support(self):
        """Test the optimizer support levels."""
        support_levels = self.optimizer.get_support_level()
        self.assertIn("gradient", support_levels)
        self.assertIn("bounds", support_levels)
        self.assertIn("initial_point", support_levels)

    def test_minimize_simple_quadratic(self):
        """Test the optimizer on a simple quadratic function."""

        def quadratic_function(x):
            """Test function."""
            return np.sum((x - 3) ** 2)

        initial_point = np.array([0.0, 0.0])
        result = self.optimizer.minimize(fun=quadratic_function, x0=initial_point)
        self.assertTrue(np.allclose(result.x, np.array([3.0, 3.0]), atol=1e-2))

    def test_optimizer_settings(self):
        """Test the optimizer settings."""
        settings = self.optimizer.settings
        self.assertEqual(settings["maxiter"], 400)
        self.assertEqual(settings["maxfev"], 1000)


if __name__ == "__main__":
    unittest.main()
