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

"""Test of BOBYQA optimizer"""

import unittest
from test import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_machine_learning.optimizers import BOBYQA


@ddt
class TestOptimizerBOBYQA(QiskitAlgorithmsTestCase):
    """Test BOBYQA optimizer"""

    def setUp(self):
        super().setUp()
        self.quadratic_objective = lambda x: x[0] ** 2 + x[1] ** 2
        self.initial_point = np.array([1.0, 1.0])
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]

    def test_optimizer_minimize(self):
        """Test BOBYQA optimizer minimize method"""
        try:
            bobyqa = BOBYQA(maxiter=100)
            result = bobyqa.minimize(
                self.quadratic_objective, self.initial_point, bounds=self.bounds
            )
            self.assertAlmostEqual(result.fun, 0.0, places=6)
            self.assertTrue(np.allclose(result.x, np.zeros_like(self.initial_point), atol=1e-2))
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

    def test_optimizer_without_bounds(self):
        """Test BOBYQA optimizer without bounds (should raise an error)"""
        try:
            bobyqa = BOBYQA(maxiter=100)
            with self.assertRaises(ValueError):
                bobyqa.minimize(self.quadratic_objective, self.initial_point)
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

    def test_settings(self):
        """Test settings property"""
        try:
            bobyqa = BOBYQA(maxiter=100)
            settings = bobyqa.settings
            self.assertEqual(settings["maxiter"], 100)
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))

    def test_support_level(self):
        """Test support level"""
        try:
            bobyqa = BOBYQA(maxiter=100)
            support_level = bobyqa.get_support_level()
            self.assertEqual(support_level["gradient"], "ignored")
            self.assertEqual(support_level["bounds"], "required")
            self.assertEqual(support_level["initial_point"], "required")
        except MissingOptionalLibraryError as error:
            self.skipTest(str(error))


if __name__ == "__main__":
    unittest.main()
