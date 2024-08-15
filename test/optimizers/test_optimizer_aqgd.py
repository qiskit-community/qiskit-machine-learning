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

"""Test of AQGD optimizer"""

import unittest
from test import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning import AlgorithmError
from qiskit_machine_learning.gradients import LinCombEstimatorGradient
from qiskit_machine_learning.optimizers import AQGD
from qiskit_machine_learning.utils import algorithm_globals


@ddt
class TestOptimizerAQGD(QiskitAlgorithmsTestCase):
    """Test AQGD optimizer using RY for analytic gradient with VQE"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        self.qubit_op = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.estimator = Estimator()
        self.gradient = LinCombEstimatorGradient(self.estimator)

    def test_raises_exception(self):
        """tests that AQGD raises an exception when incorrect values are passed."""
        self.assertRaises(AlgorithmError, AQGD, maxiter=[1000], eta=[1.0, 0.5], momentum=[0.0, 0.5])

    def test_max_grouped_evals_non_parallelizable(self):
        """Tests max_grouped_evals for an objective function that cannot be parallelized"""

        # Define the objective function (toy example for functionality)
        def quadratic_objective(x: np.ndarray) -> float:
            # Check if only a single point as parameters is passed
            if np.array(x).ndim != 1:
                raise ValueError("The function expects a vector.")

            return x[0] ** 2 + x[1] ** 2 - 2 * x[0] * x[1]

        # Define initial point
        initial_point = np.array([1, 2.23])
        # Test max_evals_grouped raises no error for max_evals_grouped=1
        aqgd = AQGD(maxiter=100, max_evals_grouped=1)
        x_new = aqgd.minimize(quadratic_objective, initial_point).x
        self.assertAlmostEqual(sum(np.round(x_new / max(x_new), 7)), 0)
        # Test max_evals_grouped raises an error for max_evals_grouped=2
        aqgd.set_max_evals_grouped(2)
        with self.assertRaises(ValueError):
            aqgd.minimize(quadratic_objective, initial_point)


if __name__ == "__main__":
    unittest.main()
