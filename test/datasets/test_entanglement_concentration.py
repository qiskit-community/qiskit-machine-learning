# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Ad Hoc Data"""

from test import QiskitMachineLearningTestCase

import unittest
import itertools
import numpy as np
from ddt import ddt, unpack, idata

from qiskit.quantum_info import Statevector, partial_trace
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.datasets import entanglement_concentration_data


#  pylint: disable=invalid-name
def _compute_ce(sv):
    """Computing CE using Mathematical Expression due to Beckey, J. L. et al.
    (alternatively SWAP test can be used if done in a Quantum Circuit)"""
    n = sv.num_qubits

    # Convert to density matrix
    rho = sv.to_operator().data
    ce_sum = 0.0

    # Generate all non-empty subsets of qubit indices
    qubit_indices = list(range(n))

    for r in range(1, n + 1):
        for subset in itertools.combinations(qubit_indices, r):

            # Compute the reduced density matrix for the subset
            traced_out = [i for i in qubit_indices if i not in subset]
            reduced_rho = partial_trace(rho, traced_out)
            ce_sum += reduced_rho.purity()

    ce = 1 - (ce_sum / (2 ** n))

    return ce


@ddt
class TestEntangledConcentration(QiskitMachineLearningTestCase):
    """Test Entanglement Concentration Generator"""

    @idata([(n, mode) for n in [3, 4] for mode in ["easy", "hard"]])
    @unpack
    def test_default_params(self, n, mode):
        """Checking for right shapes and labels"""
        x_train, y_train, x_test, y_test = entanglement_concentration_data(
            training_size=4,
            test_size=4,
            n=n,
            mode=mode,
            one_hot=False,
        )
        np.testing.assert_array_equal(x_train.shape, (8, 2 ** n, 1))
        np.testing.assert_array_equal(x_test.shape, (8, 2 ** n, 1))
        np.testing.assert_array_almost_equal(y_train, np.array([0] * 4 + [1] * 4))
        np.testing.assert_array_almost_equal(y_test, np.array([0] * 4 + [1] * 4))

        # Now one_hot=True
        _, y_train_oh, _, y_test_oh = entanglement_concentration_data(
            training_size=4,
            test_size=4,
            n=n,
            mode=mode,
            one_hot=True,
        )
        np.testing.assert_array_equal(y_train_oh, np.array([[1, 0]] * 4 + [[0, 1]] * 4))
        np.testing.assert_array_equal(y_test_oh, np.array([[1, 0]] * 4 + [[0, 1]] * 4))

    @idata([(n,) for n in [3, 4]])
    @unpack
    def test_statevector_format(self, n):
        """Check if output values are normalized qiskit.circuit_info.Statevector objects"""
        x_train, _, _, _ = entanglement_concentration_data(
            training_size=4, test_size=1, n=n, formatting="statevector"
        )
        for state in x_train:
            self.assertIsInstance(state, Statevector)

            norm = np.linalg.norm(state.data)
            self.assertAlmostEqual(norm, 1.0, places=4)

    @idata(
        [
            (3, "easy", [0.18, 0.40]),
            (3, "hard", [0.28, 0.40]),
            (4, "easy", [0.12, 0.43]),
            (4, "hard", [0.22, 0.34]),
        ]
    )
    @unpack
    def test_CE_values(self, n, mode, targets):
        """Check if the right CE values are generated"""

        algorithm_globals.random_seed = 2

        count = 25

        x_train, _, _, _ = entanglement_concentration_data(
            training_size=count, test_size=0, n=n, mode=mode, formatting="statevector"
        )

        low_ce = np.mean([_compute_ce(x_train[i]) for i in range(count)])
        high_ce = np.mean([_compute_ce(x_train[i + count]) for i in range(count)])

        self.assertTrue(abs(low_ce - targets[0]) < 0.02)
        self.assertTrue(abs(high_ce - targets[1]) < 0.02)

    def test_error_raises(self):
        """Check if parameter errors are handled"""
        with self.assertRaises(ValueError):
            entanglement_concentration_data(training_size=4, test_size=1, n=1)

        with self.assertRaises(ValueError):
            entanglement_concentration_data(training_size=4, test_size=1, n=6)


if __name__ == "__main__":
    unittest.main()
