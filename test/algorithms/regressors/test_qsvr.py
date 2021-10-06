# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QSVR """

import unittest
from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.exceptions import (
    QiskitMachineLearningError,
    QiskitMachineLearningWarning,
)
from qiskit_machine_learning.kernels import QuantumKernel


class TestQSVR(QiskitMachineLearningTestCase):
    """Test QSVR Algorithm"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.asarray([0, 1])

    def test_qsvr(self):
        """Test QSVR"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvr = QSVR(quantum_kernel=qkernel)
        qsvr.fit(self.sample_train, self.label_train)
        score = qsvr.score(self.sample_test, self.label_test)

        self.assertAlmostEqual(score, 0.38365, places=4)

    def test_empty_kernel(self):
        """Test QSVR with empty QuantumKernel"""
        qkernel = QuantumKernel()
        qsvr = QSVR(quantum_kernel=qkernel)

        with self.assertRaises(QiskitMachineLearningError):
            _ = qsvr.fit(self.sample_train, self.label_train)

    def test_change_kernel(self):
        """Test QSVR with QuantumKernel later"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvr = QSVR()
        qsvr.quantum_kernel = qkernel
        qsvr.fit(self.sample_train, self.label_train)
        score = qsvr.score(self.sample_test, self.label_test)

        self.assertAlmostEqual(score, 0.38365, places=4)

    def test_qsvr_parameters(self):
        """Test QSVR with extra constructor parameters"""

        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvr = QSVR(quantum_kernel=qkernel, tol=1e-4, C=0.5)
        qsvr.fit(self.sample_train, self.label_train)
        score = qsvr.score(self.sample_test, self.label_test)

        self.assertAlmostEqual(score, 0.38365, places=4)

    def test_qsvc_to_string(self):
        """Test QSVR print works when no *args passed in"""
        qsvr = QSVR()
        _ = str(qsvr)

    def test_with_kernel_parameter(self):
        """Test QSVC with the `kernel` argument."""
        with self.assertWarns(QiskitMachineLearningWarning):
            QSVR(kernel=1)


if __name__ == "__main__":
    unittest.main()
