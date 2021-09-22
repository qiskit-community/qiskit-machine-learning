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

""" Test QSVC """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, QuantumKernelTrainer
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


def generate_tunable_feature_map():
    """
    Create a 2 qubit circuit consisting of 2 free parameters and 2 data bound parameters.
    """
    data_block = ZZFeatureMap(2)
    tunable_block = ZZFeatureMap(2)
    free_parameters = tunable_block.parameters

    for i in range(len(free_parameters)):
        free_parameters[i]._name = f"θ[{i}]"

    feature_map = data_block.compose(tunable_block).compose(data_block)

    return feature_map, free_parameters


class TestQSVC(QiskitMachineLearningTestCase):
    """Test QSVC Algorithm"""

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

    def test_qsvc(self):
        """Test QSVC"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_empty_kernel(self):
        """Test QSVC with empty QuantumKernel"""
        qkernel = QuantumKernel()
        qsvc = QSVC(quantum_kernel=qkernel)

        with self.assertRaises(QiskitMachineLearningError):
            _ = qsvc.fit(self.sample_train, self.label_train)

    def test_change_kernel(self):
        """Test QSVC with QuantumKernel later"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC()
        qsvc.quantum_kernel = qkernel
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_qsvc_parameters(self):
        """Test QSVC with extra constructor parameters"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC(quantum_kernel=qkernel, tol=1e-4, C=0.5)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_unbound_free_params(self):
        """Test QSVC with extra constructor parameters"""
        fm, fp = generate_tunable_feature_map()
        qkernel = QuantumKernel(
            feature_map=fm,
            free_parameters=fp,
            quantum_instance=BasicAer.get_backend("qasm_simulator"),
        )

        qkt = QuantumKernelTrainer(qkernel)

        qsvc = QSVC(quantum_kernel=qkt)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
