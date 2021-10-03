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

""" Test Pegasos QSVC """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import PegasosQSVC


from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


class TestPegasosQSVC(QiskitMachineLearningTestCase):
    """Test Pegasos QSVC Algorithm"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # number of qubits is equal to the number of features
        self.q = 2
        # number of steps performed during the training procedure
        self.T = 100

        self.feature_map = ZFeatureMap(feature_dimension=self.q, reps=1)

        sample, label = make_blobs(n_samples=20, n_features=2, centers=2, random_state=3, shuffle=True)
        sample = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(sample)

        # split into train and test set
        self.sample_train = sample[:15]
        self.label_train = label[:15]
        self.sample_test = sample[15:]
        self.label_test = label[15:]

    def test_qsvc(self):
        """Test PegasosQSVC"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=1000, num_steps=100)

        pegasos_qsvc.fit(self.sample_train, self.label_train)
        score = pegasos_qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_empty_kernel(self):
        """Test PegasosQSVC with empty QuantumKernel"""
        qkernel = QuantumKernel()
        pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel)

        with self.assertRaises(QiskitMachineLearningError):
            _ = pegasos_qsvc.fit(self.sample_train, self.label_train)

    def test_change_kernel(self):
        """Test QSVC with QuantumKernel later"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        pegasos_qsvc = PegasosQSVC(C=1000, num_steps=100)
        pegasos_qsvc.quantum_kernel = qkernel
        pegasos_qsvc.fit(self.sample_train, self.label_train)
        score = pegasos_qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1)

    def test_labels(self):
        """Test PegasosQSVC with different integer labels than {0, 1}"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=1000, num_steps=100)

        label_train_temp = self.label_train.copy()
        label_train_temp[self.label_train==0] = 2
        label_train_temp[self.label_train==1] = 3


        label_test_temp = self.label_test.copy()
        label_test_temp[self.label_test==0] = 2
        label_test_temp[self.label_test==1] = 3

        pegasos_qsvc.fit(self.sample_train, label_train_temp)
        score = pegasos_qsvc.score(self.sample_test, label_test_temp)

        self.assertEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
