# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QuantumKernelTrainer """

from test import QiskitMachineLearningTestCase

import unittest

import numpy as np

import qiskit
from qiskit.utils import algorithm_globals, optionals
from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.algorithms.classifiers import QSVC


class TestQuantumKernelTrainer(QiskitMachineLearningTestCase):
    """Test QuantumKernelTrainer Algorithm"""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598
        self.optimizer = COBYLA(maxiter=25)
        # pylint: disable=no-member
        self.backend = qiskit.providers.aer.AerSimulator(method="statevector")
        data_block = ZZFeatureMap(2)
        trainable_block = ZZFeatureMap(2, parameter_prefix="θ")
        training_parameters = trainable_block.parameters

        self.feature_map = data_block.compose(trainable_block).compose(data_block)
        self.training_parameters = training_parameters

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
        self.label_test = np.asarray([1, 0])

        self.quantum_kernel = QuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
            quantum_instance=self.backend,
        )

    def test_qkt(self):
        """Test QuantumKernelTrainer"""
        with self.subTest("check default fit"):
            qkt = QuantumKernelTrainer(quantum_kernel=self.quantum_kernel)
            qkt_result = qkt.fit(self.sample_train, self.label_train)

            # Ensure kernel training functions and is deterministic
            qsvc = QSVC(quantum_kernel=qkt_result.quantum_kernel)
            qsvc.fit(self.sample_train, self.label_train)
            score = qsvc.score(self.sample_test, self.label_test)
            self.assertGreaterEqual(score, 0.5)

        with self.subTest("check fit with params"):
            loss = SVCLoss(C=0.8, gamma="auto")
            qkt = QuantumKernelTrainer(quantum_kernel=self.quantum_kernel, loss=loss)
            qkt_result = qkt.fit(self.sample_train, self.label_train)

            # Ensure user parameters are bound to real values
            self.assertEqual(len(qkt_result.quantum_kernel.get_unbound_training_parameters()), 0)

            # Ensure kernel training functions and is deterministic
            qsvc = QSVC(quantum_kernel=qkt_result.quantum_kernel)
            qsvc.fit(self.sample_train, self.label_train)
            score = qsvc.score(self.sample_test, self.label_test)
            self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
