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

""" Test QuantumKernelTrainer """

from test import QiskitMachineLearningTestCase

import unittest

import numpy as np

from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import SVCAlignment


def generate_feature_map():
    """
    Create a 2 qubit circuit consisting of 2 user parameters and 2 data bound parameters.
    """
    data_block = ZZFeatureMap(2)
    tunable_block = ZZFeatureMap(2)
    user_parameters = tunable_block.parameters

    for i, _ in enumerate(user_parameters):
        user_parameters[i]._name = f"θ[{i}]"

    feature_map = data_block.compose(tunable_block).compose(data_block)

    return feature_map, user_parameters


class TestQuantumKernelTrainer(QiskitMachineLearningTestCase):
    """Test QuantumKernelTrainer Algorithm"""

    def setUp(self):
        super().setUp()
        self.backend = Aer.get_backend("qasm_simulator")
        self.feature_map, self.user_parameters = generate_feature_map()

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

        self.quantum_kernel = QuantumKernel(
            feature_map=self.feature_map,
            user_parameters=self.user_parameters,
            quantum_instance=self.backend,
        )

    def test_qkt(self):
        """Test QuantumKernelTrainer"""
        self.setUp()
        with self.subTest("check default fit"):
            qkt = QuantumKernelTrainer()
            qkt.fit_kernel(self.quantum_kernel, self.sample_train, self.label_train)
            # Ensure user parameters are bound to real values
            self.assertFalse(self.quantum_kernel.unbound_user_parameters())

        with self.subTest("check fith with params"):
            self.setUp()
            loss = SVCAlignment(C=0.8, gamma="auto")
            qkt = QuantumKernelTrainer(loss=loss)
            qkt.fit_kernel(self.quantum_kernel, self.sample_train, self.label_train)
            # Ensure user parameters are bound to real values
            self.assertFalse(self.quantum_kernel.unbound_user_parameters())


if __name__ == "__main__":
    unittest.main()
