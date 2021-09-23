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

import unittest

from qiskit import QuantumCircuit, Aer
from test import QiskitMachineLearningTestCase
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QuantumKernelTrainer

import numpy as np


def generate_feature_map():
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


class TestQuantumKernelTrainer(QiskitMachineLearningTestCase):
    """Test QuantumKernelTrainer Algorithm"""

    def setUp(self):
        super().setUp()
        self.backend = Aer.get_backend("qasm_simulator")
        self.feature_map, self.free_parameters = generate_feature_map()

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
            free_parameters=self.free_parameters,
            quantum_instance=self.backend,
        )

    def test_qkt(self):
        """Test QKT"""
        qkt = QuantumKernelTrainer(self.quantum_kernel)


if __name__ == "__main__":
    unittest.main()
