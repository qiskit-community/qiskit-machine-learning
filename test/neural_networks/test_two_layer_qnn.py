# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Two Layer QNN."""

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data
from qiskit.providers.aer import StatevectorSimulator
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import TwoLayerQNN


@ddt
class TestTwoLayerQNN(QiskitMachineLearningTestCase):
    """Two Layer QNN Tests."""

    def setUp(self):
        super().setUp()

        # specify "run configuration"
        quantum_instance = QuantumInstance(StatevectorSimulator())

        # define QNN
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        self.qnn = TwoLayerQNN(num_qubits, feature_map=feature_map,
                               ansatz=ansatz, quantum_instance=quantum_instance)

        self.qnn_no_qi = TwoLayerQNN(num_qubits, feature_map=feature_map,
                                     ansatz=ansatz)

    @data(
        "qi",
        "no_qi"
    )
    def test_qnn_simple_new(self, qnn_type: str):
        """Simple Opflow QNN Test for a specified neural network."""

        input_data = np.zeros(self.qnn.num_inputs)
        weights = np.zeros(self.qnn.num_weights)

        if qnn_type == "qi":
            qnn = self.qnn
        else:
            qnn = self.qnn_no_qi

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, (1, *self.qnn.output_shape))

        # test backward pass
        result = self.qnn.backward(input_data, weights)
        # batch dimension of 1
        self.assertEqual(result[0].shape, (1, *self.qnn.output_shape, self.qnn.num_inputs))
        self.assertEqual(result[1].shape, (1, *self.qnn.output_shape, self.qnn.num_weights))

    @data(
        "qi",
        "no_qi"
    )
    def _test_qnn_batch(self, qnn_type: str):
        """Batched Opflow QNN Test for the specified network."""
        batch_size = 10

        input_data = np.arange(batch_size * self.qnn.num_inputs)\
            .reshape((batch_size, self.qnn.num_inputs))
        weights = np.zeros(self.qnn.num_weights)

        if qnn_type == "qi":
            qnn = self.qnn
        else:
            qnn = self.qnn_no_qi

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, (batch_size, *self.qnn.output_shape))

        # test backward pass
        result = self.qnn.backward(input_data, weights)
        self.assertEqual(result[0].shape,
                         (batch_size, *self.qnn.output_shape, self.qnn.num_inputs))
        self.assertEqual(result[1].shape,
                         (batch_size, *self.qnn.output_shape, self.qnn.num_weights))


if __name__ == '__main__':
    unittest.main()
