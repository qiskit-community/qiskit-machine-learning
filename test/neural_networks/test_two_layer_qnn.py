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
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.neural_networks import TwoLayerQNN


@ddt
class TestTwoLayerQNN(QiskitMachineLearningTestCase):
    """Two Layer QNN Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345
        # specify "run configuration"
        quantum_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # define QNN
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        self.qnn = TwoLayerQNN(
            num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=quantum_instance,
        )

        self.qnn_no_qi = TwoLayerQNN(num_qubits, feature_map=feature_map, ansatz=ansatz)

    @data(
        ("qi", True),
        ("no_qi", True),
        ("qi", False),
        ("no_qi", False),
    )
    def test_qnn_simple_new(self, config):
        """Simple Opflow QNN Test for a specified neural network."""
        qnn_type, input_grad_required = config

        input_data = np.zeros(self.qnn.num_inputs)
        weights = np.zeros(self.qnn.num_weights)

        if qnn_type == "qi":
            qnn = self.qnn
        else:
            qnn = self.qnn_no_qi
        qnn.input_gradients = input_grad_required

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, (1, *qnn.output_shape))

        # test backward pass
        result = qnn.backward(input_data, weights)
        # batch dimension of 1
        if qnn.input_gradients:
            self.assertEqual(result[0].shape, (1, *qnn.output_shape, qnn.num_inputs))
        else:
            self.assertIsNone(result[0])

        self.assertEqual(result[1].shape, (1, *qnn.output_shape, qnn.num_weights))

    @data(
        ("qi", True),
        ("no_qi", True),
        ("qi", False),
        ("no_qi", False),
    )
    def _test_qnn_batch(self, config):
        """Batched Opflow QNN Test for the specified network."""
        qnn_type, input_grad_required = config

        batch_size = 10

        input_data = np.arange(batch_size * self.qnn.num_inputs).reshape(
            (batch_size, self.qnn.num_inputs)
        )
        weights = np.zeros(self.qnn.num_weights)

        if qnn_type == "qi":
            qnn = self.qnn
        else:
            qnn = self.qnn_no_qi
        qnn.input_gradients = input_grad_required

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, (batch_size, *qnn.output_shape))

        # test backward pass
        result = qnn.backward(input_data, weights)
        if qnn.input_gradients:
            self.assertEqual(result[0].shape, (batch_size, *qnn.output_shape, qnn.num_inputs))
        else:
            self.assertIsNone(result[0])

        self.assertEqual(result[1].shape, (batch_size, *qnn.output_shape, qnn.num_weights))


if __name__ == "__main__":
    unittest.main()
