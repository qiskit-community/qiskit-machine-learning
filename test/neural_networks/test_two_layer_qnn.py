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
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import TwoLayerQNN


class TestTwoLayerQNN(QiskitMachineLearningTestCase):
    """Two Layer QNN Tests."""

    def setUp(self):
        super().setUp()

        # specify "run configuration"
        backend = Aer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend)

        # define QNN
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits)
        var_form = RealAmplitudes(num_qubits, reps=1)
        self.qnn = TwoLayerQNN(num_qubits, feature_map=feature_map,
                               var_form=var_form, quantum_instance=quantum_instance)

        self.qnn_no_qi = TwoLayerQNN(num_qubits, feature_map=feature_map,
                                     var_form=var_form)

    def _test_qnn_simple(self, qnn: TwoLayerQNN):
        """Simple Opflow QNN Test for a specified neural network."""

        input_data = np.zeros(self.qnn.num_inputs)
        weights = np.zeros(self.qnn.num_weights)

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, (1, *self.qnn.output_shape))

        # test backward pass
        result = self.qnn.backward(input_data, weights)
        # batch dimension of 1
        self.assertEqual(result[0].shape, (1, *self.qnn.output_shape, self.qnn.num_inputs))
        self.assertEqual(result[1].shape, (1, *self.qnn.output_shape, self.qnn.num_weights))

    def _test_qnn_batch(self, qnn: TwoLayerQNN):
        """Batched Opflow QNN Test for the specified network."""
        batch_size = 10

        input_data = np.arange(batch_size * self.qnn.num_inputs)\
            .reshape((batch_size, self.qnn.num_inputs))
        weights = np.zeros(self.qnn.num_weights)

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, (batch_size, *self.qnn.output_shape))

        # test backward pass
        result = self.qnn.backward(input_data, weights)
        self.assertEqual(result[0].shape,
                         (batch_size, *self.qnn.output_shape, self.qnn.num_inputs))
        self.assertEqual(result[1].shape,
                         (batch_size, *self.qnn.output_shape, self.qnn.num_weights))

    def test_qnn_simple(self):
        """Simple Opflow QNN Test on a network with an instance of QuantumInstance."""
        self._test_qnn_simple(self.qnn)

    def test_qnn_batch(self):
        """Batched Opflow QNN Test on a network with an instance of QuantumInstance."""
        self._test_qnn_batch(self.qnn)

    def test_no_quantum_instance(self):
        """Simple Opflow QNN Test on a network without QuantumInstance."""
        self._test_qnn_simple(self.qnn_no_qi)

    def test_no_quantum_instance_batch(self):
        """Batched Opflow QNN Test on a network without QuantumInstance."""
        self._test_qnn_batch(self.qnn_no_qi)


if __name__ == '__main__':
    unittest.main()
