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
        feature_map = ZZFeatureMap(2)
        var_form = RealAmplitudes(2, reps=1)
        self.qnn = TwoLayerQNN(2, feature_map=feature_map,
                               var_form=var_form, quantum_instance=quantum_instance)

    def test_two_layer_qnn1(self):
        """ Opflow QNN Test """

        input_data = np.zeros(self.qnn.num_inputs)
        weights = np.zeros(self.qnn.num_weights)

        # test forward pass
        result = self.qnn.forward(input_data, weights)
        self.assertEqual(result.shape, self.qnn.output_shape)

        # test backward pass
        result = self.qnn.backward(input_data, weights)
        self.assertEqual(result[0].shape, (*self.qnn.output_shape, self.qnn.num_inputs))
        self.assertEqual(result[1].shape, (*self.qnn.output_shape, self.qnn.num_weights))


if __name__ == '__main__':
    unittest.main()
