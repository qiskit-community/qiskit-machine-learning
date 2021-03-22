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

"""Test Neural Network."""

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data

from qiskit_machine_learning.neural_networks import NeuralNetwork


class _NeuralNetwork(NeuralNetwork):
    """Dummy implementation to test the abstract neural network class."""

    def _forward(self, input_data, weights):
        """Expects as input either None, or a 2-dim array and returns."""

        # handle None input
        if self.num_inputs == 0 and input_data is None:
            return np.zeros(self.output_shape)

        return np.zeros(self.output_shape)

    def _backward(self, input_data, weights):
        # return None if there are no weights
        input_grad = None
        if self.num_inputs > 0:
            input_grad = np.zeros((*self.output_shape, self.num_inputs))

        weight_grad = None
        if self.num_weights > 0:
            weight_grad = np.zeros((*self.output_shape, self.num_weights))

        return input_grad, weight_grad


@ddt
class TestNeuralNetwork(QiskitMachineLearningTestCase):
    """Neural Network Tests."""

    @data(
        # no input
        ((0, 0, True, 1), None),
        ((0, 1, True, 1), None),
        ((0, 1, True, 2), None),
        ((0, 1, True, (2, 2)), None),

        # 1d input
        ((1, 0, True, 1), 0),
        ((1, 1, True, 1), 0),
        ((1, 1, True, 2), 0),
        ((1, 1, True, (2, 2)), 0),

        # multi-dimensional input and weights
        ((2, 2, True, (2, 2)), [0, 0])
    )
    def test_forward_shape(self, params):
        """Test forward shape."""

        config, input_data = params
        network = _NeuralNetwork(*config)

        shape = network.forward(input_data, np.zeros(network.num_weights)).shape
        self.assertEqual(shape, network.output_shape)

    @data(
        # no input
        ((0, 0, True, 1), None),
        ((0, 1, True, 1), None),
        ((0, 1, True, 2), None),
        ((0, 1, True, (2, 2)), None),

        # 1d input
        ((1, 0, True, 1), 0),
        ((1, 1, True, 1), 0),
        ((1, 1, True, 2), 0),
        ((1, 1, True, (2, 2)), 0),

        # multi-dimensional input and weights
        ((2, 2, True, (2, 2)), [0, 0])
    )
    def test_backward_shape(self, params):
        """ Test backward shape """

        config, input_data = params
        network = _NeuralNetwork(*config)

        input_grad, weights_grad = network.backward(input_data, np.zeros(network.num_weights))

        if network.num_inputs > 0:
            self.assertEqual(input_grad.shape, (*network.output_shape, network.num_inputs))
        else:
            self.assertEqual(input_grad, None)

        if network.num_weights > 0:
            self.assertEqual(weights_grad.shape, (*network.output_shape, network.num_weights))
        else:
            self.assertEqual(weights_grad, None)


if __name__ == '__main__':
    unittest.main()
