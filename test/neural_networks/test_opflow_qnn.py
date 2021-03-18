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

""" Test Opflow QNN """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from qiskit import Aer
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliExpectation, Gradient, StateFn, PauliSumOp, ListOp, X, Y, Z
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import OpflowQNN


class TestOpflowQNN(QiskitMachineLearningTestCase):
    """Opflow QNN Tests."""

    def setUp(self):
        super().setUp()

        # specify "run configuration"
        backend = Aer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend)

        # specify how to evaluate expected values and gradients
        expval = PauliExpectation()
        gradient = Gradient()

        # construct parametrized circuit
        params = [Parameter('input1'), Parameter('weight1')]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        qc_sfn = StateFn(qc)

        # construct cost operator
        cost_operator = StateFn(PauliSumOp.from_list([('Z', 1.0), ('X', 1.0)]))

        # combine operator and circuit to objective function
        op = ~cost_operator @ qc_sfn

        # define QNN
        self.qnn = OpflowQNN(op, [params[0]], [params[1]], expval,
                             gradient, quantum_instance=quantum_instance)

    def test_opflow_qnn1(self):
        """ Opflow QNN Test """

        input_data = np.zeros(self.qnn.num_inputs)
        weights = np.zeros(self.qnn.num_weights)

        # test forward pass
        result = self.qnn.forward(input_data, weights)
        print(result)
        self.assertEqual(result.shape, self.qnn.output_shape)

        # test backward pass
        result = self.qnn.backward(input_data, weights)
        self.assertEqual(result[0].shape, (self.qnn.num_inputs, *self.qnn.output_shape))
        self.assertEqual(result[1].shape, (self.qnn.num_weights, *self.qnn.output_shape))

    # def test_opflow_batch(self):
    #     print(f"output shape {self.qnn._output_shape}")
    #     batch_size = 10
    #     input_data = np.zeros((batch_size, self.qnn.num_inputs))
    #     weights = np.zeros(self.qnn.num_weights)
    #
    #     result = self.qnn.forward(input_data, weights)
    #     print(result)
    #

if __name__ == '__main__':
    unittest.main()
