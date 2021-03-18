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

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

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
        self._quantum_instance = QuantumInstance(backend)

    def test_opflow_qnn_simple(self):
        """Simple Opflow QNN Test."""

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
        qnn = OpflowQNN(op, [params[0]], [params[1]], expval,
                             gradient, quantum_instance=self._quantum_instance)

        input_data = np.zeros(qnn.num_inputs)
        weights = np.zeros(qnn.num_weights)

        # test forward pass
        result = qnn.forward(input_data, weights)
        self.assertEqual(result.shape, qnn.output_shape)

        # test backward pass
        result = qnn.backward(input_data, weights)
        # batch dimension of 1
        self.assertEqual(result[0].shape, (1, *qnn.output_shape, qnn.num_inputs))
        self.assertEqual(result[1].shape, (1, *qnn.output_shape, qnn.num_weights))

    def _test_opflow_batch(self):
        expval = PauliExpectation()
        gradient = Gradient()

        # construct parametrized circuit
        input_params = [Parameter('input1'), Parameter('input2')]
        weight_params = [Parameter('weight1'), Parameter('weight2')]
        qc = QuantumCircuit(2)
        qc.h(range(2))
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.rx(weight_params[0], 0)
        qc.rx(weight_params[1], 1)
        qc_sfn = StateFn(qc)

        # construct cost operator
        cost_operator = StateFn(PauliSumOp.from_list([('Z', 1.0), ('X', 1.0)]))

        # combine operator and circuit to objective function
        op = ~cost_operator @ qc_sfn
        print(op)
        op = op.bind_parameters({input_params[0]: 1, input_params[1]: 2, weight_params[0]: 3, weight_params[1]: 4})
        qc = expval.convert(op).to_circuit()
        print(qc)

        # define QNN
        qnn = OpflowQNN(op, input_params, weight_params, expval,
                        gradient, quantum_instance=self._quantum_instance)

        batch_size = 10
        input_data = np.zeros((batch_size, qnn.num_inputs))
        weights = np.zeros(qnn.num_weights)
        result = qnn.forward(input_data, weights)
        print(result)


if __name__ == '__main__':
    unittest.main()
