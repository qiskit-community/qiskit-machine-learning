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

"""Test Torch Connector."""

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector


class TestCircuitQNN(QiskitMachineLearningTestCase):
    """Torch Connector Tests."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        self.sv_quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.qasm_quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100)

    def test_torch_with_opflow_qnn(self):
        """Torch Connector + Opflow QNN Test."""
        try:
            import torch
        except ImportError as ex:
            self.skipTest("Torch doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        # create QNN
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits)
        var_form = RealAmplitudes(num_qubits, reps=1)
        qnn = TwoLayerQNN(num_qubits=2, feature_map=feature_map, var_form=var_form,
                          quantum_instance=self.sv_quantum_instance)

        # connect to torch
        torch_qnn = TorchConnector(qnn)

        # test single input
        input_data = torch.Tensor(np.ones(qnn.num_inputs))
        output = torch_qnn(input_data)
        self.assertEqual(output.shape, (1,))

        # test batch input
        # TODO

        # test autograd
        func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
        input_data = (
            torch.randn(qnn.num_inputs, dtype=torch.double, requires_grad=True),
            torch.randn(qnn.num_weights, dtype=torch.double, requires_grad=True),
            qnn
        )
        test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)
        self.assertTrue(test)

    def test_torch_with_circuit_qnn(self):
        """Torch Connector + Circuit QNN Test."""
        try:
            import torch
        except ImportError as ex:
            self.skipTest("Torch doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        # create QNN
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits)
        var_form = RealAmplitudes(num_qubits, reps=1)
        qc = feature_map.copy()
        qc.append(var_form, range(feature_map.num_qubits))
        qnn = CircuitQNN(qc, input_params=feature_map.parameters, weight_params=var_form.parameters,
                         quantum_instance=self.qasm_quantum_instance)

        # connect to torch
        torch_qnn = TorchConnector(qnn)

        # test single input
        input_data = torch.Tensor(np.ones(qnn.num_inputs))
        output = torch_qnn(input_data)
        self.assertEqual(output.shape, (1,))

        # test batch input
        # TODO

        # test autograd
        func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
        input_data = (
            torch.randn(qnn.num_inputs, dtype=torch.double, requires_grad=True),
            torch.randn(qnn.num_weights, dtype=torch.double, requires_grad=True),
            qnn
        )
        test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)
        self.assertTrue(test)

    def test_torch_with_classical_nn(self):
        """Torch Connector + Classical NN Test."""
        # TODO
        pass


if __name__ == '__main__':
    unittest.main()
