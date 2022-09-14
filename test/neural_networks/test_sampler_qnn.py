# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Sampler QNN with Terra primitives."""
import numpy as np
from test import QiskitMachineLearningTestCase

from qiskit.primitives import Sampler
from qiskit.algorithms.gradients import ParamShiftSamplerGradient, FiniteDiffSamplerGradient
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.neural_networks.sampler_qnn import SamplerQNN

algorithm_globals.random_seed = 42
from test.connectors.test_torch import TestTorch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class TestSamplerQNN(QiskitMachineLearningTestCase):
    """Sampler QNN Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345

        # define test circuit
        num_qubits = 3
        self.qc = RealAmplitudes(num_qubits, entanglement="linear", reps=1)
        self.qi_qasm = QuantumInstance(Aer.get_backend("aer_simulator"))
        self.sampler = Sampler()

    def test_forward_pass(self):

        parity = lambda x: "{:b}".format(x).count("1") % 2
        output_shape = 2  # this is required in case of a callable with dense output

        circuit_qnn = CircuitQNN(
            self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=self.qi_qasm,
        )

        sampler_qnn = SamplerQNN(
            sampler=self.sampler,
            circuit=self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            interpret=parity,
            output_shape=output_shape,
        )

        inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
        weights = algorithm_globals.random.random(circuit_qnn._num_weights)

        circuit_qnn_fwd = circuit_qnn.forward(inputs, weights)
        sampler_qnn_fwd = sampler_qnn.forward(inputs, weights)

        np.testing.assert_array_almost_equal(
            np.asarray(sampler_qnn_fwd), np.asarray(circuit_qnn_fwd), 0.1
        )

    def test_backward_pass(self):

        parity = lambda x: "{:b}".format(x).count("1") % 2
        output_shape = 2  # this is required in case of a callable with dense output
        from qiskit.opflow import Gradient
        circuit_qnn = CircuitQNN(
            self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=self.qi_qasm,
            gradient=Gradient("param_shift"),
        )

        sampler_qnn = SamplerQNN(
            sampler=self.sampler,
            circuit=self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            interpret=parity,
            output_shape=output_shape,
            gradient=ParamShiftSamplerGradient(self.sampler),
        )

        inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
        weights = algorithm_globals.random.random(circuit_qnn._num_weights)

        circuit_qnn_fwd = circuit_qnn.backward(inputs, weights)
        sampler_qnn_fwd = sampler_qnn.backward(inputs, weights)

        print(circuit_qnn_fwd)
        print(sampler_qnn_fwd)
        np.testing.assert_array_almost_equal(
            np.asarray(sampler_qnn_fwd[1]), np.asarray(circuit_qnn_fwd[1]), 0.1
        )

    def test_input_gradients(self):

        parity = lambda x: "{:b}".format(x).count("1") % 2
        output_shape = 2  # this is required in case of a callable with dense output
        from qiskit.opflow import Gradient
        circuit_qnn = CircuitQNN(
            self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=self.qi_qasm,
            gradient=Gradient("param_shift"),
            input_gradients=True
        )

        sampler_qnn = SamplerQNN(
            sampler=self.sampler,
            circuit=self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            interpret=parity,
            output_shape=output_shape,
            gradient=ParamShiftSamplerGradient(self.sampler),
            input_gradients=True

        )

        inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
        weights = algorithm_globals.random.random(circuit_qnn._num_weights)

        circuit_qnn_fwd = circuit_qnn.backward(inputs, weights)
        sampler_qnn_fwd = sampler_qnn.backward(inputs, weights)

        print(circuit_qnn_fwd)
        print(sampler_qnn_fwd)
        np.testing.assert_array_almost_equal(
            np.asarray(sampler_qnn_fwd), np.asarray(circuit_qnn_fwd), 0.1
        )

        from qiskit_machine_learning.connectors import TorchConnector
        import torch

        model = TorchConnector(sampler_qnn)
        func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
        input_data = (
            torch.randn(
                model.neural_network.num_inputs,
                dtype=torch.double,
                requires_grad=True,
            ),
            torch.randn(
                model.neural_network.num_weights,
                dtype=torch.double,
                requires_grad=True,
            ),
            model.neural_network,
            False,
        )
        test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)  # type: ignore
        self.assertTrue(test)

    # def test_torch_connector(self):
    #     from qiskit_machine_learning.connectors import TorchConnector
