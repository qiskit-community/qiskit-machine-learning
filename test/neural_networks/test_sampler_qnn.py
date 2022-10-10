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
import itertools
import unittest
import numpy as np
from test import QiskitMachineLearningTestCase

from ddt import ddt, data, idata, unpack

from qiskit.circuit import QuantumCircuit, Parameter
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DEFAULT = "default"
SHOTS = "shots"
SAMPLING = [True, False] # TODO
SAMPLERS = [DEFAULT, SHOTS]
INTERPRET_TYPES = [0, 1, 2]
BATCH_SIZES = [1, 2]

@ddt
class TestSamplerQNN(QiskitMachineLearningTestCase):
    """Sampler QNN Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345

        # # define test circuit
        # num_qubits = 3
        # self.qc = RealAmplitudes(num_qubits, entanglement="linear", reps=1)

        # define feature map and ansatz
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits, reps=1)
        var_form = RealAmplitudes(num_qubits, reps=1)

        # construct circuit
        self.qc = QuantumCircuit(num_qubits)
        self.qc.append(feature_map, range(2))
        self.qc.append(var_form, range(2))

        # store params
        self.input_params = list(feature_map.parameters)
        self.weight_params = list(var_form.parameters)

        # define interpret functions
        def interpret_1d(x):
            return sum((s == "1" for s in f"{x:0b}")) % 2

        self.interpret_1d = interpret_1d
        self.output_shape_1d = 2  # takes values in {0, 1}

        def interpret_2d(x):
            return np.array([self.interpret_1d(x), 2 * self.interpret_1d(x)])

        self.interpret_2d = interpret_2d
        self.output_shape_2d = (
            2,
            3,
        )  # 1st dim. takes values in {0, 1} 2nd dim in {0, 1, 2}

        # define sampler primitives
        self.sampler = Sampler()
        self.sampler_shots = Sampler(options={"shots":100})

    def _get_qnn(self, sampler_type, interpret_id):
        """Construct QNN from configuration."""

        # get quantum instance
        if sampler_type == SHOTS:
            sampler = self.sampler_shots
        elif sampler_type == DEFAULT:
            sampler = self.sampler
        else:
            sampler = None

        # get interpret setting
        interpret = None
        output_shape = None
        if interpret_id == 1:
            interpret = self.interpret_1d
            output_shape = self.output_shape_1d
        elif interpret_id == 2:
            interpret = self.interpret_2d
            output_shape = self.output_shape_2d

        # construct QNN
        qnn = SamplerQNN(
            sampler,
            self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=interpret,
            output_shape=output_shape,
        )
        return qnn

    def _verify_qnn(
        self,
        qnn: CircuitQNN,
        sampler_type: str,
        batch_size: int,
    ) -> None:
        """
        Verifies that a QNN functions correctly

        Args:
            qnn: a QNN to check
            sampler_type:
            batch_size:

        Returns:
            None.
        """
        input_data = np.zeros((batch_size, qnn.num_inputs))
        weights = np.zeros(qnn.num_weights)

        # evaluate QNN forward pass
        result = qnn.forward(input_data, weights)
        self.assertTrue(isinstance(result, np.ndarray))
        # check forward result shape
        self.assertEqual(result.shape, (batch_size, *qnn.output_shape))

        # evaluate QNN backward pass
        input_grad, weights_grad = qnn.backward(input_data, weights)

        self.assertIsNone(input_grad)
        # verify that input gradients are None if turned off
        self.assertEqual(
            weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights)
        )

        # verify that input gradients are not None if turned on
        qnn.input_gradients = True
        input_grad, weights_grad = qnn.backward(input_data, weights)

        self.assertEqual(input_grad.shape, (batch_size, *qnn.output_shape, qnn.num_inputs))
        self.assertEqual(
            weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights)
            )

    @idata(itertools.product(SAMPLERS, INTERPRET_TYPES, BATCH_SIZES))
    @unpack
    def test_sampler_qnn(
        self, sampler_type, interpret_type, batch_size
    ):
        """Sampler QNN Test."""
        qnn = self._get_qnn(sampler_type, interpret_type)
        self._verify_qnn(qnn, sampler_type, batch_size)


    #
    # def test_forward_pass(self):
    #
    #     parity = lambda x: "{:b}".format(x).count("1") % 2
    #     output_shape = 2  # this is required in case of a callable with dense output
    #
    #     circuit_qnn = CircuitQNN(
    #         self.qc,
    #         input_params=self.qc.parameters[:3],
    #         weight_params=self.qc.parameters[3:],
    #         sparse=False,
    #         interpret=parity,
    #         output_shape=output_shape,
    #         quantum_instance=self.qi_qasm,
    #     )
    #
    #     sampler_qnn = SamplerQNN(
    #         sampler=self.sampler,
    #         circuit=self.qc,
    #         input_params=self.qc.parameters[:3],
    #         weight_params=self.qc.parameters[3:],
    #         interpret=parity,
    #         output_shape=output_shape,
    #     )
    #
    #     inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
    #     weights = algorithm_globals.random.random(circuit_qnn._num_weights)
    #
    #     circuit_qnn_fwd = circuit_qnn.forward(inputs, weights)
    #     sampler_qnn_fwd = sampler_qnn.forward(inputs, weights)
    #
    #     np.testing.assert_array_almost_equal(
    #         np.asarray(sampler_qnn_fwd), np.asarray(circuit_qnn_fwd), 0.1
    #     )
    #
    # def test_backward_pass(self):
    #
    #     parity = lambda x: "{:b}".format(x).count("1") % 2
    #     output_shape = 2  # this is required in case of a callable with dense output
    #     from qiskit.opflow import Gradient
    #
    #     circuit_qnn = CircuitQNN(
    #         self.qc,
    #         input_params=self.qc.parameters[:3],
    #         weight_params=self.qc.parameters[3:],
    #         sparse=False,
    #         interpret=parity,
    #         output_shape=output_shape,
    #         quantum_instance=self.qi_qasm,
    #         gradient=Gradient("param_shift"),
    #     )
    #
    #     sampler_qnn = SamplerQNN(
    #         sampler=self.sampler,
    #         circuit=self.qc,
    #         input_params=self.qc.parameters[:3],
    #         weight_params=self.qc.parameters[3:],
    #         interpret=parity,
    #         output_shape=output_shape,
    #         gradient=ParamShiftSamplerGradient(self.sampler),
    #     )
    #
    #     inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
    #     weights = algorithm_globals.random.random(circuit_qnn._num_weights)
    #
    #     circuit_qnn_fwd = circuit_qnn.backward(inputs, weights)
    #     sampler_qnn_fwd = sampler_qnn.backward(inputs, weights)
    #
    #     print(circuit_qnn_fwd)
    #     print(sampler_qnn_fwd)
    #     np.testing.assert_array_almost_equal(
    #         np.asarray(sampler_qnn_fwd[1]), np.asarray(circuit_qnn_fwd[1]), 0.1
    #     )
    #
    # def test_input_gradients(self):
    #
    #     parity = lambda x: "{:b}".format(x).count("1") % 2
    #     output_shape = 2  # this is required in case of a callable with dense output
    #     from qiskit.opflow import Gradient
    #
    #     circuit_qnn = CircuitQNN(
    #         self.qc,
    #         input_params=self.qc.parameters[:3],
    #         weight_params=self.qc.parameters[3:],
    #         sparse=False,
    #         interpret=parity,
    #         output_shape=output_shape,
    #         quantum_instance=self.qi_qasm,
    #         gradient=Gradient("param_shift"),
    #         input_gradients=True,
    #     )
    #
    #     sampler_qnn = SamplerQNN(
    #         sampler=self.sampler,
    #         circuit=self.qc,
    #         input_params=self.qc.parameters[:3],
    #         weight_params=self.qc.parameters[3:],
    #         interpret=parity,
    #         output_shape=output_shape,
    #         gradient=ParamShiftSamplerGradient(self.sampler),
    #         input_gradients=True,
    #     )
    #
    #     inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
    #     weights = algorithm_globals.random.random(circuit_qnn._num_weights)
    #
    #     circuit_qnn_fwd = circuit_qnn.backward(inputs, weights)
    #     sampler_qnn_fwd = sampler_qnn.backward(inputs, weights)
    #
    #     print(circuit_qnn_fwd)
    #     print(sampler_qnn_fwd)
    #     np.testing.assert_array_almost_equal(
    #         np.asarray(sampler_qnn_fwd), np.asarray(circuit_qnn_fwd), 0.1
    #     )
    #
    #     from qiskit_machine_learning.connectors import TorchConnector
    #     import torch
    #
    #     model = TorchConnector(sampler_qnn)
    #     func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
    #     input_data = (
    #         torch.randn(
    #             model.neural_network.num_inputs,
    #             dtype=torch.double,
    #             requires_grad=True,
    #         ),
    #         torch.randn(
    #             model.neural_network.num_weights,
    #             dtype=torch.double,
    #             requires_grad=True,
    #         ),
    #         model.neural_network,
    #         False,
    #     )
    #     test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)  # type: ignore
    #     self.assertTrue(test)

    # def test_torch_connector(self):
    #     from qiskit_machine_learning.connectors import TorchConnector
