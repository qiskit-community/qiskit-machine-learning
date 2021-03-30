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

from typing import List

from test import QiskitMachineLearningTestCase

import numpy as np

from ddt import ddt, data

try:
    from torch import Tensor
    from torch.nn import CrossEntropyLoss
except ImportError:
    class Tensor:  # type: ignore
        """ Empty Tensor class
            Replacement if torch.Tensor is not present.
        """
        pass

    class CrossEntropyLoss:  # type: ignore
        """ Empty Tensor class
            Replacement if torch.nn.CrossEntropyLoss is not present.
        """
        pass

from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit.opflow import StateFn, ListOp, PauliSumOp

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN, OpflowQNN
from qiskit_machine_learning.connectors import TorchConnector


@ddt
class TestTorchConnector(QiskitMachineLearningTestCase):
    """Torch Connector Tests."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        self.sv_quantum_instance = QuantumInstance(StatevectorSimulator())
        self.qasm_quantum_instance = QuantumInstance(QasmSimulator(), shots=100)

    def validate_output_shape(self, model: TorchConnector, test_data: List[Tensor]) -> None:
        """Creates a Linear PyTorch module with the same in/out dimensions as the given model,
            applies the list of test input data to both, and asserts that they have the same
            output shape.

            Args:
                model: model to be tested
                test_data: list of test input tensors

            Raises:
                MissingOptionalLibraryError: torch not installed
                QiskitMachineLearningError: Invalid input.
        """
        try:
            from torch.nn import Linear
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname='Pytorch',
                name='TorchConnector',
                pip_install="pip install 'qiskit-machine-learning[torch]'") from ex

        # create benchmark model
        in_dim = model.neural_network.num_inputs
        out_dim = 0
        if len(model.neural_network.output_shape) != 1:
            raise QiskitMachineLearningError('Function only works for one dimensional output')
        out_dim = model.neural_network.output_shape[0]
        linear = Linear(in_dim, out_dim)

        # iterate over test data and validate behavior of model
        for x in test_data:

            # test linear model and track whether it failed or store the output shape
            c_worked = True
            try:
                c_shape = linear(x).shape
            except Exception:  # pylint: disable=broad-except
                c_worked = False

            # test quantum model and track whether it failed or store the output shape
            q_worked = True
            try:
                q_shape = model(x).shape
            except Exception:  # pylint: disable=broad-except
                q_worked = False

            # compare results and assert that the behavior is equal
            self.assertEqual(c_worked, q_worked)
            if c_worked:
                self.assertEqual(c_shape, q_shape)

    def validate_backward_pass(self, model: TorchConnector) -> None:
        """Uses PyTorch to validate the backward pass / autograd.

        Args:
            model: The model to be tested.
        """
        try:
            import torch
        except ImportError as ex:
            self.skipTest('pytorch not installed, skipping test: {}'.format(str(ex)))

        # test autograd
        func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
        input_data = (
            torch.randn(model.neural_network.num_inputs, dtype=torch.double, requires_grad=True),
            torch.randn(model.neural_network.num_weights, dtype=torch.double, requires_grad=True),
            model.neural_network,
            False
        )
        test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)  # type: ignore
        self.assertTrue(test)

    @data(
        'sv', 'qasm'
    )
    def test_opflow_qnn_1_1(self, q_i):
        """ Test Torch Connector + Opflow QNN with input/output dimension 1/1."""

        if q_i == 'sv':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        # construct simple feature map
        param_x = Parameter('x')
        feature_map = QuantumCircuit(1, name='fm')
        feature_map.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter('y')
        var_form = QuantumCircuit(1, name='vf')
        var_form.ry(param_y, 0)

        # construct QNN with statevector simulator
        qnn = TwoLayerQNN(1, feature_map, var_form, quantum_instance=quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor(1),
                Tensor([1]),
                Tensor([1, 2]),
                Tensor([[1], [2]]),
                Tensor([[[1], [2]], [[3], [4]]])
            ]

            # test model
            self.validate_output_shape(model, test_data)
            if q_i == 'sv':
                self.validate_backward_pass(model)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data(
        'sv', 'qasm'
    )
    def test_opflow_qnn_2_1(self, q_i):
        """ Test Torch Connector + Opflow QNN with input/output dimension 2/1."""

        if q_i == 'sv':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        # construct QNN
        qnn = TwoLayerQNN(2, quantum_instance=quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor(1),
                Tensor([1, 2]),
                Tensor([[1, 2]]),
                Tensor([[1], [2]]),
                Tensor([[[1], [2]], [[3], [4]]])
            ]

            # test model
            self.validate_output_shape(model, test_data)
            if q_i == 'sv':
                self.validate_backward_pass(model)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data(
        'sv', 'qasm'
    )
    def test_opflow_qnn_2_2(self, q_i):
        """ Test Torch Connector + Opflow QNN with input/output dimension 2/2."""

        if q_i == 'sv':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        # construct parametrized circuit
        params_1 = [Parameter('input1'), Parameter('weight1')]
        qc_1 = QuantumCircuit(1)
        qc_1.h(0)
        qc_1.ry(params_1[0], 0)
        qc_1.rx(params_1[1], 0)
        qc_sfn_1 = StateFn(qc_1)

        # construct cost operator
        h_1 = StateFn(PauliSumOp.from_list([('Z', 1.0), ('X', 1.0)]))

        # combine operator and circuit to objective function
        op_1 = ~h_1 @ qc_sfn_1

        # construct parametrized circuit
        params_2 = [Parameter('input2'), Parameter('weight2')]
        qc_2 = QuantumCircuit(1)
        qc_2.h(0)
        qc_2.ry(params_2[0], 0)
        qc_2.rx(params_2[1], 0)
        qc_sfn_2 = StateFn(qc_2)

        # construct cost operator
        h_2 = StateFn(PauliSumOp.from_list([('Z', 1.0), ('X', 1.0)]))

        # combine operator and circuit to objective function
        op_2 = ~h_2 @ qc_sfn_2

        op = ListOp([op_1, op_2])

        qnn = OpflowQNN(op, [params_1[0], params_2[0]], [params_1[1], params_2[1]],
                        quantum_instance=quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor(1),
                Tensor([1, 2]),
                Tensor([[1], [2]]),
                Tensor([[1, 2], [3, 4]])
            ]

            # test model
            self.validate_output_shape(model, test_data)
            if q_i == 'sv':
                self.validate_backward_pass(model)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, 'sv'),
        (None, None, True, 'sv'),
        (lambda x: np.sum(x) % 2, 2, False, 'sv'),
        (lambda x: np.sum(x) % 2, 2, True, 'sv'),
        (None, None, False, 'qasm'),
        (None, None, True, 'qasm'),
        (lambda x: np.sum(x) % 2, 2, False, 'qasm'),
        (lambda x: np.sum(x) % 2, 2, True, 'qasm'),
    )
    def test_circuit_qnn_1_1(self, config):
        """Torch Connector + Circuit QNN with no interpret, dense output,
        and input/output shape 1/1 ."""

        interpret, output_shape, sparse, q_i = config
        if q_i == 'sv':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        qc = QuantumCircuit(1)

        # construct simple feature map
        param_x = Parameter('x')
        qc.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter('y')
        qc.ry(param_y, 0)

        qnn = CircuitQNN(qc, [param_x], [param_y],
                         sparse=sparse,
                         sampling=False,
                         interpret=interpret,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor(1),
                Tensor([1, 2]),
                Tensor([[1], [2]]),
                Tensor([[[1], [2]], [[3], [4]]])
            ]

            # test model
            self.validate_output_shape(model, test_data)
            if q_i == 'sv':
                self.validate_backward_pass(model)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, 'sv'),
        (None, None, True, 'sv'),
        (lambda x: np.sum(x) % 2, 2, False, 'sv'),
        (lambda x: np.sum(x) % 2, 2, True, 'sv'),
        (None, None, False, 'qasm'),
        (None, None, True, 'qasm'),
        (lambda x: np.sum(x) % 2, 2, False, 'qasm'),
        (lambda x: np.sum(x) % 2, 2, True, 'qasm'),
    )
    def test_circuit_qnn_1_8(self, config):
        """Torch Connector + Circuit QNN with no interpret, dense output,
        and input/output shape 1/8 ."""

        interpret, output_shape, sparse, q_i = config
        if q_i == 'sv':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        qc = QuantumCircuit(3)

        # construct simple feature map
        param_x = Parameter('x')
        qc.ry(param_x, range(3))

        # construct simple feature map
        param_y = Parameter('y')
        qc.ry(param_y, range(3))

        qnn = CircuitQNN(qc, [param_x], [param_y],
                         sparse=sparse,
                         sampling=False,
                         interpret=interpret,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor(1),
                Tensor([1, 2]),
                Tensor([[1], [2]]),
                Tensor([[[1], [2]], [[3], [4]]])
            ]

            # test model
            self.validate_output_shape(model, test_data)
            if q_i == 'sv':
                self.validate_backward_pass(model)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, 'sv'),
        (None, None, True, 'sv'),
        (lambda x: np.sum(x) % 2, 2, False, 'sv'),
        (lambda x: np.sum(x) % 2, 2, True, 'sv'),
        (None, None, False, 'qasm'),
        (None, None, True, 'qasm'),
        (lambda x: np.sum(x) % 2, 2, False, 'qasm'),
        (lambda x: np.sum(x) % 2, 2, True, 'qasm'),
    )
    def test_circuit_qnn_2_4(self, config):
        """Torch Connector + Circuit QNN with no interpret, dense output,
        and input/output shape 1/8 ."""

        interpret, output_shape, sparse, q_i = config
        if q_i == 'sv':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        qc = QuantumCircuit(2)

        # construct simple feature map
        param_x_1, param_x_2 = Parameter('x1'), Parameter('x2')
        qc.ry(param_x_1, range(2))
        qc.ry(param_x_2, range(2))

        # construct simple feature map
        param_y = Parameter('y')
        qc.ry(param_y, range(2))

        qnn = CircuitQNN(qc, [param_x_1, param_x_2], [param_y],
                         sparse=sparse,
                         sampling=False,
                         interpret=interpret,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor(1),
                Tensor([1, 2]),
                Tensor([[1], [2]]),
                Tensor([[1, 2], [3, 4]]),
                Tensor([[[1], [2]], [[3], [4]]])
            ]

            # test model
            self.validate_output_shape(model, test_data)
            if q_i == 'sv':
                self.validate_backward_pass(model)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data(
        # interpret
        (None),
        (lambda x: np.sum(x) % 2)
    )
    def test_circuit_qnn_sampling(self, interpret):
        """Test Torch Connector + Circuit QNN for sampling."""

        qc = QuantumCircuit(2)

        # construct simple feature map
        param_x1, param_x2 = Parameter('x1'), Parameter('x2')
        qc.ry(param_x1, range(2))
        qc.ry(param_x2, range(2))

        # construct simple feature map
        param_y = Parameter('y')
        qc.ry(param_y, range(2))

        qnn = CircuitQNN(qc, [param_x1, param_x2], [param_y],
                         sparse=False,
                         sampling=True,
                         interpret=interpret,
                         output_shape=None,
                         quantum_instance=self.qasm_quantum_instance)
        try:
            model = TorchConnector(qnn)

            test_data = [
                Tensor([2, 2])
                # TODO: test batching
            ]
            for i, x in enumerate(test_data):
                if i == 0:
                    self.assertEqual(model(x).shape, qnn.output_shape)
                else:
                    # TODO: test batching
                    pass
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_batch_gradients(self):
        """Test backward pass for batch input."""

        # construct random data set
        num_inputs = 2
        num_samples = 10
        x = np.random.rand(num_samples, num_inputs)

        # set up QNN
        qnn = TwoLayerQNN(num_qubits=num_inputs, quantum_instance=self.sv_quantum_instance)

        # set up PyTorch module
        initial_weights = np.random.rand(qnn.num_weights)
        model = TorchConnector(qnn, initial_weights=initial_weights)

        # test single gradient
        w = model.weights.detach().numpy()
        res_qnn = qnn.forward(x[0, :], w)

        # construct finite difference gradient for weights
        eps = 1e-4
        grad = np.zeros(w.shape)
        for k in range(len(w)):
            delta = np.zeros(w.shape)
            delta[k] += eps

            f_1 = qnn.forward(x[0, :], w + delta)
            f_2 = qnn.forward(x[0, :], w - delta)

            grad[k] = (f_1 - f_2) / (2*eps)

        grad_qnn = qnn.backward(x[0, :], w)[1][0, 0, :]
        self.assertAlmostEqual(np.linalg.norm(grad - grad_qnn), 0.0, places=4)

        model.zero_grad()
        res_model = model(Tensor(x[0, :]))
        self.assertAlmostEqual(np.linalg.norm(res_model.detach().numpy() - res_qnn[0]), 0.0,
                               places=4)
        res_model.backward()
        grad_model = model.weights.grad
        self.assertAlmostEqual(np.linalg.norm(grad_model.detach().numpy() - grad_qnn), 0.0,
                               places=4)

        # test batch input
        batch_grad = np.zeros((*w.shape, num_samples, 1))
        for k in range(len(w)):
            delta = np.zeros(w.shape)
            delta[k] += eps

            f_1 = qnn.forward(x, w + delta)
            f_2 = qnn.forward(x, w - delta)

            batch_grad[k] = (f_1 - f_2) / (2*eps)

        batch_grad = np.sum(batch_grad, axis=1)
        batch_grad_qnn = np.sum(qnn.backward(x, w)[1], axis=0)
        self.assertAlmostEqual(np.linalg.norm(batch_grad - batch_grad_qnn.transpose()),
                               0.0, places=4)

        model.zero_grad()
        batch_res_model = sum(model(Tensor(x)))
        batch_res_model.backward()
        self.assertAlmostEqual(
            np.linalg.norm(model.weights.grad.numpy() - batch_grad.transpose()[0]), 0.0, places=4)

    def test_classification_with_pytorch(self):
        """Test classification with PyTorch."""

        num_inputs = 2
        num_samples = 20
        X = 2*np.random.rand(num_samples, num_inputs) - 1  # pylint: disable=invalid-name
        y01 = 1*(np.sum(X, axis=1) >= 0)  # in { 0,  1}

        X_ = Tensor(X)  # pylint: disable=invalid-name
        y01_ = Tensor(y01).reshape(len(y01)).long()

        feature_map = ZZFeatureMap(num_inputs)
        var_form = RealAmplitudes(num_inputs, entanglement='linear', reps=1)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(var_form, range(num_inputs))

        def parity(x):
            return '{:b}'.format(x).count('1') % 2
        output_shape = 2  # parity = 0, 1

        qnn2 = CircuitQNN(qc, input_params=feature_map.parameters,
                          weight_params=var_form.parameters,
                          interpret=parity, output_shape=output_shape,
                          quantum_instance=self.sv_quantum_instance)

        # set up PyTorch module
        initial_weights = 0.1*(2*np.random.rand(qnn2.num_weights) - 1)
        model2 = TorchConnector(qnn2, initial_weights)
        model2.train()

        f_loss = CrossEntropyLoss()

        output = model2(X_)
        loss = f_loss(output, y01_)
        loss.backward()

        for x in model2.parameters():
            print(x)

        print('done')


if __name__ == '__main__':
    unittest.main()
