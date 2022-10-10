# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Torch Connector."""

from typing import List
from test.connectors.test_torch import TestTorch

import numpy as np

from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import StateFn, ListOp, PauliSumOp

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN, OpflowQNN
from qiskit_machine_learning.connectors import TorchConnector
import qiskit_machine_learning.optionals as _optionals


@ddt
class TestTorchConnector(TestTorch):
    """Torch Connector Tests."""

    def _validate_output_shape(self, model: TorchConnector, test_data: List) -> None:
        """Creates a Linear PyTorch module with the same in/out dimensions as the given model,
        applies the list of test input data to both, and asserts that they have the same
        output shape.

        Args:
            model: model to be tested
            test_data: list of test input tensors

        Raises:
            QiskitMachineLearningError: Invalid input.
        """
        from torch.nn import Linear

        # create benchmark model
        in_dim = model.neural_network.num_inputs
        if len(model.neural_network.output_shape) != 1:
            raise QiskitMachineLearningError("Function only works for one dimensional output")
        out_dim = model.neural_network.output_shape[0]
        # we target our tests to either cpu or gpu
        linear = Linear(in_dim, out_dim, device=self._device)
        model.to(self._device)

        # iterate over test data and validate behavior of model
        for x in test_data:
            x = x.to(self._device)
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
            with self.subTest("c_worked == q_worked", tensor=x):
                self.assertEqual(c_worked, q_worked)
            if c_worked and q_worked:
                with self.subTest("c_shape == q_shape", tensor=x):
                    self.assertEqual(c_shape, q_shape)

    def _validate_backward_pass(self, model: TorchConnector) -> None:
        """Uses PyTorch to validate the backward pass / autograd.

        Args:
            model: The model to be tested.
        """
        import torch

        # test autograd
        func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
        input_data = (
            torch.randn(
                model.neural_network.num_inputs,
                dtype=torch.double,
                requires_grad=True,
                device=self._device,
            ),
            torch.randn(
                model.neural_network.num_weights,
                dtype=torch.double,
                requires_grad=True,
                device=self._device,
            ),
            model.neural_network,
            False,
        )
        test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)  # type: ignore
        self.assertTrue(test)

    @data("sv", "qasm")
    def test_opflow_qnn_1_1(self, q_i):
        """Test Torch Connector + Opflow QNN with input/output dimension 1/1."""
        from torch import Tensor

        if q_i == "sv":
            quantum_instance = self._sv_quantum_instance
        else:
            quantum_instance = self._qasm_quantum_instance

        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        # construct QNN with statevector simulator
        qnn = TwoLayerQNN(
            1, feature_map, ansatz, quantum_instance=quantum_instance, input_gradients=True
        )
        model = TorchConnector(qnn)

        test_data = [
            Tensor([1]),
            Tensor([1, 2]),
            Tensor([[1], [2]]),
            Tensor([[[1], [2]], [[3], [4]]]),
        ]

        # test model
        self._validate_output_shape(model, test_data)
        if q_i == "sv":
            self._validate_backward_pass(model)

    @data("sv", "qasm")
    def test_opflow_qnn_2_1(self, q_i):
        """Test Torch Connector + Opflow QNN with input/output dimension 2/1."""
        from torch import Tensor

        if q_i == "sv":
            quantum_instance = self._sv_quantum_instance
        else:
            quantum_instance = self._qasm_quantum_instance

        # construct QNN
        qnn = TwoLayerQNN(2, quantum_instance=quantum_instance, input_gradients=True)
        model = TorchConnector(qnn)

        test_data = [
            Tensor([1]),
            Tensor([1, 2]),
            Tensor([[1, 2]]),
            Tensor([[1], [2]]),
            Tensor([[[1], [2]], [[3], [4]]]),
        ]

        # test model
        self._validate_output_shape(model, test_data)
        if q_i == "sv":
            self._validate_backward_pass(model)

    @data("sv", "qasm")
    def test_opflow_qnn_2_2(self, q_i):
        """Test Torch Connector + Opflow QNN with input/output dimension 2/2."""
        from torch import Tensor

        if q_i == "sv":
            quantum_instance = self._sv_quantum_instance
        else:
            quantum_instance = self._qasm_quantum_instance

        # construct parametrized circuit
        params_1 = [Parameter("input1"), Parameter("weight1")]
        qc_1 = QuantumCircuit(1)
        qc_1.h(0)
        qc_1.ry(params_1[0], 0)
        qc_1.rx(params_1[1], 0)
        qc_sfn_1 = StateFn(qc_1)

        # construct cost operator
        h_1 = StateFn(PauliSumOp.from_list([("Z", 1.0), ("X", 1.0)]))

        # combine operator and circuit to objective function
        op_1 = ~h_1 @ qc_sfn_1

        # construct parametrized circuit
        params_2 = [Parameter("input2"), Parameter("weight2")]
        qc_2 = QuantumCircuit(1)
        qc_2.h(0)
        qc_2.ry(params_2[0], 0)
        qc_2.rx(params_2[1], 0)
        qc_sfn_2 = StateFn(qc_2)

        # construct cost operator
        h_2 = StateFn(PauliSumOp.from_list([("Z", 1.0), ("X", 1.0)]))

        # combine operator and circuit to objective function
        op_2 = ~h_2 @ qc_sfn_2

        op = ListOp([op_1, op_2])

        qnn = OpflowQNN(
            op,
            [params_1[0], params_2[0]],
            [params_1[1], params_2[1]],
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)

        test_data = [
            Tensor([1]),
            Tensor([1, 2]),
            Tensor([[1], [2]]),
            Tensor([[1, 2], [3, 4]]),
        ]

        # test model
        self._validate_output_shape(model, test_data)
        if q_i == "sv":
            self._validate_backward_pass(model)

    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, "sv"),
        (None, None, True, "sv"),
        (lambda x: np.sum(x) % 2, 2, False, "sv"),
        (lambda x: np.sum(x) % 2, 2, True, "sv"),
        (None, None, False, "qasm"),
        (None, None, True, "qasm"),
        (lambda x: np.sum(x) % 2, 2, False, "qasm"),
        (lambda x: np.sum(x) % 2, 2, True, "qasm"),
    )
    def test_circuit_qnn_1_1(self, config):
        """Torch Connector + Circuit QNN with no interpret, dense output,
        and input/output shape 1/1 ."""
        from torch import Tensor

        interpret, output_shape, sparse, q_i = config
        if sparse and not _optionals.HAS_SPARSE:
            self.skipTest("sparse library is required to run this test")
            return
        if q_i == "sv":
            quantum_instance = self._sv_quantum_instance
        else:
            quantum_instance = self._qasm_quantum_instance

        qc = QuantumCircuit(1)

        # construct simple feature map
        param_x = Parameter("x")
        qc.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter("y")
        qc.ry(param_y, 0)

        qnn = CircuitQNN(
            qc,
            [param_x],
            [param_y],
            sparse=sparse,
            sampling=False,
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)

        test_data = [
            Tensor([1]),
            Tensor([1, 2]),
            Tensor([[1], [2]]),
            Tensor([[[1], [2]], [[3], [4]]]),
        ]

        # test model
        self._validate_output_shape(model, test_data)
        if q_i == "sv":
            self._validate_backward_pass(model)

    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, "sv"),
        (None, None, True, "sv"),
        (lambda x: np.sum(x) % 2, 2, False, "sv"),
        (lambda x: np.sum(x) % 2, 2, True, "sv"),
        (None, None, False, "qasm"),
        (None, None, True, "qasm"),
        (lambda x: np.sum(x) % 2, 2, False, "qasm"),
        (lambda x: np.sum(x) % 2, 2, True, "qasm"),
    )
    def test_circuit_qnn_1_8(self, config):
        """Torch Connector + Circuit QNN with no interpret, dense output,
        and input/output shape 1/8 ."""
        from torch import Tensor

        interpret, output_shape, sparse, q_i = config
        if sparse and not _optionals.HAS_SPARSE:
            self.skipTest("sparse library is required to run this test")
            return
        if q_i == "sv":
            quantum_instance = self._sv_quantum_instance
        else:
            quantum_instance = self._qasm_quantum_instance

        qc = QuantumCircuit(3)

        # construct simple feature map
        param_x = Parameter("x")
        qc.ry(param_x, range(3))

        # construct simple feature map
        param_y = Parameter("y")
        qc.ry(param_y, range(3))

        qnn = CircuitQNN(
            qc,
            [param_x],
            [param_y],
            sparse=sparse,
            sampling=False,
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)

        test_data = [
            Tensor([1]),
            Tensor([1, 2]),
            Tensor([[1], [2]]),
            Tensor([[[1], [2]], [[3], [4]]]),
        ]

        # test model
        self._validate_output_shape(model, test_data)
        if q_i == "sv":
            self._validate_backward_pass(model)

    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, "sv"),
        (None, None, True, "sv"),
        (lambda x: np.sum(x) % 2, 2, False, "sv"),
        (lambda x: np.sum(x) % 2, 2, True, "sv"),
        (None, None, False, "qasm"),
        (None, None, True, "qasm"),
        (lambda x: np.sum(x) % 2, 2, False, "qasm"),
        (lambda x: np.sum(x) % 2, 2, True, "qasm"),
    )
    def test_circuit_qnn_2_4(self, config):
        """Torch Connector + Circuit QNN with no interpret, dense output,
        and input/output shape 1/8 ."""
        from torch import Tensor

        interpret, output_shape, sparse, q_i = config
        if sparse and not _optionals.HAS_SPARSE:
            self.skipTest("sparse library is required to run this test")
            return
        if q_i == "sv":
            quantum_instance = self._sv_quantum_instance
        else:
            quantum_instance = self._qasm_quantum_instance

        qc = QuantumCircuit(2)

        # construct simple feature map
        param_x_1, param_x_2 = Parameter("x1"), Parameter("x2")
        qc.ry(param_x_1, range(2))
        qc.ry(param_x_2, range(2))

        # construct simple feature map
        param_y = Parameter("y")
        qc.ry(param_y, range(2))

        qnn = CircuitQNN(
            qc,
            [param_x_1, param_x_2],
            [param_y],
            sparse=sparse,
            sampling=False,
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)

        test_data = [
            Tensor([1]),
            Tensor([1, 2]),
            Tensor([[1], [2]]),
            Tensor([[1, 2], [3, 4]]),
            Tensor([[[1], [2]], [[3], [4]]]),
        ]

        # test model
        self._validate_output_shape(model, test_data)
        if q_i == "sv":
            self._validate_backward_pass(model)

    def test_circuit_qnn_without_parameters(self):
        """Tests CircuitQNN without parameters."""
        quantum_instance = self._sv_quantum_instance
        qc = QuantumCircuit(2)
        param_y = Parameter("y")
        qc.ry(param_y, range(2))

        qnn = CircuitQNN(
            circuit=qc,
            input_params=[param_y],
            sparse=False,
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)
        self._validate_backward_pass(model)

        qnn = CircuitQNN(
            circuit=qc,
            weight_params=[param_y],
            sparse=False,
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)
        self._validate_backward_pass(model)

    @data(
        # interpret
        (None),
        (lambda x: np.sum(x) % 2),
    )
    def test_circuit_qnn_sampling(self, interpret):
        """Test Torch Connector + Circuit QNN for sampling."""
        from torch import Tensor

        qc = QuantumCircuit(2)

        # construct simple feature map
        param_x1, param_x2 = Parameter("x1"), Parameter("x2")
        qc.ry(param_x1, range(2))
        qc.ry(param_x2, range(2))

        # construct simple feature map
        param_y = Parameter("y")
        qc.ry(param_y, range(2))

        qnn = CircuitQNN(
            qc,
            [param_x1, param_x2],
            [param_y],
            sparse=False,
            sampling=True,
            interpret=interpret,
            output_shape=None,
            quantum_instance=self._qasm_quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)
        model.to(self._device)

        test_data = [Tensor([2, 2]), Tensor([[1, 1], [2, 2]])]
        for i, x in enumerate(test_data):
            x = x.to(self._device)
            if i == 0:
                self.assertEqual(model(x).shape, qnn.output_shape)
            else:
                shape = model(x).shape
                self.assertEqual(shape, (len(x), *qnn.output_shape))

    def test_opflow_qnn_batch_gradients(self):
        """Test backward pass for batch input."""
        import torch

        # construct random data set
        num_inputs = 2
        num_samples = 10
        x = np.random.rand(num_samples, num_inputs)

        # set up QNN
        qnn = TwoLayerQNN(
            num_qubits=num_inputs,
            quantum_instance=self._sv_quantum_instance,
        )

        # set up PyTorch module
        initial_weights = np.random.rand(qnn.num_weights)
        model = TorchConnector(qnn, initial_weights=initial_weights)
        model.to(self._device)

        # test single gradient
        w = model.weight.detach().cpu().numpy()
        res_qnn = qnn.forward(x[0, :], w)

        # construct finite difference gradient for weight
        eps = 1e-4
        grad = np.zeros(w.shape)
        for k in range(len(w)):
            delta = np.zeros(w.shape)
            delta[k] += eps

            f_1 = qnn.forward(x[0, :], w + delta)
            f_2 = qnn.forward(x[0, :], w - delta)

            grad[k] = (f_1 - f_2) / (2 * eps)

        grad_qnn = qnn.backward(x[0, :], w)[1][0, 0, :]
        self.assertAlmostEqual(np.linalg.norm(grad - grad_qnn), 0.0, places=4)

        model.zero_grad()
        res_model = model(torch.tensor(x[0, :], device=self._device))
        self.assertAlmostEqual(
            np.linalg.norm(res_model.detach().cpu().numpy() - res_qnn[0]), 0.0, places=4
        )
        res_model.backward()
        grad_model = model.weight.grad
        self.assertAlmostEqual(
            np.linalg.norm(grad_model.detach().cpu().numpy() - grad_qnn), 0.0, places=4
        )

        # test batch input
        batch_grad = np.zeros((*w.shape, num_samples, 1))
        for k in range(len(w)):
            delta = np.zeros(w.shape)
            delta[k] += eps

            f_1 = qnn.forward(x, w + delta)
            f_2 = qnn.forward(x, w - delta)

            batch_grad[k] = (f_1 - f_2) / (2 * eps)

        batch_grad = np.sum(batch_grad, axis=1)
        batch_grad_qnn = np.sum(qnn.backward(x, w)[1], axis=0)
        self.assertAlmostEqual(
            np.linalg.norm(batch_grad - batch_grad_qnn.transpose()), 0.0, places=4
        )

        model.zero_grad()
        batch_res_model = sum(model(torch.tensor(x, device=self._device)))
        batch_res_model.backward()
        self.assertAlmostEqual(
            np.linalg.norm(model.weight.grad.detach().cpu().numpy() - batch_grad.transpose()[0]),
            0.0,
            places=4,
        )

    @data(
        # output_shape, interpret
        (4, None),
        (2, lambda x: f"{x:b}".count("1") % 2),
    )
    def test_circuit_qnn_batch_gradients(self, config):
        """Test batch gradient computation of CircuitQNN gives the same result as the sum of
        individual gradients."""
        import torch
        from torch.nn import MSELoss
        from torch.optim import SGD

        output_shape, interpret = config
        num_inputs = 2

        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        qnn = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=self._sv_quantum_instance,
        )

        # set up PyTorch module
        initial_weights = np.array([0.1] * qnn.num_weights)
        model = TorchConnector(qnn, initial_weights)
        model.to(self._device)

        # random data set
        x = torch.rand(5, 2)
        y = torch.rand(5, output_shape)

        # define optimizer and loss
        optimizer = SGD(model.parameters(), lr=0.1)
        f_loss = MSELoss(reduction="sum")

        sum_of_individual_losses = 0.0
        for x_i, y_i in zip(x, y):
            x_i = x_i.to(self._device)
            y_i = y_i.to(self._device)
            output = model(x_i)
            sum_of_individual_losses += f_loss(output, y_i)
        optimizer.zero_grad()
        sum_of_individual_losses.backward()
        sum_of_individual_gradients = model.weight.grad.detach().cpu()

        x = x.to(self._device)
        y = y.to(self._device)
        output = model(x)
        batch_loss = f_loss(output, y)
        optimizer.zero_grad()
        batch_loss.backward()
        batch_gradients = model.weight.grad.detach().cpu()

        self.assertAlmostEqual(
            np.linalg.norm(sum_of_individual_gradients - batch_gradients), 0.0, places=4
        )

        self.assertAlmostEqual(
            sum_of_individual_losses.detach().cpu().numpy(),
            batch_loss.detach().cpu().numpy(),
            places=4,
        )
