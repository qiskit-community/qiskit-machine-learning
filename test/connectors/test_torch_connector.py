# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2024, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Torch Connector."""

import itertools
from typing import cast, Any

from test.connectors.test_torch import TestTorch

import numpy as np
from ddt import ddt, data, unpack, idata
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.connectors.torch_connector import _TorchNNFunction
from qiskit_machine_learning.primitives import QMLSampler as Sampler, QMLEstimator as Estimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


@ddt
class TestTorchConnector(TestTorch):
    """Torch Connector Tests."""

    def setup_test(self):

        algorithm_globals.random_seed = 123

        super().setup_test()
        import torch

        # pylint: disable=attribute-defined-outside-init
        self._test_data = [
            torch.tensor([1.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        ]

    def _validate_backward_automatically(self, model: TorchConnector) -> None:
        """Uses PyTorch to validate the backward pass / autograd.

        Args:
            model: The model to be tested.
        """
        import torch

        # test autograd
        func = _TorchNNFunction.apply  # (input, weights, qnn)
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

    def _validate_forward(self, model: TorchConnector):
        import torch

        for batch_size in [1, 2]:
            input_data = torch.rand((batch_size, model.neural_network.num_inputs))

            connector_output = model(input_data).detach()
            qnn_output = model.neural_network.forward(
                input_data.detach().numpy(), model.weight.detach().numpy()
            )

            self.assertEqual(connector_output.is_sparse, model.sparse)

            if model.sparse:
                connector_output = connector_output.to_dense()
            if model.neural_network.sparse:
                import sparse

                self.assertTrue(isinstance(qnn_output, sparse.SparseArray))
                # mypy
                qnn_output = cast(sparse.SparseArray, qnn_output)
                qnn_output = qnn_output.todense()

            np.testing.assert_almost_equal(connector_output.numpy(), qnn_output)

        # test with a wrong input size
        wrong_input = torch.rand((1, model.neural_network.num_inputs + 1))
        with self.assertRaises(QiskitMachineLearningError):
            model(wrong_input)

    def _validate_backward(self, model: TorchConnector):
        import torch

        for batch_size in [1, 2]:
            input_data = torch.rand((batch_size, model.neural_network.num_inputs))

            model.zero_grad()

            connector_fwd_out = model(input_data)
            if model.sparse:
                connector_fwd_out = connector_fwd_out.to_dense()

            # we need a scalar function to trigger gradients
            torch.sum(connector_fwd_out).backward()

            connector_backward = model.weight.grad
            self.assertEqual(connector_backward.is_sparse, model.sparse)
            if model.sparse:
                self.assertTrue(connector_backward.is_sparse)
                connector_backward = connector_backward.to_dense()

            _, qnn_backward = model.neural_network.backward(
                input_data.detach().numpy(), model.weight.detach().numpy()
            )
            if model.neural_network.sparse:
                import sparse

                self.assertTrue(isinstance(qnn_backward, sparse.SparseArray))
                # mypy
                qnn_backward = cast(sparse.SparseArray, qnn_backward)
                qnn_backward = qnn_backward.todense()

            fin_diff_grad = self._evaluate_fin_diff_gradient(model, input_data)
            np.testing.assert_almost_equal(fin_diff_grad, qnn_backward, decimal=4)

            # sum up across batch size and across all parameters
            qnn_backward = np.sum(qnn_backward, axis=(0, 1))

            np.testing.assert_almost_equal(connector_backward.numpy(), qnn_backward)

    def _evaluate_fin_diff_gradient(self, model: TorchConnector, input_data):
        qnn = model.neural_network
        weights = model.weight.detach().numpy()

        eps = 1e-4
        grad = np.zeros((len(input_data), *qnn.output_shape, qnn.num_weights))
        for k in range(qnn.num_weights):
            delta = np.zeros_like(weights)
            delta[k] += eps

            f_1 = qnn.forward(input_data, weights + delta)
            f_2 = qnn.forward(input_data, weights - delta)
            if qnn.sparse:
                import sparse

                # mypy
                f_1 = cast(sparse.SparseArray, f_1)
                f_2 = cast(sparse.SparseArray, f_2)

                f_1 = f_1.todense()
                f_2 = f_2.todense()

            grad[:, :, k] = (f_1 - f_2) / (2 * eps)
        return grad

    @idata(
        # num qubits, sparse_connector, sparse_qnn, interpret
        itertools.product(
            [1, 2], [True, False], [True, False], [lambda x: f"{x:b}".count("1") % 2, None]
        )
    )
    @unpack
    def test_sampler_qnn(self, num_qubits, sparse_connector, sparse_qnn, interpret):
        """Test TorchConnector on SamplerQNN."""
        if interpret is not None:
            output_shape = 2
        else:
            output_shape = None

        fmap = z_feature_map(num_qubits, reps=1)
        ansatz = real_amplitudes(num_qubits, reps=1)
        qc = QuantumCircuit(num_qubits)
        qc.compose(fmap, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = SamplerQNN(
            sampler=Sampler(),
            circuit=qc,
            input_params=fmap.parameters,
            weight_params=ansatz.parameters,
            sparse=sparse_qnn,
            interpret=interpret,
            output_shape=output_shape,
            input_gradients=True,
        )

        try:
            model = TorchConnector(qnn, sparse=sparse_connector)
        except QiskitMachineLearningError as qml_ex:
            if sparse_connector and not sparse_qnn:
                self.skipTest("Skipping test when connector is sparse and qnn is not sparse")
            else:
                raise QiskitMachineLearningError(
                    "Unexpected exception during initialization"
                ) from qml_ex

        self._validate_forward(model)
        self._validate_backward(model)
        self._validate_backward_automatically(model)

    @data(
        (1, None),
        (1, SparsePauliOp.from_list([("Z", 1)])),
        (2, None),
        (2, SparsePauliOp.from_list([("ZZ", 1)])),
        (2, [SparsePauliOp.from_list([("ZI", 1)]), SparsePauliOp.from_list([("IZ", 1)])]),
    )
    @unpack
    def test_estimator_qnn(self, num_qubits, observables):
        """Test TorchConnector on EstimatorQNN."""
        fmap = z_feature_map(num_qubits, reps=1)
        ansatz = real_amplitudes(num_qubits, reps=1)
        qc = QuantumCircuit(num_qubits)
        qc.compose(fmap, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            estimator=Estimator(default_precision=0.01, seed=123),
            circuit=qc,
            observables=observables,
            input_params=fmap.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )

        model = TorchConnector(qnn)

        self._validate_forward(model)
        self._validate_backward(model)
        self._validate_backward_automatically(model)

    def _create_convolutional_layer(
        self, input_channel: int, output_channel: int, num_qubits: int, num_weight: int
    ):
        import torch
        from qiskit.circuit import Parameter

        class ConvolutionalLayer(torch.nn.Module):
            """
            Quantum Convolutional Neural Network layer implemented using Qiskit and PyTorch.

            Args:
                input_channel (int): Number of input channels.
                output_channel (int): Number of output channels.
                num_qubits (int): Number of qubits to represent weights.
                num_weight (int): Number of weight parameters in the quantum circuit.
                kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
                stride (int, optional): Stride of the convolution. Defaults to 1.
            """

            # pylint: disable=too-many-positional-arguments
            def __init__(
                self,
                input_channel: int,
                output_channel: int,
                num_qubits: int,
                num_weight: int,
                kernel_size: int = 3,
                stride: int = 1,
            ):
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.input_channel = input_channel
                self.output_channel = output_channel
                self.num_weight = num_weight
                self.num_input = kernel_size * kernel_size * input_channel
                self.num_qubits = num_qubits
                self.qnn = TorchConnector(self.sampler())
                if 2**num_qubits < output_channel:
                    raise ValueError(
                        (
                            f"The output channel must be >= 2**num_qubits. "
                            f"Got output_channel {output_channel:d} < 2 ** {num_qubits:d}"
                        )
                    )

            @staticmethod
            def build_circuit(
                num_weights: int, num_input: int, num_qubits: int = 3
            ) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
                """
                Build the quantum circuit for the convolutional layer.

                Args:
                    num_weights (int): Number of weight parameters.
                    num_input (int): Number of input parameters.
                    num_qubits (int, optional): Number of qubits for representing parameters.
                        Defaults to 3.

                Returns:
                    tuple[QuantumCircuit, list[Parameter], list[Parameter]]: Quantum circuit,
                        list of weight parameters, list of input parameters.
                """
                qc = QuantumCircuit(num_qubits)
                weight_params = [Parameter(f"w{i}") for i in range(num_weights)]
                input_params = [Parameter(f"x{i}") for i in range(num_input)]

                # Construct the quantum circuit with the parameters
                for i in range(num_qubits):
                    qc.h(i)
                for i in range(num_input):
                    qc.ry(input_params[i] * 2 * torch.pi, i % num_qubits)
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)
                for i in range(num_weights):
                    qc.rx(weight_params[i] * 2 * torch.pi, i % num_qubits)
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)

                return qc, weight_params, input_params

            def sampler(self) -> SamplerQNN:
                """
                Creates a SamplerQNN object representing the quantum neural network.

                Returns:
                    SamplerQNN: Quantum neural network.
                """
                qc, weight_params, input_params = self.build_circuit(
                    self.num_weight, self.num_input, 3
                )

                # Use SamplerQNN to convert the quantum circuit to a PyTorch module
                return SamplerQNN(
                    sampler=Sampler(),
                    circuit=qc,
                    weight_params=weight_params,
                    interpret=self.interpret,  # type: ignore
                    input_params=input_params,
                    output_shape=self.output_channel,
                )

            def interpret(self, output: float | int) -> Any:
                """
                Interprets the output from the quantum circuit.

                Args:
                    output (float | int): Output from the quantum circuit.

                Returns:
                    Any: Remainder of the output divided by the
                    number of output channels.
                """
                return output % self.output_channel

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                """
                Perform forward pass through the quantum convolutional layer.

                Args:
                    input_tensor (torch.Tensor): Input tensor.

                Returns:
                    torch.Tensor: Output tensor after convolution.
                """
                height = len(range(0, input_tensor.shape[-2] - self.kernel_size + 1, self.stride))
                width = len(range(0, input_tensor.shape[-1] - self.kernel_size + 1, self.stride))
                output = torch.zeros((input_tensor.shape[0], self.output_channel, height, width))
                input_tensor = torch.nn.functional.unfold(
                    input_tensor[...], kernel_size=self.kernel_size, stride=self.stride
                )
                qnn_output = self.qnn(input_tensor.permute(2, 0, 1)).permute(1, 2, 0)
                qnn_output = torch.reshape(
                    qnn_output, shape=(input_tensor.shape[0], self.output_channel, height, width)
                )
                output += qnn_output
                return output

        return ConvolutionalLayer(input_channel, output_channel, num_qubits, num_weight, stride=1)

    @data(
        ((1, 2, 1), [0.3, 1.1]),
        ((1, 3, 1), [0.2, 0.7, 1.4]),
    )
    @unpack
    def test_batched_input_gradients_analytic_estimator_qnn(self, input_shape, x_vals):
        """Tests analytic input and weight gradient on EstimatorQNN with non-uniform loss."""
        import torch
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(1)
        x_param = Parameter("x")
        w_param = Parameter("w")
        qc.ry(x_param, 0)
        qc.rz(w_param, 0)

        observables = [SparsePauliOp.from_list([("Z", 1.0)]), SparsePauliOp.from_list([("X", 1.0)])]
        estimator = Estimator(default_precision=0.0)
        qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            observables=observables,
            input_params=[x_param],
            weight_params=[w_param],
            input_gradients=True,
        )

        w_val = 0.4
        model = TorchConnector(qnn, initial_weights=[w_val])

        inputs = torch.tensor(
            np.array(x_vals).reshape(input_shape), dtype=torch.float, requires_grad=True
        )
        outputs = model(inputs)
        c0, c1 = 1.7, -0.6
        loss = c0 * outputs[..., 0].sum() + c1 * outputs[..., 1].sum()
        loss.backward()

        # d(output0)/dx = -sin(x), d(output1)/dx = cos(x)*cos(w)
        expected_input_grad = np.array(
            [-c0 * np.sin(x) + c1 * np.cos(x) * np.cos(w_val) for x in x_vals]
        ).reshape(input_shape)
        np.testing.assert_allclose(
            inputs.grad.detach().cpu().numpy(), expected_input_grad, atol=1e-5
        )

        # d(output0)/dw = 0, d(output1)/dw = -sin(x)*sin(w)
        expected_weight_grad = np.array([sum(-c1 * np.sin(x) * np.sin(w_val) for x in x_vals)])
        np.testing.assert_allclose(
            model.weight.grad.detach().cpu().numpy(), expected_weight_grad, atol=1e-5
        )

    @data(
        ((1,), [0.3], False),
        ((1,), [0.3], True),
        ((1, 2, 1), [0.3, 0.9], False),
        ((1, 2, 1), [0.3, 0.9], True),
    )
    @unpack
    def test_multidimensional_sampler_qnn_output_gradients(self, input_shape, x_vals, sparse):
        """Tests analytic input and weight gradients on SamplerQNN with 2x2 output."""
        import torch
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(2)
        x = Parameter("x")
        w = Parameter("w")
        qc.ry(x, 0)
        qc.ry(w, 1)

        def interpret_grid(val: int) -> tuple[int, int]:
            return (val // 2, val % 2)

        sampler = Sampler()  # exact deterministic mode
        qnn = SamplerQNN(
            circuit=qc,
            sampler=sampler,
            input_params=[x],
            weight_params=[w],
            sparse=sparse,
            interpret=interpret_grid,
            output_shape=(2, 2),
            input_gradients=True,
        )

        w_val = 0.5
        model = TorchConnector(qnn, initial_weights=[w_val], sparse=sparse)

        inputs = torch.tensor(
            np.array(x_vals).reshape(input_shape), dtype=torch.float, requires_grad=True
        )
        outputs = model(inputs)
        if sparse:
            outputs = outputs.to_dense()

        weights_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
        loss = (outputs * weights_matrix).sum()
        loss.backward()

        # With little-endian qubit order, loss = 1 + sin²(x/2) + 2 sin²(w/2).
        expected_input_grad = (0.5 * np.sin(x_vals)).reshape(input_shape)
        input_grad = inputs.grad
        if sparse:
            input_grad = input_grad.to_dense()
        np.testing.assert_allclose(
            input_grad.detach().cpu().numpy(), expected_input_grad, atol=1e-5
        )

        expected_weight_grad = np.array([len(x_vals) * np.sin(w_val)])
        weight_grad = model.weight.grad
        if sparse:
            weight_grad = weight_grad.to_dense()
        np.testing.assert_allclose(
            weight_grad.detach().cpu().numpy(), expected_weight_grad, atol=1e-5
        )
