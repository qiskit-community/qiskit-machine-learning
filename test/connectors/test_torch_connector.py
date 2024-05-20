# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
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
from typing import cast, Union, List, Tuple, Any

from test.connectors.test_torch import TestTorch

import numpy as np
from ddt import ddt, data, unpack, idata
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors.torch_connector import _get_einsum_signature


@ddt
class TestTorchConnector(TestTorch):
    """Torch Connector Tests."""

    def setup_test(self):
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

        fmap = ZFeatureMap(num_qubits, reps=1)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        qc = QuantumCircuit(num_qubits)
        qc.compose(fmap, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = SamplerQNN(
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
        fmap = ZFeatureMap(num_qubits, reps=1)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        qc = QuantumCircuit(num_qubits)
        qc.compose(fmap, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
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
            ) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
                """
                Build the quantum circuit for the convolutional layer.

                Args:
                    num_weights (int): Number of weight parameters.
                    num_input (int): Number of input parameters.
                    num_qubits (int, optional): Number of qubits for representing parameters.
                        Defaults to 3.

                Returns:
                    Tuple[QuantumCircuit, List[Parameter], List[Parameter]]: Quantum circuit,
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
                    circuit=qc,
                    weight_params=weight_params,
                    interpret=self.interpret,  # type: ignore
                    input_params=input_params,
                    output_shape=self.output_channel,
                )

            def interpret(self, output: Union[float, int]) -> Any:
                """
                Interprets the output from the quantum circuit.

                Args:
                    output (Union[float, int]): Output from the quantum circuit.

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

    def test_get_einsum_signature(self):
        """
        Tests the functionality of `_get_einsum_signature` function.

        This function is tested with valid inputs (e.g., `n_dimensions` = 3) and expected outputs.
        It also covers error handling scenarios for invalid input arguments.

        Error Handling Tests:
        - Raises a RuntimeError when the character limit for signature generation is exceeded.

        Additional Test Scenario:
        - Tests `_get_einsum_signature` in the context of a convolutional layer based on Sampler QNN,
          using a 4-D input tensor (batch, channel, height, width) to ensure compatibility.
          This test mirrors Issue https://github.com/qiskit-community/qiskit-machine-learning/issues/716

        Note: Backward test is run implicitly within the context of convolutional layer execution.

        pylint disable=SPELLING
        """
        # Test raises for exceeding character limit
        with self.assertRaises(RuntimeError):
            _get_einsum_signature(30)

        # Test with a convolutional layer based on Sampler QNN and 4-D input
        # Refer to issue https://github.com/qiskit-community/qiskit-machine-learning/issues/716
        import torch

        model = self._create_convolutional_layer(3, 1, 3, 3)
        input_tensor = torch.rand((2, 3, 6, 6))
        input_tensor.requires_grad = True
        output_tensor = model.forward(input_tensor)
        output_tensor = torch.sum(output_tensor)

        # Run `backward`, which calls `_get_einsum_signature`
        output_tensor.backward()
