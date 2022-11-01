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

"""Abstract class to test PyTorch hybrid networks."""

from typing import Optional, Union, cast
from test.connectors.test_torch import TestTorch
import numpy as np
from ddt import ddt, idata

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import (
    CircuitQNN,
    TwoLayerQNN,
    NeuralNetwork,
    EstimatorQNN,
    SamplerQNN,
)
from qiskit_machine_learning.connectors import TorchConnector


@ddt
class TestTorchNetworks(TestTorch):
    """Base class for hybrid PyTorch network tests."""

    def _create_network(self, qnn: NeuralNetwork, output_size: int):
        from torch import cat
        from torch.nn import Linear, Module
        import torch.nn.functional as F

        # set up dummy hybrid PyTorch module
        class Net(Module):
            """PyTorch nn module."""

            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 2)
                self.qnn = TorchConnector(
                    qnn, np.array([0.1] * qnn.num_weights)
                )  # Apply torch connector
                self.fc2 = Linear(output_size, 1)  # shape depends on the type of the QNN

            def forward(self, x):
                """Forward pass."""
                x = F.relu(self.fc1(x))
                x = self.qnn(x)  # apply QNN
                x = self.fc2(x)
                return cat((x, 1 - x), -1)

        return Net()

    def _create_circuit_qnn(self) -> CircuitQNN:
        output_shape, interpret = 2, lambda x: f"{x:b}".count("1") % 2
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
            input_gradients=True,  # for hybrid qnn
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=self._sv_quantum_instance,
        )
        return qnn

    def _create_opflow_qnn(self) -> TwoLayerQNN:
        num_inputs = 2

        # set up QNN
        qnn = TwoLayerQNN(
            num_qubits=num_inputs,
            quantum_instance=self._sv_quantum_instance,
            input_gradients=True,  # for hybrid qnn
        )
        return qnn

    def _create_estimator_qnn(self) -> EstimatorQNN:
        num_inputs = 2

        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,  # for hybrid qnn
        )
        return qnn

    def _create_sampler_qnn(self) -> SamplerQNN:
        output_shape, interpret = 2, lambda x: f"{x:b}".count("1") % 2
        num_inputs = 2

        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,  # for hybrid qnn
            interpret=interpret,
            output_shape=output_shape,
        )
        return qnn

    @idata(["opflow", "circuit_qnn", "sampler_qnn", "estimator_qnn"])
    def test_hybrid_batch_gradients(self, qnn_type: str):
        """Test gradient back-prop for batch input in a qnn."""
        import torch
        from torch.nn import MSELoss
        from torch.optim import SGD

        qnn: Optional[Union[CircuitQNN, TwoLayerQNN, SamplerQNN, EstimatorQNN]] = None
        if qnn_type == "opflow":
            qnn = self._create_opflow_qnn()
            output_size = 1
        elif qnn_type == "circuit_qnn":
            qnn = self._create_circuit_qnn()
            output_size = 2
        elif qnn_type == "sampler_qnn":
            qnn = self._create_sampler_qnn()
            output_size = 2
        elif qnn_type == "estimator_qnn":
            qnn = self._create_estimator_qnn()
            output_size = 1
        else:
            raise ValueError("Unsupported QNN type")

        model = self._create_network(qnn, output_size=output_size)
        model.to(self._device)

        # random data set
        x = torch.rand((5, 4), device=self._device)
        y = torch.rand((5, 2), device=self._device)

        # define optimizer and loss
        optimizer = SGD(model.parameters(), lr=0.1)
        f_loss = MSELoss(reduction="sum")

        # loss and gradients without batch
        optimizer.zero_grad(set_to_none=True)
        sum_of_individual_losses = 0.0
        for x_i, y_i in zip(x, y):
            output = model(x_i)
            sum_of_individual_losses += f_loss(output, y_i)
        cast(torch.Tensor, sum_of_individual_losses).backward()
        sum_of_individual_gradients = 0.0
        for n, param in model.named_parameters():
            # make sure gradient is not None
            self.assertFalse(param.grad is None)
            if n.endswith(".weight"):
                sum_of_individual_gradients += np.sum(param.grad.detach().cpu().numpy())

        # loss and gradients with batch
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        batch_loss = f_loss(output, y)
        batch_loss.backward()
        batch_gradients = 0.0
        for n, param in model.named_parameters():
            # make sure gradient is not None
            self.assertFalse(param.grad is None)
            if n.endswith(".weight"):
                batch_gradients += np.sum(param.grad.detach().cpu().numpy())

        # making sure they are equivalent
        self.assertAlmostEqual(
            cast(float, np.linalg.norm(sum_of_individual_gradients - batch_gradients)),
            0.0,
            places=4,
        )

        self.assertAlmostEqual(
            cast(torch.Tensor, sum_of_individual_losses).detach().cpu().numpy(),
            batch_loss.detach().cpu().numpy(),
            places=4,
        )
