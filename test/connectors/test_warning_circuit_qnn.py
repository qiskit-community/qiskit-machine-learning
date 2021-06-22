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
import warnings
from typing import List

from test import QiskitMachineLearningTestCase, requires_extra_library

from copy import deepcopy
import numpy as np

from ddt import ddt, data

try:
    from torch import Tensor
    from torch.nn import MSELoss
    from torch.optim import SGD
except ImportError:

    class Tensor:  # type: ignore
        """Empty Tensor class
        Replacement if torch.Tensor is not present.
        """

        pass


from qiskit import QuantumCircuit, Aer
from qiskit.providers.aer import AerSimulator
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import StateFn, ListOp, PauliSumOp

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN, OpflowQNN
from qiskit_machine_learning.connectors import TorchConnector


@ddt
class TestTorchConnector(QiskitMachineLearningTestCase):
    """Torch Connector Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345
        # specify quantum instances
        self.sv_quantum_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        self.qasm_quantum_instance = QuantumInstance(
            AerSimulator(),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        try:
            import torch

            torch.manual_seed(algorithm_globals.random_seed)
        except ImportError:
            pass

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
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            ) from ex

        # create benchmark model
        in_dim = model.neural_network.num_inputs
        if len(model.neural_network.output_shape) != 1:
            raise QiskitMachineLearningError("Function only works for one dimensional output")
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
            with self.subTest("c_worked == q_worked", tensor=x):
                self.assertEqual(c_worked, q_worked)
            if c_worked and q_worked:
                with self.subTest("c_shape == q_shape", tensor=x):
                    self.assertEqual(c_shape, q_shape)

    def validate_backward_pass(self, model: TorchConnector) -> None:
        """Uses PyTorch to validate the backward pass / autograd.

        Args:
            model: The model to be tested.

        Raises:
            MissingOptionalLibraryError: torch not installed
        """
        try:
            import torch
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            ) from ex

        # test autograd
        func = TorchConnector._TorchNNFunction.apply  # (input, weights, qnn)
        input_data = (
            torch.randn(model.neural_network.num_inputs, dtype=torch.double, requires_grad=True),
            torch.randn(model.neural_network.num_weights, dtype=torch.double, requires_grad=True),
            model.neural_network,
            False,
        )
        test = torch.autograd.gradcheck(func, input_data, eps=1e-4, atol=1e-3)  # type: ignore
        self.assertTrue(test)


    @data(
        # interpret, output_shape, sparse, quantum_instance
        (None, None, False, "sv"),
        (None, None, True, "sv"),
        (None, 1, False, "sv"),
        (None, 1, True, "sv"),
        (lambda x: np.sum(x) % 2, 2, False, "sv"),
        (lambda x: np.sum(x) % 2, 2, True, "sv"),

        (None, None, False, "qasm"),
        (None, None, True, "qasm"),
        (None, 1, False, "quasm"),
        (None, 1, True, "quasm"),
        (lambda x: np.sum(x) % 2, 2, False, "qasm"),
        (lambda x: np.sum(x) % 2, 2, True, "qasm")
    )
    @requires_extra_library
    def test_circuit_qnn_1_1(self, config):
        """Torch Connector + Circuit QNN with no sampling and input/output shape 1/1 ."""

        interpret, output_shape, sparse, q_i = config
        if q_i == "sv":
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        qc = QuantumCircuit(1)

        # construct simple feature map
        param_x = Parameter("x")
        qc.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter("y")
        qc.ry(param_y, 0)

        # check warning when output_shape defined without interpret
        if interpret is None and output_shape is not None:
            with self.assertLogs(level="WARNING") as cm:
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
                self.assertEqual(cm.output, ["WARNING:qiskit_machine_learning.neural_networks.circuit_qnn:No interpret "
                                             "function given, custom output_shape will be overridden by default "
                                             "output_shape: 2^num_qubits. "])
        else:
            with self.assertLogs(logger, logging.WARN) as cm:
                # We want to assert there are no warnings, but the 'assertLogs' method does not support that.
                # Therefore, we are adding a dummy warning, and then we will assert it is the only warning.
                logger.warn("Dummy warning")
                # DO STUFF

            self.assertEqual(
                ["Dummy warning"],
                cm.output,
            )
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

    @data(
        # interpret
        (None),
        (lambda x: np.sum(x) % 2),
    )
    @requires_extra_library
    def test_circuit_qnn_sampling(self, interpret):
        """Test Torch Connector + Circuit QNN for sampling."""

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
            quantum_instance=self.qasm_quantum_instance,
            input_gradients=True,
        )
        model = TorchConnector(qnn)

        test_data = [Tensor([2, 2]), Tensor([[1, 1], [2, 2]])]
        for i, x in enumerate(test_data):
            if i == 0:
                self.assertEqual(model(x).shape, qnn.output_shape)
            else:
                shape = model(x).shape
                self.assertEqual(shape, (len(x), *qnn.output_shape))

    # @data(
    #     # interpret, output_shape, sparse, quantum_instance
    #     (None, None, False, "sv"),
    #     (None, None, True, "sv"),
    #     (lambda x: np.sum(x) % 2, 2, False, "sv"),
    #     (lambda x: np.sum(x) % 2, 2, True, "sv"),
    #     (None, None, False, "qasm"),
    #     (None, None, True, "qasm"),
    #     (lambda x: np.sum(x) % 2, 2, False, "qasm"),
    #     (lambda x: np.sum(x) % 2, 2, True, "qasm"),
    # )
    # @requires_extra_library
    # def test_circuit_qnn_1_8(self, config):
    #     """Torch Connector + Circuit QNN with no interpret, dense output,
    #     and input/output shape 1/8 ."""
    #
    #     interpret, output_shape, sparse, q_i = config
    #     if q_i == "sv":
    #         quantum_instance = self.sv_quantum_instance
    #     else:
    #         quantum_instance = self.qasm_quantum_instance
    #
    #     qc = QuantumCircuit(3)
    #
    #     # construct simple feature map
    #     param_x = Parameter("x")
    #     qc.ry(param_x, range(3))
    #
    #     # construct simple feature map
    #     param_y = Parameter("y")
    #     qc.ry(param_y, range(3))
    #
    #     qnn = CircuitQNN(
    #         qc,
    #         [param_x],
    #         [param_y],
    #         sparse=sparse,
    #         sampling=False,
    #         interpret=interpret,
    #         output_shape=output_shape,
    #         quantum_instance=quantum_instance,
    #         input_gradients=True,
    #     )
    #     model = TorchConnector(qnn)
    #
    #     test_data = [
    #         Tensor([1]),
    #         Tensor([1, 2]),
    #         Tensor([[1], [2]]),
    #         Tensor([[[1], [2]], [[3], [4]]]),
    #     ]
    #
    #     # test model
    #     self.validate_output_shape(model, test_data)
    #     if q_i == "sv":
    #         self.validate_backward_pass(model)
    #
    # @data(
    #     # interpret, output_shape, sparse, quantum_instance
    #     (None, None, False, "sv"),
    #     (None, None, True, "sv"),
    #     (lambda x: np.sum(x) % 2, 2, False, "sv"),
    #     (lambda x: np.sum(x) % 2, 2, True, "sv"),
    #     (None, None, False, "qasm"),
    #     (None, None, True, "qasm"),
    #     (lambda x: np.sum(x) % 2, 2, False, "qasm"),
    #     (lambda x: np.sum(x) % 2, 2, True, "qasm"),
    # )
    # @requires_extra_library
    # def test_circuit_qnn_2_4(self, config):
    #     """Torch Connector + Circuit QNN with no interpret, dense output,
    #     and input/output shape 1/8 ."""
    #
    #     interpret, output_shape, sparse, q_i = config
    #     if q_i == "sv":
    #         quantum_instance = self.sv_quantum_instance
    #     else:
    #         quantum_instance = self.qasm_quantum_instance
    #
    #     qc = QuantumCircuit(2)
    #
    #     # construct simple feature map
    #     param_x_1, param_x_2 = Parameter("x1"), Parameter("x2")
    #     qc.ry(param_x_1, range(2))
    #     qc.ry(param_x_2, range(2))
    #
    #     # construct simple feature map
    #     param_y = Parameter("y")
    #     qc.ry(param_y, range(2))
    #
    #     qnn = CircuitQNN(
    #         qc,
    #         [param_x_1, param_x_2],
    #         [param_y],
    #         sparse=sparse,
    #         sampling=False,
    #         interpret=interpret,
    #         output_shape=output_shape,
    #         quantum_instance=quantum_instance,
    #         input_gradients=True,
    #     )
    #     model = TorchConnector(qnn)
    #
    #     test_data = [
    #         Tensor([1]),
    #         Tensor([1, 2]),
    #         Tensor([[1], [2]]),
    #         Tensor([[1, 2], [3, 4]]),
    #         Tensor([[[1], [2]], [[3], [4]]]),
    #     ]
    #
    #     # test model
    #     self.validate_output_shape(model, test_data)
    #     if q_i == "sv":
    #         self.validate_backward_pass(model)
    #
    # @data(
    #     # interpret
    #     (None),
    #     (lambda x: np.sum(x) % 2),
    # )
    # @requires_extra_library
    # def test_circuit_qnn_sampling(self, interpret):
    #     """Test Torch Connector + Circuit QNN for sampling."""
    #
    #     qc = QuantumCircuit(2)
    #
    #     # construct simple feature map
    #     param_x1, param_x2 = Parameter("x1"), Parameter("x2")
    #     qc.ry(param_x1, range(2))
    #     qc.ry(param_x2, range(2))
    #
    #     # construct simple feature map
    #     param_y = Parameter("y")
    #     qc.ry(param_y, range(2))
    #
    #     qnn = CircuitQNN(
    #         qc,
    #         [param_x1, param_x2],
    #         [param_y],
    #         sparse=False,
    #         sampling=True,
    #         interpret=interpret,
    #         output_shape=None,
    #         quantum_instance=self.qasm_quantum_instance,
    #         input_gradients=True,
    #     )
    #     model = TorchConnector(qnn)
    #
    #     test_data = [Tensor([2, 2]), Tensor([[1, 1], [2, 2]])]
    #     for i, x in enumerate(test_data):
    #         if i == 0:
    #             self.assertEqual(model(x).shape, qnn.output_shape)
    #         else:
    #             shape = model(x).shape
    #             self.assertEqual(shape, (len(x), *qnn.output_shape))
    #
    # @data(
    #     # output_shape, interpret
    #     (1, None),
    #     (2, lambda x: "{:b}".format(x).count("1") % 2),
    # )
    # @requires_extra_library
    # def test_circuit_qnn_batch_gradients(self, config):
    #     """Test batch gradient computation of CircuitQNN gives the same result as the sum of
    #     individual gradients."""
    #
    #     output_shape, interpret = config
    #     num_inputs = 2
    #
    #     feature_map = ZZFeatureMap(num_inputs)
    #     ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)
    #
    #     qc = QuantumCircuit(num_inputs)
    #     qc.append(feature_map, range(num_inputs))
    #     qc.append(ansatz, range(num_inputs))
    #
    #     qnn = CircuitQNN(
    #         qc,
    #         input_params=feature_map.parameters,
    #         weight_params=ansatz.parameters,
    #         interpret=interpret,
    #         output_shape=output_shape,
    #         quantum_instance=self.sv_quantum_instance,
    #     )
    #
    #     # set up PyTorch module
    #     initial_weights = np.array([0.1] * qnn.num_weights)
    #     model = TorchConnector(qnn, initial_weights)
    #
    #     # random data set
    #     x = Tensor(np.random.rand(5, 2))
    #     y = Tensor(np.random.rand(5, output_shape))
    #
    #     # define optimizer and loss
    #     optimizer = SGD(model.parameters(), lr=0.1)
    #     f_loss = MSELoss(reduction="sum")
    #
    #     sum_of_individual_losses = 0.0
    #     for x_i, y_i in zip(x, y):
    #         output = model(x_i)
    #         sum_of_individual_losses += f_loss(output, y_i)
    #     optimizer.zero_grad()
    #     sum_of_individual_losses.backward()
    #     sum_of_individual_gradients = deepcopy(model.weight.grad)
    #
    #     output = model(x)
    #     batch_loss = f_loss(output, y)
    #     optimizer.zero_grad()
    #     batch_loss.backward()
    #     batch_gradients = deepcopy(model.weight.grad)
    #
    #     self.assertAlmostEqual(
    #         np.linalg.norm(sum_of_individual_gradients - batch_gradients), 0.0, places=4
    #     )
    #
    #     self.assertAlmostEqual(
    #         sum_of_individual_losses.detach().numpy(),
    #         batch_loss.detach().numpy(),
    #         places=4,
    #     )


if __name__ == "__main__":
    unittest.main()
