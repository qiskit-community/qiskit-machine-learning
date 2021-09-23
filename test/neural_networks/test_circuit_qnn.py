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

"""Test Opflow QNN."""

import unittest

from test import QiskitMachineLearningTestCase, requires_extra_library

from ddt import ddt, data

import numpy as np

from qiskit import Aer
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import CircuitQNN

QASM = "qasm"

STATEVECTOR = "statevector"


@ddt
class TestCircuitQNN(QiskitMachineLearningTestCase):
    """Opflow QNN Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345
        # specify "run configuration"
        self.quantum_instance_sv = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        self.quantum_instance_qasm = QuantumInstance(
            AerSimulator(),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

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
            return sum([s == "1" for s in f"{x:0b}"]) % 2

        self.interpret_1d = interpret_1d
        self.output_shape_1d = 2  # takes values in {0, 1}

        def interpret_2d(x):
            return np.array([self.interpret_1d(x), 2 * self.interpret_1d(x)])

        self.interpret_2d = interpret_2d
        self.output_shape_2d = (
            2,
            3,
        )  # 1st dim. takes values in {0, 1} 2nd dim in {0, 1, 2}

    def _get_qnn(self, sparse, sampling, quantum_instance_type, interpret_id):
        """Construct QNN from configuration."""

        # get quantum instance
        if quantum_instance_type == STATEVECTOR:
            quantum_instance = self.quantum_instance_sv
        elif quantum_instance_type == QASM:
            quantum_instance = self.quantum_instance_qasm
        else:
            quantum_instance = None

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
        qnn = CircuitQNN(
            self.qc,
            self.input_params,
            self.weight_params,
            sparse=sparse,
            sampling=sampling,
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
        )
        return qnn

    def _verify_qnn(
        self,
        qnn: CircuitQNN,
        quantum_instance_type: str,
        batch_size: int,
    ) -> None:
        """
        Verifies that a QNN functions correctly

        Args:
            qnn: a QNN to check
            quantum_instance_type:
            batch_size:

        Returns:
            None.

        Raises:
            MissingOptionalLibraryError: if ``sparse`` library is not installed.
        """
        try:
            from sparse import SparseArray
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="sparse",
                name="SparseArray",
                pip_install="pip install 'qiskit-machine-learning[sparse]'",
            ) from ex

        input_data = np.zeros((batch_size, qnn.num_inputs))
        weights = np.zeros(qnn.num_weights)

        # if sampling and statevector, make sure it fails
        if quantum_instance_type == STATEVECTOR and qnn.sampling:
            with self.assertRaises(QiskitMachineLearningError):
                qnn.forward(input_data, weights)
        else:
            # evaluate QNN forward pass
            result = qnn.forward(input_data, weights)

            # make sure forward result is sparse if it should be
            if qnn.sparse and not qnn.sampling:
                self.assertTrue(isinstance(result, SparseArray))
            else:
                self.assertTrue(isinstance(result, np.ndarray))

            # check forward result shape
            self.assertEqual(result.shape, (batch_size, *qnn.output_shape))

            input_grad, weights_grad = qnn.backward(input_data, weights)
            if qnn.sampling:
                self.assertIsNone(input_grad)
                self.assertIsNone(weights_grad)
            else:
                self.assertIsNone(input_grad)
                self.assertEqual(
                    weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights)
                )

            # verify that input gradients are None if turned off
            qnn.input_gradients = True
            input_grad, weights_grad = qnn.backward(input_data, weights)
            if qnn.sampling:
                self.assertIsNone(input_grad)
                self.assertIsNone(weights_grad)
            else:
                self.assertEqual(input_grad.shape, (batch_size, *qnn.output_shape, qnn.num_inputs))
                self.assertEqual(
                    weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights)
                )

    @data(
        # sparse, sampling, quantum_instance_type, interpret (0=no, 1=1d, 2=2d), batch_size
        (True, True, STATEVECTOR, 0, 1),
        (True, True, STATEVECTOR, 0, 2),
        (True, True, STATEVECTOR, 1, 1),
        (True, True, STATEVECTOR, 1, 2),
        (True, True, STATEVECTOR, 2, 1),
        (True, True, STATEVECTOR, 2, 2),
        (True, True, QASM, 0, 1),
        (True, True, QASM, 0, 2),
        (True, True, QASM, 1, 1),
        (True, True, QASM, 1, 2),
        (True, True, QASM, 2, 1),
        (True, True, QASM, 2, 2),
        (True, False, STATEVECTOR, 0, 1),
        (True, False, STATEVECTOR, 0, 2),
        (True, False, STATEVECTOR, 1, 1),
        (True, False, STATEVECTOR, 1, 2),
        (True, False, STATEVECTOR, 2, 1),
        (True, False, STATEVECTOR, 2, 2),
        (True, False, QASM, 0, 1),
        (True, False, QASM, 0, 2),
        (True, False, QASM, 1, 1),
        (True, False, QASM, 1, 2),
        (True, False, QASM, 2, 1),
        (True, False, QASM, 2, 2),
        (False, True, STATEVECTOR, 0, 1),
        (False, True, STATEVECTOR, 0, 2),
        (False, True, STATEVECTOR, 1, 1),
        (False, True, STATEVECTOR, 1, 2),
        (False, True, STATEVECTOR, 2, 1),
        (False, True, STATEVECTOR, 2, 2),
        (False, True, QASM, 0, 1),
        (False, True, QASM, 0, 2),
        (False, True, QASM, 1, 1),
        (False, True, QASM, 1, 2),
        (False, True, QASM, 2, 1),
        (False, True, QASM, 2, 2),
        (False, False, STATEVECTOR, 0, 1),
        (False, False, STATEVECTOR, 0, 2),
        (False, False, STATEVECTOR, 1, 1),
        (False, False, STATEVECTOR, 1, 2),
        (False, False, STATEVECTOR, 2, 1),
        (False, False, STATEVECTOR, 2, 2),
        (False, False, QASM, 0, 1),
        (False, False, QASM, 0, 2),
        (False, False, QASM, 1, 1),
        (False, False, QASM, 1, 2),
        (False, False, QASM, 2, 1),
        (False, False, QASM, 2, 2),
    )
    @requires_extra_library
    def test_circuit_qnn(self, config):
        """Circuit QNN Test."""
        # get configuration
        sparse, sampling, quantum_instance_type, interpret_id, batch_size = config

        # get QNN
        qnn = self._get_qnn(sparse, sampling, quantum_instance_type, interpret_id)
        self._verify_qnn(qnn, quantum_instance_type, batch_size)

    @data(
        # sparse, sampling, quantum_instance_type, interpret (0=no, 1=1d, 2=2d), batch_size
        (True, False, STATEVECTOR, 0, 1),
        (True, False, STATEVECTOR, 0, 2),
        (True, False, STATEVECTOR, 1, 1),
        (True, False, STATEVECTOR, 1, 2),
        (True, False, STATEVECTOR, 2, 1),
        (True, False, STATEVECTOR, 2, 2),
        (False, False, STATEVECTOR, 0, 1),
        (False, False, STATEVECTOR, 0, 2),
        (False, False, STATEVECTOR, 1, 1),
        (False, False, STATEVECTOR, 1, 2),
        (False, False, STATEVECTOR, 2, 1),
        (False, False, STATEVECTOR, 2, 2),
    )
    @requires_extra_library
    def test_circuit_qnn_gradient(self, config):
        """Circuit QNN Gradient Test."""

        # get configuration
        sparse, sampling, quantum_instance_type, interpret_id, batch_size = config

        # get QNN
        qnn = self._get_qnn(sparse, sampling, quantum_instance_type, interpret_id)
        qnn.input_gradients = True
        input_data = np.ones((batch_size, qnn.num_inputs))
        weights = np.ones(qnn.num_weights)
        input_grad, weights_grad = qnn.backward(input_data, weights)

        # test input gradients
        eps = 1e-2
        for k in range(qnn.num_inputs):
            delta = np.zeros(input_data.shape)
            delta[:, k] = eps

            f_1 = qnn.forward(input_data + delta, weights)
            f_2 = qnn.forward(input_data - delta, weights)
            if sparse:
                grad = (f_1.todense() - f_2.todense()) / (2 * eps)
                input_grad_ = (
                    input_grad.todense()
                    .reshape((batch_size, -1, qnn.num_inputs))[:, :, k]
                    .reshape(grad.shape)
                )
                diff = input_grad_ - grad
            else:
                grad = (f_1 - f_2) / (2 * eps)
                input_grad_ = input_grad.reshape((batch_size, -1, qnn.num_inputs))[:, :, k].reshape(
                    grad.shape
                )
                diff = input_grad_ - grad
            self.assertAlmostEqual(np.max(np.abs(diff)), 0.0, places=3)

        # test weight gradients
        eps = 1e-2
        for k in range(qnn.num_weights):
            delta = np.zeros(weights.shape)
            delta[k] = eps

            f_1 = qnn.forward(input_data, weights + delta)
            f_2 = qnn.forward(input_data, weights - delta)
            if sparse:
                grad = (f_1.todense() - f_2.todense()) / (2 * eps)
                weights_grad_ = (
                    weights_grad.todense()
                    .reshape((batch_size, -1, qnn.num_weights))[:, :, k]
                    .reshape(grad.shape)
                )
                diff = weights_grad_ - grad
            else:
                grad = (f_1 - f_2) / (2 * eps)
                weights_grad_ = weights_grad.reshape((batch_size, -1, qnn.num_weights))[
                    :, :, k
                ].reshape(grad.shape)
                diff = weights_grad_ - grad
            self.assertAlmostEqual(np.max(np.abs(diff)), 0.0, places=3)

    @data(
        # sparse, sampling, quantum_instance_type, interpret (0=no, 1=1d, 2=2d), batch_size
        (True, True, STATEVECTOR, 0, 1),
        (True, True, STATEVECTOR, 0, 2),
        (True, True, STATEVECTOR, 1, 1),
        (True, True, STATEVECTOR, 1, 2),
        (True, True, STATEVECTOR, 2, 1),
        (True, True, STATEVECTOR, 2, 2),
        (True, True, QASM, 0, 1),
        (True, True, QASM, 0, 2),
        (True, True, QASM, 1, 1),
        (True, True, QASM, 1, 2),
        (True, True, QASM, 2, 1),
        (True, False, STATEVECTOR, 0, 1),
        (True, False, STATEVECTOR, 0, 2),
        (True, False, STATEVECTOR, 1, 1),
        (True, False, STATEVECTOR, 1, 2),
        (True, False, STATEVECTOR, 2, 1),
        (True, False, STATEVECTOR, 2, 2),
        (True, False, QASM, 0, 1),
        (True, False, QASM, 0, 2),
        (True, False, QASM, 1, 1),
        (True, False, QASM, 1, 2),
        (True, False, QASM, 2, 1),
        (True, False, QASM, 2, 2),
        (False, True, STATEVECTOR, 0, 1),
        (False, True, STATEVECTOR, 0, 2),
        (False, True, STATEVECTOR, 1, 1),
        (False, True, STATEVECTOR, 1, 2),
        (False, True, STATEVECTOR, 2, 1),
        (False, True, STATEVECTOR, 2, 2),
        (False, True, QASM, 0, 1),
        (False, True, QASM, 0, 2),
        (False, True, QASM, 1, 1),
        (False, True, QASM, 1, 2),
        (False, True, QASM, 2, 1),
        (False, True, QASM, 2, 2),
        (False, False, STATEVECTOR, 0, 1),
        (False, False, STATEVECTOR, 0, 2),
        (False, False, STATEVECTOR, 1, 1),
        (False, False, STATEVECTOR, 1, 2),
        (False, False, STATEVECTOR, 2, 1),
        (False, False, STATEVECTOR, 2, 2),
        (False, False, QASM, 0, 1),
        (False, False, QASM, 0, 2),
        (False, False, QASM, 1, 1),
        (False, False, QASM, 1, 2),
        (False, False, QASM, 2, 1),
        (False, False, QASM, 2, 2),
    )
    @requires_extra_library
    def test_no_quantum_instance(self, config):
        """Circuit QNN Test with and without QuantumInstance."""
        # get configuration
        sparse, sampling, quantum_instance_type, interpret_id, batch_size = config

        # get QNN with QuantumInstance
        qnn_qi = self._get_qnn(sparse, sampling, quantum_instance_type, interpret_id)

        # get QNN without QuantumInstance
        qnn_no_qi = self._get_qnn(sparse, sampling, None, interpret_id)

        with self.assertRaises(QiskitMachineLearningError):
            qnn_no_qi.sample(input_data=None, weights=None)

        with self.assertRaises(QiskitMachineLearningError):
            qnn_no_qi.probabilities(input_data=None, weights=None)

        with self.assertRaises(QiskitMachineLearningError):
            qnn_no_qi.probability_gradients(input_data=None, weights=None)

        if quantum_instance_type == STATEVECTOR:
            quantum_instance = self.quantum_instance_sv
        elif quantum_instance_type == QASM:
            quantum_instance = self.quantum_instance_qasm
        else:
            # must never happen
            quantum_instance = None

        qnn_no_qi.quantum_instance = quantum_instance

        self.assertEqual(qnn_qi.output_shape, qnn_no_qi.output_shape)
        self._verify_qnn(qnn_no_qi, quantum_instance_type, batch_size)


if __name__ == "__main__":
    unittest.main()
