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

from test import QiskitMachineLearningTestCase

import itertools
import unittest
import numpy as np

from ddt import ddt, idata, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.algorithms.gradients import ParamShiftSamplerGradient
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.neural_networks.sampler_qnn import SamplerQNN
import qiskit_machine_learning.optionals as _optionals

algorithm_globals.random_seed = 42

DEFAULT = "default"
SHOTS = "shots"
SPARSE = [True, False]
SAMPLERS = [DEFAULT, SHOTS]
INTERPRET_TYPES = [2]
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
        self.sampler_shots = Sampler(options={"shots": 100})

    def _get_qnn(self, sparse, sampler_type, interpret_id):
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
            sampler=sampler,
            circuit=self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sparse=sparse,
            interpret=interpret,
            output_shape=output_shape,
        )
        return qnn

    def _verify_qnn(
        self,
        qnn: CircuitQNN,
        batch_size: int,
    ) -> None:
        """
        Verifies that a QNN functions correctly
        """
        # pylint: disable=import-error
        from sparse import SparseArray

        input_data = np.zeros((batch_size, qnn.num_inputs))
        weights = np.zeros(qnn.num_weights)

        # evaluate QNN forward pass
        result = qnn.forward(input_data, weights)

        # make sure forward result is sparse if it should be
        if qnn.sparse:
            self.assertTrue(isinstance(result, SparseArray))
        else:
            self.assertTrue(isinstance(result, np.ndarray))

        # check forward result shape
        self.assertEqual(result.shape, (batch_size, *qnn.output_shape))

        # evaluate QNN backward pass
        input_grad, weights_grad = qnn.backward(input_data, weights)

        # verify that input gradients are None if turned off
        self.assertIsNone(input_grad)
        self.assertEqual(weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights))

        if qnn.sparse:
            self.assertTrue(isinstance(weights_grad, SparseArray))
        else:
            self.assertTrue(isinstance(weights_grad, np.ndarray))

        # verify that input gradients are not None if turned on
        qnn.input_gradients = True
        input_grad, weights_grad = qnn.backward(input_data, weights)

        self.assertEqual(input_grad.shape, (batch_size, *qnn.output_shape, qnn.num_inputs))
        self.assertEqual(weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights))

        if qnn.sparse:
            self.assertTrue(isinstance(weights_grad, SparseArray))
            self.assertTrue(isinstance(input_grad, SparseArray))
        else:
            self.assertTrue(isinstance(weights_grad, np.ndarray))
            self.assertTrue(isinstance(input_grad, np.ndarray))

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    @idata(itertools.product(SPARSE, SAMPLERS, INTERPRET_TYPES, BATCH_SIZES))
    @unpack
    def test_sampler_qnn(self, sparse: bool, sampler_type, interpret_type, batch_size):
        """Sampler QNN Test."""
        qnn = self._get_qnn(sparse, sampler_type, interpret_type)
        self._verify_qnn(qnn, batch_size)

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    @idata(itertools.product(SPARSE, INTERPRET_TYPES, BATCH_SIZES))
    def test_sampler_qnn_gradient(self, config):
        """Sampler QNN Gradient Test."""

        # get configuration
        sparse, interpret_id, batch_size = config

        # get QNN
        qnn = self._get_qnn(sparse, DEFAULT, interpret_id)

        # set input gradients to True
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

            grad = (f_1 - f_2) / (2 * eps)
            weights_grad_ = weights_grad.reshape((batch_size, -1, qnn.num_weights))[
                :, :, k
            ].reshape(grad.shape)
            diff = weights_grad_ - grad
            self.assertAlmostEqual(np.max(np.abs(diff)), 0.0, places=3)

    def test_circuit_vs_sampler_qnn(self):
        """Circuit vs Sampler QNN Test. To be removed once CircuitQNN is deprecated"""
        from qiskit.opflow import Gradient
        import importlib

        aer = importlib.import_module("qiskit.providers.aer")

        parity = lambda x: f"{x:b}".count("1") % 2
        output_shape = 2  # this is required in case of a callable with dense output

        qi_qasm = QuantumInstance(
            aer.AerSimulator(),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        circuit_qnn = CircuitQNN(
            self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=qi_qasm,
            gradient=Gradient("param_shift"),
            input_gradients=True,
        )

        sampler_qnn = SamplerQNN(
            sampler=self.sampler,
            circuit=self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            interpret=parity,
            output_shape=output_shape,
            gradient=ParamShiftSamplerGradient(self.sampler),
            input_gradients=True,
        )

        inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
        weights = algorithm_globals.random.random(circuit_qnn._num_weights)

        circuit_qnn_fwd = circuit_qnn.backward(inputs, weights)
        sampler_qnn_fwd = sampler_qnn.backward(inputs, weights)

        np.testing.assert_array_almost_equal(
            np.asarray(sampler_qnn_fwd), np.asarray(circuit_qnn_fwd), 0.1
        )
