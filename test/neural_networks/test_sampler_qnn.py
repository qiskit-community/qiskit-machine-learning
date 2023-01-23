# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Sampler QNN with Terra primitives."""
from __future__ import annotations

from test import QiskitMachineLearningTestCase

import itertools
import unittest
import numpy as np

from ddt import ddt, idata

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.neural_networks.sampler_qnn import SamplerQNN
import qiskit_machine_learning.optionals as _optionals

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


DEFAULT = "default"
SHOTS = "shots"
SPARSE = [True, False]
SAMPLERS = [DEFAULT, SHOTS]
INTERPRET_TYPES = [0, 1, 2]
BATCH_SIZES = [2]
INPUT_GRADS = [True, False]


@ddt
class TestSamplerQNN(QiskitMachineLearningTestCase):
    """Sampler QNN Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345

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
        self.sampler_shots = Sampler(options={"shots": 100, "seed": 42})

        self.array_type = {True: SparseArray, False: np.ndarray}

    def _get_qnn(
        self, sparse, sampler_type, interpret_id, input_params, weight_params, input_grads
    ):
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
            input_params=input_params,
            weight_params=weight_params,
            sparse=sparse,
            interpret=interpret,
            output_shape=output_shape,
            input_gradients=input_grads,
        )
        return qnn

    def _verify_qnn(
        self,
        qnn: SamplerQNN,
        batch_size: int,
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> None:
        """
        Verifies that a QNN functions correctly
        """
        # evaluate QNN forward pass
        result = qnn.forward(input_data, weights)

        if input_data is None:
            batch_size = 1

        self.assertTrue(isinstance(result, self.array_type[qnn.sparse]))
        # check forward result shape
        self.assertEqual(result.shape, (batch_size, *qnn.output_shape))

        # evaluate QNN backward pass
        input_grad, weights_grad = qnn.backward(input_data, weights)

        if qnn.input_gradients:
            if input_data is not None:
                self.assertEqual(input_grad.shape, (batch_size, *qnn.output_shape, qnn.num_inputs))
                self.assertTrue(isinstance(input_grad, self.array_type[qnn.sparse]))
            else:
                # verify that input gradients are None if turned off
                self.assertIsNone(input_grad)
            if weights is not None:
                self.assertEqual(
                    weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights)
                )
                self.assertTrue(isinstance(weights_grad, self.array_type[qnn.sparse]))
            else:
                # verify that weight gradients are None if no weights
                self.assertIsNone(weights_grad)

        else:
            # verify that input gradients are None if turned off
            self.assertIsNone(input_grad)
            if weights is not None:
                self.assertEqual(
                    weights_grad.shape, (batch_size, *qnn.output_shape, qnn.num_weights)
                )
                self.assertTrue(isinstance(weights_grad, self.array_type[qnn.sparse]))
            else:
                # verify that weight gradients are None if no weights
                self.assertIsNone(weights_grad)

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    @idata(itertools.product(SPARSE, SAMPLERS, INTERPRET_TYPES, BATCH_SIZES, INPUT_GRADS))
    def test_sampler_qnn(self, config):
        """Sampler QNN Test."""

        sparse, sampler_type, interpret_type, batch_size, input_grads = config
        # Test QNN with input and weight params
        qnn = self._get_qnn(
            sparse,
            sampler_type,
            interpret_type,
            input_params=self.input_params,
            weight_params=self.weight_params,
            input_grads=True,
        )
        input_data = np.zeros((batch_size, qnn.num_inputs))
        weights = np.zeros(qnn.num_weights)
        self._verify_qnn(qnn, batch_size, input_data, weights)

        # Test QNN with no input params
        qnn = self._get_qnn(
            sparse,
            sampler_type,
            interpret_type,
            input_params=None,
            weight_params=self.weight_params + self.input_params,
            input_grads=input_grads,
        )
        input_data = None
        weights = np.zeros(qnn.num_weights)
        self._verify_qnn(qnn, batch_size, input_data, weights)

        # Test QNN with no weight params
        qnn = self._get_qnn(
            sparse,
            sampler_type,
            interpret_type,
            input_params=self.weight_params + self.input_params,
            weight_params=None,
            input_grads=input_grads,
        )
        input_data = np.zeros((batch_size, qnn.num_inputs))
        weights = None
        self._verify_qnn(qnn, batch_size, input_data, weights)

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    @idata(itertools.product(SPARSE, INTERPRET_TYPES, BATCH_SIZES))
    def test_sampler_qnn_gradient(self, config):
        """Sampler QNN Gradient Test."""

        # get configuration
        sparse, interpret_id, batch_size = config

        # get QNN
        qnn = self._get_qnn(
            sparse,
            DEFAULT,
            interpret_id,
            input_params=self.input_params,
            weight_params=self.weight_params,
            input_grads=True,
        )

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

    def test_setters_getters(self):
        """Test Sampler QNN properties."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        qc.measure_all()
        sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=[params[0]],
            weight_params=[params[1]],
        )
        with self.subTest("Test input_params getter."):
            self.assertEqual(sampler_qnn.input_params, [params[0]])
        with self.subTest("Test weight_params getter."):
            self.assertEqual(sampler_qnn.weight_params, [params[1]])
        with self.subTest("Test input_gradients setter and getter."):
            self.assertFalse(sampler_qnn.input_gradients)
            sampler_qnn.input_gradients = True
            self.assertTrue(sampler_qnn.input_gradients)

    def test_no_parameters(self):
        """Test when some parameters are not set."""
        params = [Parameter("p0"), Parameter("p1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        with self.subTest("no inputs"):
            sampler_qnn = SamplerQNN(
                circuit=qc,
                weight_params=params,
            )
            self._verify_qnn(sampler_qnn, 1, input_data=None, weights=[1, 2])

            sampler_qnn.input_gradients = True
            self._verify_qnn(sampler_qnn, 1, input_data=None, weights=[1, 2])

        with self.subTest("no weights"):
            sampler_qnn = SamplerQNN(
                circuit=qc,
                input_params=params,
            )
            self._verify_qnn(sampler_qnn, 1, input_data=[1, 2], weights=None)

            sampler_qnn.input_gradients = True
            self._verify_qnn(sampler_qnn, 1, input_data=[1, 2], weights=None)

        with self.subTest("no parameters"):
            qc = qc.bind_parameters([1, 2])

            sampler_qnn = SamplerQNN(
                circuit=qc,
            )

            self._verify_qnn(sampler_qnn, 1, input_data=None, weights=None)

            sampler_qnn.input_gradients = True
            self._verify_qnn(sampler_qnn, 1, input_data=None, weights=None)
