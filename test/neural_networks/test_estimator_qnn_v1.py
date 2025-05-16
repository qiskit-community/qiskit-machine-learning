# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test EstimatorQNN """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

CASE_DATA = {
    "shape_1_1": {
        "test_data": [1, [1], [[1], [2]], [[[1], [2]], [[3], [4]]]],
        "weights": [1],
        "correct_forwards": [
            [[0.08565359]],
            [[0.08565359]],
            [[0.08565359], [-0.90744233]],
            [[[0.08565359], [-0.90744233]], [[-1.06623996], [-0.24474149]]],
        ],
        "correct_weight_backwards": [
            [[[0.70807342]]],
            [[[0.70807342]]],
            [[[0.70807342]], [[0.7651474]]],
            [[[[0.70807342]], [[0.7651474]]], [[[0.11874839]], [[-0.63682734]]]],
        ],
        "correct_input_backwards": [
            [[[-1.13339757]]],
            [[[-1.13339757]]],
            [[[-1.13339757]], [[-0.68445233]]],
            [[[[-1.13339757]], [[-0.68445233]]], [[[0.39377522]], [[1.10996765]]]],
        ],
    },
    "shape_2_1": {
        "test_data": [[1, 2], [[1, 2]], [[1, 2], [3, 4]]],
        "weights": [1, 2],
        "correct_forwards": [
            [[0.41256026]],
            [[0.41256026]],
            [[0.41256026], [0.72848859]],
        ],
        "correct_weight_backwards": [
            [[[0.12262287, -0.17203964]]],
            [[[0.12262287, -0.17203964]]],
            [[[0.12262287, -0.17203964]], [[0.03230095, -0.04531817]]],
        ],
        "correct_input_backwards": [
            [[[-0.81570272, -0.39688474]]],
            [[[-0.81570272, -0.39688474]]],
            [[[-0.81570272, -0.39688474]], [[0.25229775, 0.67111573]]],
        ],
    },
    "shape_1_2": {
        "test_data": [
            [1],
            [[1], [2]],
            [[[1], [2]], [[3], [4]]],
        ],
        "weights": [1],
        "correct_forwards": [
            [[0.08565359, 0.17130718]],
            [[0.08565359, 0.17130718], [-0.90744233, -1.81488467]],
            [
                [[0.08565359, 0.17130718], [-0.90744233, -1.81488467]],
                [[-1.06623996, -2.13247992], [-0.24474149, -0.48948298]],
            ],
        ],
        "correct_weight_backwards": [
            [[[0.70807342], [1.41614684]]],
            [[[0.70807342], [1.41614684]], [[0.7651474], [1.5302948]]],
            [
                [[[0.70807342], [1.41614684]], [[0.7651474], [1.5302948]]],
                [[[0.11874839], [0.23749678]], [[-0.63682734], [-1.27365468]]],
            ],
        ],
        "correct_input_backwards": [
            [[[-1.13339757], [-2.26679513]]],
            [[[-1.13339757], [-2.26679513]], [[-0.68445233], [-1.36890466]]],
            [
                [[[-1.13339757], [-2.26679513]], [[-0.68445233], [-1.36890466]]],
                [[[0.39377522], [0.78755044]], [[1.10996765], [2.2199353]]],
            ],
        ],
    },
    "shape_2_2": {
        "test_data": [[1, 2], [[1, 2], [3, 4]]],
        "weights": [1, 2],
        "correct_forwards": [
            [[-0.07873524, 0.4912955]],
            [[-0.07873524, 0.4912955], [-0.0207402, 0.74922879]],
        ],
        "correct_weight_backwards": [
            [[[0.12262287, -0.17203964], [0, 0]]],
            [[[0.12262287, -0.17203964], [0, 0]], [[0.03230095, -0.04531817], [0, 0]]],
        ],
        "correct_input_backwards": [
            [[[-0.05055532, -0.17203964], [-0.7651474, -0.2248451]]],
            [
                [[-0.05055532, -0.17203964], [-0.7651474, -0.2248451]],
                [[0.14549777, 0.02401345], [0.10679997, 0.64710228]],
            ],
        ],
    },
    "no_input_parameters": {
        "test_data": [None],
        "weights": [1, 1],
        "correct_forwards": [[[0.08565359]]],
        "correct_weight_backwards": [[[[-1.13339757, 0.70807342]]]],
        "correct_input_backwards": [None],
    },
    "no_weight_parameters": {
        "test_data": [[1, 1]],
        "weights": None,
        "correct_forwards": [[[0.08565359]]],
        "correct_weight_backwards": [None],
        "correct_input_backwards": [[[[-1.13339757, 0.70807342]]]],
    },
    "no_parameters": {
        "test_data": [None],
        "weights": None,
        "correct_forwards": [[[1]]],
        "correct_weight_backwards": [None],
        "correct_input_backwards": [None],
    },
    "default_observables": {
        "test_data": [[[1], [2]]],
        "weights": [1],
        "correct_forwards": [[[-0.45464871], [-0.4912955]]],
        "correct_weight_backwards": [[[[0.70807342]], [[0.7651474]]]],
        "correct_input_backwards": [[[[-0.29192658]], [[0.2248451]]]],
    },
    "single_observable": {
        "test_data": [1, [1], [[1], [2]], [[[1], [2]], [[3], [4]]]],
        "weights": [1],
        "correct_forwards": [
            [[0.08565359]],
            [[0.08565359]],
            [[0.08565359], [-0.90744233]],
            [[[0.08565359], [-0.90744233]], [[-1.06623996], [-0.24474149]]],
        ],
        "correct_weight_backwards": [
            [[[0.70807342]]],
            [[[0.70807342]]],
            [[[0.70807342]], [[0.7651474]]],
            [[[[0.70807342]], [[0.7651474]]], [[[0.11874839]], [[-0.63682734]]]],
        ],
        "correct_input_backwards": [
            [[[-1.13339757]]],
            [[[-1.13339757]]],
            [[[-1.13339757]], [[-0.68445233]]],
            [[[[-1.13339757]], [[-0.68445233]]], [[[0.39377522]], [[1.10996765]]]],
        ],
    },
}


class TestEstimatorQNN(QiskitMachineLearningTestCase):
    """EstimatorQNN Tests. The correct references is obtained from EstimatorQNN"""

    def _test_network_passes(
        self,
        estimator_qnn,
        case_data,
    ):
        algorithm_globals.random_seed = 52
        test_data = case_data["test_data"]
        weights = case_data["weights"]
        correct_forwards = case_data["correct_forwards"]
        correct_weight_backwards = case_data["correct_weight_backwards"]
        correct_input_backwards = case_data["correct_input_backwards"]

        # test forward pass
        with self.subTest("forward pass"):
            for i, inputs in enumerate(test_data):
                forward = estimator_qnn.forward(inputs, weights)
                np.testing.assert_allclose(forward, correct_forwards[i], atol=1e-3)
        # test backward pass without input_gradients
        with self.subTest("backward pass without input gradients"):
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                if correct_weight_backwards[i] is None:
                    self.assertIsNone(weight_backward)
                else:
                    np.testing.assert_allclose(
                        weight_backward, correct_weight_backwards[i], atol=1e-3
                    )
                self.assertIsNone(input_backward)
        # test backward pass with input_gradients
        with self.subTest("backward pass with input gradients"):
            estimator_qnn.input_gradients = True
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                if correct_weight_backwards[i] is None:
                    self.assertIsNone(weight_backward)
                else:
                    np.testing.assert_allclose(
                        weight_backward, correct_weight_backwards[i], atol=1e-3
                    )
                if correct_input_backwards[i] is None:
                    self.assertIsNone(input_backward)
                else:
                    np.testing.assert_allclose(
                        input_backward, correct_input_backwards[i], atol=1e-3
                    )

    def test_estimator_qnn_1_1(self):
        """Test Estimator QNN with input/output dimension 1/1."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op],
            input_params=[params[0]],
            weight_params=[params[1]],
        )

        self._test_network_passes(estimator_qnn, CASE_DATA["shape_1_1"])

    def test_estimator_qnn_2_1(self):
        """Test Estimator QNN with input/output dimension 2/1."""
        params = [
            Parameter("input1"),
            Parameter("input2"),
            Parameter("weight1"),
            Parameter("weight2"),
        ]
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.rx(params[2], 0)
        qc.rx(params[3], 1)
        op = SparsePauliOp.from_list([("ZZ", 1), ("XX", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op],
            input_params=params[:2],
            weight_params=params[2:],
        )

        self._test_network_passes(estimator_qnn, CASE_DATA["shape_2_1"])

    def test_estimator_qnn_1_2(self):
        """Test Estimator QNN with input/output dimension 1/2."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        op1 = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        op2 = SparsePauliOp.from_list([("Z", 2), ("X", 2)])

        # construct QNN
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op1, op2],
            input_params=[params[0]],
            weight_params=[params[1]],
        )

        self._test_network_passes(estimator_qnn, CASE_DATA["shape_1_2"])

    def test_estimator_qnn_2_2(self):
        """Test Estimator QNN with input/output dimension 2/2."""
        params = [
            Parameter("input1"),
            Parameter("input2"),
            Parameter("weight1"),
            Parameter("weight2"),
        ]
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.rx(params[2], 0)
        qc.rx(params[3], 1)
        op1 = SparsePauliOp.from_list([("ZZ", 1)])
        op2 = SparsePauliOp.from_list([("XX", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op1, op2],
            input_params=params[:2],
            weight_params=params[2:],
        )

        self._test_network_passes(estimator_qnn, CASE_DATA["shape_2_2"])

    def test_no_input_parameters(self):
        """Test Estimator QNN with no input parameters."""
        params = [Parameter("weight0"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op],
            input_params=None,
            weight_params=params,
        )
        self._test_network_passes(estimator_qnn, CASE_DATA["no_input_parameters"])

    def test_no_weight_parameters(self):
        """Test Estimator QNN with no weight parameters."""
        params = [Parameter("input0"), Parameter("input1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op],
            input_params=params,
            weight_params=None,
        )
        self._test_network_passes(estimator_qnn, CASE_DATA["no_weight_parameters"])

    def test_no_parameters(self):
        """Test Estimator QNN with no parameters."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op],
            input_params=None,
            weight_params=None,
        )
        self._test_network_passes(estimator_qnn, CASE_DATA["no_parameters"])

    def test_default_observables(self):
        """Test Estimator QNN with default observables."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            input_params=[params[0]],
            weight_params=[params[1]],
        )
        self._test_network_passes(estimator_qnn, CASE_DATA["default_observables"])

    def test_single_observable(self):
        """Test Estimator QNN with single observable."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=op,
            input_params=[params[0]],
            weight_params=[params[1]],
        )
        self._test_network_passes(estimator_qnn, CASE_DATA["single_observable"])

    def test_setters_getters(self):
        """Test Estimator QNN properties."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[op],
            input_params=[params[0]],
            weight_params=[params[1]],
        )
        with self.subTest("Test circuit getter."):
            self.assertEqual(estimator_qnn.circuit, qc)
        with self.subTest("Test observables getter."):
            self.assertEqual(estimator_qnn.observables, [op])
        with self.subTest("Test input_params getter."):
            self.assertEqual(estimator_qnn.input_params, [params[0]])
        with self.subTest("Test weight_params getter."):
            self.assertEqual(estimator_qnn.weight_params, [params[1]])
        with self.subTest("Test input_gradients setter and getter."):
            self.assertFalse(estimator_qnn.input_gradients)
            estimator_qnn.input_gradients = True
            self.assertTrue(estimator_qnn.input_gradients)

    def test_qnn_qc_circuit_construction(self):
        """Test Estimator QNN properties and forward/backward pass for QNNCircuit construction"""
        num_qubits = 2
        feature_map = ZZFeatureMap(feature_dimension=num_qubits)
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)

        qnn_qc = QNNCircuit(num_qubits=num_qubits, feature_map=feature_map, ansatz=ansatz)
        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        estimator_qc = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )
        estimator_qnn_qc = EstimatorQNN(circuit=qnn_qc, input_gradients=True)

        input_data = [1, 2]
        weights = [1, 2, 3, 4]

        with self.subTest("Test if Estimator QNN properties are equal."):
            self.assertEqual(estimator_qnn_qc.input_params, estimator_qc.input_params)
            self.assertEqual(estimator_qnn_qc.weight_params, estimator_qc.weight_params)
            self.assertEqual(estimator_qnn_qc.observables, estimator_qc.observables)

        with self.subTest("Test if forward pass yields equal results."):
            forward_qc = estimator_qc.forward(input_data=input_data, weights=weights)
            forward_qnn_qc = estimator_qnn_qc.forward(input_data=input_data, weights=weights)
            np.testing.assert_array_almost_equal(forward_qc, forward_qnn_qc)

        with self.subTest("Test if backward pass yields equal results."):
            backward_qc = estimator_qc.backward(input_data=input_data, weights=weights)
            backward_qnn_qc = estimator_qnn_qc.backward(input_data=input_data, weights=weights)
            # Test if input grad is identical
            np.testing.assert_array_almost_equal(backward_qc[0], backward_qnn_qc[0])
            # Test if weights grad is identical
            np.testing.assert_array_almost_equal(backward_qc[1], backward_qnn_qc[1])

    def test_binding_order(self):
        """Test parameter binding order gives result as expected"""
        qc = ZFeatureMap(feature_dimension=2, reps=1)
        input_params = qc.parameters
        weight = Parameter("weight")
        for i in range(qc.num_qubits):
            qc.rx(weight, i)

        observable1 = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])
        estimator_qnn = EstimatorQNN(
            circuit=qc, observables=observable1, input_params=input_params, weight_params=[weight]
        )

        estimator_qnn_weights = [3]
        estimator_qnn_input = [2, 33]
        res = estimator_qnn.forward(estimator_qnn_input, estimator_qnn_weights)
        # When parameters were used in circuit order, before being assigned correctly, so inputs
        # went to input params, weights to weight params, this gave 0.00613403
        self.assertAlmostEqual(res[0][0], 0.00040017)


if __name__ == "__main__":
    unittest.main()
