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

""" Test EstimatorQNN """

from test import QiskitMachineLearningTestCase

import unittest

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN


class TestEstimatorQNN(QiskitMachineLearningTestCase):
    """EstimatorQNN Tests. The correct references is obtained from OpflowQNN"""

    def test_estimator_qnn_1_1(self):
        """Test Estimator QNN with input/output dimension 1/1."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        op = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        estimator = Estimator()
        gradient = ParamShiftEstimatorGradient(estimator)
        estimator_qnn = EstimatorQNN(
            estimator=estimator,
            circuit=qc,
            observables=[op],
            input_params=[params[0]],
            weight_params=[params[1]],
            gradient=gradient,
        )
        weights = np.array([1])

        test_data = [1, [1], [[1], [2]], [[[1], [2]], [[3], [4]]]]
        correct_forwards = [
            [[0.08565359]],
            [[0.08565359]],
            [[0.08565359], [-0.90744233]],
            [[[0.08565359], [-0.90744233]], [[-1.06623996], [-0.24474149]]],
        ]
        correct_weight_backwards = [
            [[[0.70807342]]],
            [[[0.70807342]]],
            [[[0.70807342]], [[0.7651474]]],
            [[[[0.70807342]], [[0.7651474]]], [[[0.11874839]], [[-0.63682734]]]],
        ]
        correct_input_backwards = [
            [[[-1.13339757]]],
            [[[-1.13339757]]],
            [[[-1.13339757]], [[-0.68445233]]],
            [[[[-1.13339757]], [[-0.68445233]]], [[[0.39377522]], [[1.10996765]]]],
        ]

        # test forward pass
        with self.subTest("forward pass"):
            for i, inputs in enumerate(test_data):
                forward = estimator_qnn.forward(inputs, weights)
                np.testing.assert_allclose(forward, correct_forwards[i], atol=1e-3)
        # test backward pass without input_gradients
        with self.subTest("backward pass without input gradients"):
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                np.testing.assert_allclose(weight_backward, correct_weight_backwards[i], atol=1e-3)
                self.assertIsNone(input_backward)
        # test backward pass with input_gradients
        with self.subTest("backward bass with input gradients"):
            estimator_qnn.input_gradients = True
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                np.testing.assert_allclose(weight_backward, correct_weight_backwards[i], atol=1e-3)
                np.testing.assert_allclose(input_backward, correct_input_backwards[i], atol=1e-3)

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
        estimator = Estimator()
        gradient = ParamShiftEstimatorGradient(estimator)
        estimator_qnn = EstimatorQNN(
            estimator=estimator,
            circuit=qc,
            observables=[op],
            input_params=params[:2],
            weight_params=params[2:],
            gradient=gradient,
        )
        weights = np.array([1, 2])

        test_data = [[1, 2], [[1, 2]], [[1, 2], [3, 4]]]
        correct_forwards = [
            [[0.41256026]],
            [[0.41256026]],
            [[0.41256026], [0.72848859]],
        ]
        correct_weight_backwards = [
            [[[0.12262287, -0.17203964]]],
            [[[0.12262287, -0.17203964]]],
            [[[0.12262287, -0.17203964]], [[0.03230095, -0.04531817]]],
        ]
        correct_input_backwards = [
            [[[-0.81570272, -0.39688474]]],
            [[[-0.81570272, -0.39688474]]],
            [[[-0.81570272, -0.39688474]], [[0.25229775, 0.67111573]]],
        ]
        # test forward pass
        with self.subTest("forward pass"):
            for i, inputs in enumerate(test_data):
                forward = estimator_qnn.forward(inputs, weights)
                np.testing.assert_allclose(forward, correct_forwards[i], atol=1e-3)
        # test backward pass without input_gradients
        with self.subTest("backward pass without input gradients"):
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                np.testing.assert_allclose(weight_backward, correct_weight_backwards[i], atol=1e-3)
                self.assertIsNone(input_backward)
        # test backward pass with input_gradients
        with self.subTest("backward bass with input gradients"):
            estimator_qnn.input_gradients = True
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                np.testing.assert_allclose(weight_backward, correct_weight_backwards[i], atol=1e-3)
                np.testing.assert_allclose(input_backward, correct_input_backwards[i], atol=1e-3)

    def test_estimator_qnn_1_2(self):
        """Test Estimator QNN with input/output dimension 1/2."""
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        op1 = SparsePauliOp.from_list([("Z", 1), ("X", 1)])
        op2 = SparsePauliOp.from_list([("Z", 2), ("X", 2)])

        estimator = Estimator()
        gradient = ParamShiftEstimatorGradient(estimator)
        # construct QNN
        estimator_qnn = EstimatorQNN(
            estimator=estimator,
            circuit=qc,
            observables=[op1, op2],
            input_params=[params[0]],
            weight_params=[params[1]],
            gradient=gradient,
        )
        weights = np.array([1])

        test_data = [
            np.array([1]),
            np.array([[1], [2]]),
            np.array([[[1], [2]], [[3], [4]]]),
        ]

        correct_forwards = [
            np.array([[0.08565359, 0.17130718]]),
            np.array([[0.08565359, 0.17130718], [-0.90744233, -1.81488467]]),
            np.array(
                [
                    [[0.08565359, 0.17130718], [-0.90744233, -1.81488467]],
                    [[-1.06623996, -2.13247992], [-0.24474149, -0.48948298]],
                ]
            ),
        ]
        correct_weight_backwards = [
            np.array([[[0.70807342], [1.41614684]]]),
            np.array([[[0.70807342], [1.41614684]], [[0.7651474], [1.5302948]]]),
            np.array(
                [
                    [[[0.70807342], [1.41614684]], [[0.7651474], [1.5302948]]],
                    [[[0.11874839], [0.23749678]], [[-0.63682734], [-1.27365468]]],
                ]
            ),
        ]
        correct_input_backwards = [
            np.array([[[-1.13339757], [-2.26679513]]]),
            np.array([[[-1.13339757], [-2.26679513]], [[-0.68445233], [-1.36890466]]]),
            np.array(
                [
                    [[[-1.13339757], [-2.26679513]], [[-0.68445233], [-1.36890466]]],
                    [[[0.39377522], [0.78755044]], [[1.10996765], [2.2199353]]],
                ]
            ),
        ]

        # test forward pass
        with self.subTest("forward pass"):
            for i, inputs in enumerate(test_data):
                forward = estimator_qnn.forward(inputs, weights)
                np.testing.assert_allclose(forward, correct_forwards[i], atol=1e-3)
        # test backward pass without input_gradients
        with self.subTest("backward pass without input gradients"):
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                np.testing.assert_allclose(weight_backward, correct_weight_backwards[i], atol=1e-3)
                self.assertIsNone(input_backward)
        # test backward pass with input_gradients
        with self.subTest("backward bass with input gradients"):
            estimator_qnn.input_gradients = True
            for i, inputs in enumerate(test_data):
                input_backward, weight_backward = estimator_qnn.backward(inputs, weights)
                np.testing.assert_allclose(weight_backward, correct_weight_backwards[i], atol=1e-3)
                np.testing.assert_allclose(input_backward, correct_input_backwards[i], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
