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
""" Test TrainableQuantumKernel using primitives """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel


@ddt
class TestPrimitivesTrainableQuantumKernelClassify(QiskitMachineLearningTestCase):
    """Test trainable QuantumKernel."""

    def setUp(self):
        super().setUp()

        # Create an arbitrary 3-qubit feature map circuit
        circ1 = ZZFeatureMap(3)
        circ2 = ZZFeatureMap(3, parameter_prefix="θ")
        self.feature_map = circ1.compose(circ2).compose(circ1)
        self.num_features = circ1.num_parameters
        self.training_parameters = circ2.parameters

        self.sample_train = np.array(
            [[0.53833689, 0.44832616, 0.74399926], [0.43359057, 0.11213606, 0.97568932]]
        )
        self.sample_test = np.array([0.0, 1.0, 2.0])

    def test_training_parameters(self):
        """Test assigning/re-assigning user parameters"""

        kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        with self.subTest("check basic instantiation"):
            # Ensure we can instantiate a QuantumKernel with user parameters
            self.assertEqual(kernel._training_parameters, self.training_parameters)

        with self.subTest("test wrong number of parameters"):
            # Try to set the user parameters using an incorrect number of values
            training_param_values = [2.0, 4.0, 6.0, 8.0]
            with self.assertRaises(ValueError):
                kernel.assign_training_parameters(training_param_values)

        with self.subTest("test invalid parameter assignment"):
            # Try to set the user parameters using incorrect parameter
            param_binds = {Parameter("x"): 0.5}
            with self.assertRaises(ValueError):
                kernel.assign_training_parameters(param_binds)

        with self.subTest("test parameter assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = {
                self.training_parameters[0]: 0.1,
                self.training_parameters[1]: 0.2,
                self.training_parameters[2]: 0.3,
            }
            kernel.assign_training_parameters(param_binds)

            # Ensure the values are properly bound
            np.testing.assert_array_equal(kernel.parameter_values, list(param_binds.values()))

        with self.subTest("test partial parameter assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = {self.training_parameters[0]: 0.5, self.training_parameters[1]: 0.4}
            kernel.assign_training_parameters(param_binds)

            # Ensure values were properly bound and param 2 was unchanged
            np.testing.assert_array_equal(kernel.parameter_values, [0.5, 0.4, 0.3])

        with self.subTest("test parameter list assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = [0.1, 0.7, 1.7]
            kernel.assign_training_parameters(param_binds)

            # Ensure the values are properly bound
            np.testing.assert_array_equal(kernel.parameter_values, param_binds)

        with self.subTest("test parameter array assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = np.array([0.1, 0.7, 1.7])
            kernel.assign_training_parameters(param_binds)

            # Ensure the values are properly bound
            np.testing.assert_array_equal(kernel.parameter_values, param_binds)

    @data("params_1", "params_2")
    def test_evaluate_symmetric(self, params):
        """Test kernel evaluations for different training parameters"""
        if params == "params_1":
            training_params = [0.0, 0.0, 0.0]
        else:
            training_params = [0.1, 0.531, 4.12]

        kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        kernel.assign_training_parameters(training_params)
        kernel_matrix = kernel.evaluate(self.sample_train)

        # Ensure that the calculations are correct
        np.testing.assert_allclose(
            kernel_matrix, self._get_symmetric_solution(params), rtol=1e-7, atol=1e-7
        )

    def _get_symmetric_solution(self, params):
        if params == "params_1":
            return np.array([[1.0, 0.03351197], [0.03351197, 1.0]])
        return np.array([[1.0, 0.082392], [0.082392, 1.0]])

    @data("params_1", "params_2")
    def test_evaluate_asymmetric(self, params):
        """Test kernel evaluations for different training parameters"""
        if params == "params_1":
            training_params = [0.0, 0.0, 0.0]
        else:
            training_params = [0.1, 0.531, 4.12]

        kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        kernel.assign_training_parameters(training_params)
        kernel_matrix = kernel.evaluate(self.sample_train, self.sample_test)

        # Ensure that the calculations are correct
        np.testing.assert_allclose(
            kernel_matrix, self._get_asymmetric_solution(params), rtol=1e-7, atol=1e-7
        )

    def _get_asymmetric_solution(self, params):
        if params == "params_1":
            return np.array([[0.00569059], [0.07038205]])
        return np.array([[0.10568674], [0.122404]])

    def test_incomplete_binding(self):
        """Test if an exception is raised when not all training parameter are bound."""
        # assign all parameters except the last one
        training_params = {
            self.training_parameters[i]: 0 for i in range(len(self.training_parameters) - 1)
        }

        kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        kernel.assign_training_parameters(training_params)
        with self.assertRaises(QiskitMachineLearningError):
            kernel.evaluate(self.sample_train)

    def test_properties(self):
        """Test properties of the trainable quantum kernel."""
        kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )
        self.assertEqual(len(self.training_parameters), kernel.num_training_parameters)
        self.assertEqual(self.num_features, kernel.num_features)


if __name__ == "__main__":
    unittest.main()
