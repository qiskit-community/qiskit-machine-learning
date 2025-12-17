# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test trainable quantum kernels using primitives"""

import itertools
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data, idata, unpack
from qiskit.circuit import Parameter
from qiskit.circuit.library import zz_feature_map

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.kernels import (
    TrainableFidelityQuantumKernel,
    TrainableFidelityStatevectorKernel,
)


@ddt
class TestPrimitivesTrainableQuantumKernelClassify(QiskitMachineLearningTestCase):
    """Test the Primitive-based trainable quantum kernel, and the trainable statevector kernel."""

    def setUp(self):
        super().setUp()

        # Create an arbitrary 3-qubit feature map circuit
        circ1 = zz_feature_map(3)
        circ2 = zz_feature_map(3, parameter_prefix="θ")
        self.feature_map = circ1.compose(circ2).compose(circ1)
        self.num_features = circ1.num_parameters
        self.training_parameters = circ2.parameters
        
        self.sample_train = np.array(
            [[0.53833689, 0.44832616, 0.74399926], [0.43359057, 0.11213606, 0.97568932]]
        )
        self.sample_test = np.array([0.0, 1.0, 2.0])

    @data(TrainableFidelityQuantumKernel, TrainableFidelityStatevectorKernel)
    def test_training_parameters(self, trainable_kernel_type):
        """Test assigning/re-assigning user parameters"""

        kernel = trainable_kernel_type(
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

    @idata(
        itertools.product(
            [TrainableFidelityQuantumKernel, TrainableFidelityStatevectorKernel],
            [
                ([0.0, 0.0, 0.0], [[1.0, 0.03351197], [0.03351197, 1.0]]),
                ([0.1, 0.531, 4.12], [[1.0, 0.082392], [0.082392, 1.0]]),
            ],
        )
    )
    @unpack
    def test_evaluate_symmetric(self, trainable_kernel_type, params_solution):
        """Test kernel evaluations for different training parameters"""
        kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        kernel.assign_training_parameters(params_solution[0])
        kernel_matrix = kernel.evaluate(self.sample_train)

        # Ensure that the calculations are correct
        np.testing.assert_allclose(kernel_matrix, params_solution[1], rtol=1e-7, atol=1e-7)

    @idata(
        itertools.product(
            [TrainableFidelityQuantumKernel, TrainableFidelityStatevectorKernel],
            [
                ([0.0, 0.0, 0.0], [[0.00569059], [0.07038205]]),
                ([0.1, 0.531, 4.12], [[0.10568674], [0.122404]]),
            ],
        )
    )
    @unpack
    def test_evaluate_asymmetric(self, trainable_kernel_type, params_solution):
        """Test kernel evaluations for different training parameters"""
        kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        kernel.assign_training_parameters(params_solution[0])
        kernel_matrix = kernel.evaluate(self.sample_train, self.sample_test)

        # Ensure that the calculations are correct
        np.testing.assert_allclose(kernel_matrix, params_solution[1], rtol=1e-7, atol=1e-7)

    @data(TrainableFidelityQuantumKernel, TrainableFidelityStatevectorKernel)
    def test_incomplete_binding(self, trainable_kernel_type):
        """Test if an exception is raised when not all training parameter are bound."""
        # assign all parameters except the last one
        training_params = {
            self.training_parameters[i]: 0 for i in range(len(self.training_parameters) - 1)
        }

        kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

        kernel.assign_training_parameters(training_params)
        with self.assertRaises(QiskitMachineLearningError):
            kernel.evaluate(self.sample_train)

    @data(TrainableFidelityQuantumKernel, TrainableFidelityStatevectorKernel)
    def test_properties(self, trainable_kernel_type):
        """Test properties of the trainable quantum kernel."""
        kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )
        self.assertEqual(len(self.training_parameters), kernel.num_training_parameters)
        self.assertEqual(self.num_features, kernel.num_features)


    @data(TrainableFidelityQuantumKernel, TrainableFidelityStatevectorKernel)
    def test_default_feature_map(self, trainable_kernel_type):
        """Default feature map was removed; constructing without one should error."""
        with self.subTest("Do not pass feature map at all"):
            with self.assertRaises(QiskitMachineLearningError):
                _ = trainable_kernel_type()

        with self.subTest("Pass feature map with value None"):
            with self.assertRaises(QiskitMachineLearningError):
                _ = trainable_kernel_type(feature_map=None)


if __name__ == "__main__":
    unittest.main()
