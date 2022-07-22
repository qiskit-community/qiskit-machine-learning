# # This code is part of Qiskit.
# #
# # (C) Copyright IBM 2021, 2022.
# #
# # This code is licensed under the Apache License, Version 2.0. You may
# # obtain a copy of this license in the LICENSE.txt file in the root directory
# # of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# #
# # Any modifications or derivative works of this code must retain this
# # copyright notice, and modified files need to carry a notice indicating
# # that they have been altered from the originals.

# """ Test QuantumKernel """

import functools
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import Fidelity
from sklearn.svm import SVC

from qiskit_machine_learning.primitives.kernels import TrainableQuantumKernel


@ddt
class TestPrimitivesTrainableQuantumKernelClassify(QiskitMachineLearningTestCase):
    """Test trainable QuantumKernel."""

    def setUp(self):
        super().setUp()

        # Create an arbitrary 3-qubit feature map circuit
        circ1 = ZZFeatureMap(3)
        circ2 = ZZFeatureMap(3, parameter_prefix="θ")
        self.feature_map = circ1.compose(circ2).compose(circ1)
        self.training_parameters = circ2.parameters
        self.sampler_factory = functools.partial(Sampler)

        self.kernel = TrainableQuantumKernel(
            sampler=self.sampler_factory,
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )

    def test_training_parameters(self):
        """Test assigning/re-assigning user parameters"""

        with self.subTest("check basic instantiation"):
            # Ensure we can instantiate a QuantumKernel with user parameters
            self.assertEqual(self.kernel._training_parameters, self.training_parameters)

        with self.subTest("test invalid parameter assignment"):
            # Try to set the user parameters using an incorrect number of values
            training_param_values = [2.0, 4.0, 6.0, 8.0]
            with self.assertRaises(ValueError):
                self.kernel.assign_training_parameters(training_param_values)

        with self.subTest("test parameter assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = {
                self.training_parameters[0]: 0.1,
                self.training_parameters[1]: 0.2,
                self.training_parameters[2]: 0.3,
            }
            self.kernel.assign_training_parameters(param_binds)

            # Ensure the values are properly bound
            np.testing.assert_array_equal(self.kernel.parameter_values, list(param_binds.values()))

        with self.subTest("test partial parameter assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = {self.training_parameters[0]: 0.5, self.training_parameters[1]: 0.4}
            self.kernel.assign_training_parameters(param_binds)

            # Ensure values were properly bound and param 2 was unchanged
            np.testing.assert_array_equal(self.kernel.parameter_values, [0.5, 0.4, 0.3])

        with self.subTest("test parameter list assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = [0.1, 0.7, 1.7]
            self.kernel.assign_training_parameters(param_binds)

            # Ensure the values are properly bound
            np.testing.assert_array_equal(self.kernel.parameter_values, param_binds)

        with self.subTest("test parameter array assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = np.array([0.1, 0.7, 1.7])
            self.kernel.assign_training_parameters(param_binds)

            # Ensure the values are properly bound
            np.testing.assert_array_equal(self.kernel.parameter_values, param_binds)


if __name__ == "__main__":
    unittest.main()
