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

# NOTE WELL that this file of tests should disappear when we remove the
# deprecated ``user_param`` argument/property/methods in quantum_kernel.py.
# These tests exist ONLY to ensure that these deprecated objects correctly.

""" Test QuantumKernel """

import unittest
import warnings

from test import QiskitMachineLearningTestCase

from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.kernels import QuantumKernel


class TestQuantumKernelTrainingParameters(QiskitMachineLearningTestCase):
    """Test QuantumKernel training parameter support"""

    def setUp(self):
        super().setUp()

        # Create an arbitrary 3-qubit feature map circuit
        circ1 = ZZFeatureMap(3)
        circ2 = ZZFeatureMap(3, parameter_prefix="θ")
        user_params = circ2.parameters

        self.feature_map = circ1.compose(circ2).compose(circ1)
        self.user_parameters = user_params

    def test_positional_user_parameters(self):
        """Test assigning user parameters with positional argument"""

        with self.subTest("check positional argument"):
            # Ensure we can instantiate a QuantumKernel with positional user parameters
            qkclass = QuantumKernel(
                self.feature_map,
                True,
                900,
                None,
                self.user_parameters,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.assertEqual(qkclass.user_parameters, self.user_parameters)

    def test_user_parameters(self):
        """Test assigning/re-assigning user parameters"""

        with self.subTest("check basic instantiation"):
            # Ensure we can instantiate a QuantumKernel with user parameters
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.user_parameters
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.assertEqual(qkclass.user_parameters, self.user_parameters)

        with self.subTest("test invalid parameter assignment"):
            # Instantiate a QuantumKernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.user_parameters
            )

            # Try to set the user parameters using an incorrect number of values
            user_param_values = [2.0, 4.0, 6.0, 8.0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                with self.assertRaises(ValueError):
                    qkclass.assign_user_parameters(user_param_values)

                    self.assertEqual(qkclass.get_unbound_user_parameters(), qkclass.user_parameters)

        with self.subTest("test parameter assignment"):
            # Assign params to some new values, and also test the bind_user_parameters interface
            param_binds = {
                self.user_parameters[0]: 0.1,
                self.user_parameters[1]: 0.2,
                self.user_parameters[2]: 0.3,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                qkclass.bind_user_parameters(param_binds)

                # Ensure the values are properly bound
                self.assertEqual(
                    list(qkclass.user_param_binds.values()), list(param_binds.values())
                )
                self.assertEqual(qkclass.get_unbound_user_parameters(), [])
                self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)

        with self.subTest("test partial parameter assignment"):
            # Assign params to some new values, and also test the bind_user_parameters interface
            param_binds = {self.user_parameters[0]: 0.5, self.user_parameters[1]: 0.4}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                qkclass.bind_user_parameters(param_binds)

            # Ensure values were properly bound and param 2 was unchanged
            self.assertEqual(list(qkclass.user_param_binds.values()), [0.5, 0.4, 0.3])
            self.assertEqual(qkclass.get_unbound_user_parameters(), [])
            self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)

        with self.subTest("test unassign and assign to parameter expression"):
            param_binds = {
                self.user_parameters[0]: self.user_parameters[0],
                self.user_parameters[1]: self.user_parameters[0] + self.user_parameters[2],
                self.user_parameters[2]: self.user_parameters[2],
            }
            qkclass.assign_user_parameters(param_binds)

            # Ensure quantum kernel forgets unused param 1 and unbinds param 0 and 2
            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                [self.user_parameters[0], self.user_parameters[2]],
            )
            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                list(qkclass.user_param_binds.values()),
            )
            self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)

        with self.subTest("test immediate reassignment to parameter expression"):
            # Create a new quantum kernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.user_parameters
            )
            # Create a new parameter
            new_param = Parameter("0[n]")

            # Create partial param binds with immediate reassignments to param expressions
            param_binds = {
                self.user_parameters[0]: new_param,
                self.user_parameters[1]: self.user_parameters[0] + self.user_parameters[2],
            }
            qkclass.assign_user_parameters(param_binds)

            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                [new_param, self.user_parameters[0], self.user_parameters[2]],
            )
            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                list(qkclass.user_param_binds.values()),
            )
            self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)

        with self.subTest("test bringing back old parameters"):
            param_binds = {
                new_param: self.user_parameters[1] * self.user_parameters[0]
                + self.user_parameters[2]
            }
            qkclass.assign_user_parameters(param_binds)
            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                [
                    self.user_parameters[0],
                    self.user_parameters[1],
                    self.user_parameters[2],
                ],
            )
            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                list(qkclass.user_param_binds.values()),
            )
            self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)

        with self.subTest("test assign with immediate reassign"):
            # Create a new quantum kernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.user_parameters
            )
            param_binds = {
                self.user_parameters[0]: 0.9,
                self.user_parameters[1]: self.user_parameters[0],
                self.user_parameters[2]: self.user_parameters[1],
            }
            qkclass.assign_user_parameters(param_binds)
            self.assertEqual(
                list(qkclass.user_param_binds.keys()),
                [self.user_parameters[0], self.user_parameters[1]],
            )
            self.assertEqual(
                list(qkclass.user_param_binds.values()), [0.9, self.user_parameters[1]]
            )
            self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)

        with self.subTest("test unordered assigns"):
            # Create a new quantum kernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.user_parameters
            )
            param_binds = {
                self.user_parameters[2]: self.user_parameters[1],
                self.user_parameters[1]: self.user_parameters[0],
                self.user_parameters[0]: 1.7,
            }
            qkclass.assign_user_parameters(param_binds)
            self.assertEqual(list(qkclass.user_param_binds.keys()), [self.user_parameters[0]])
            self.assertEqual(list(qkclass.user_param_binds.values()), [1.7])
            self.assertEqual(list(qkclass.user_param_binds.keys()), qkclass.user_parameters)


if __name__ == "__main__":
    unittest.main()
