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
import qiskit
from ddt import data, ddt
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.transpiler import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from qiskit.utils import QuantumInstance, algorithm_globals, optionals
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import Fidelity
from sklearn.svm import SVC

from qiskit_machine_learning.primitives.kernels import (
    QuantumKernel,
    PseudoKernel,
    TrainableQuantumKernel,
)


@ddt
class TestQuantumKernelClassify(QiskitMachineLearningTestCase):
    """Test QuantumKernel primitives for Classification using SKLearn"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.asarray([0, 1])

        self.sampler_factory = functools.partial(Sampler)

    def test_callable(self):
        """Test callable kernel in sklearn"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as kernel:
            svc = SVC(kernel=kernel.evaluate)
            svc.fit(self.sample_train, self.label_train)
            score = svc.score(self.sample_test, self.label_test)

            self.assertEqual(score, 0.5)

    def test_precomputed(self):
        """Test precomputed kernel in sklearn"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as kernel:
            kernel_train = kernel.evaluate(x_vec=self.sample_train)
            kernel_test = kernel.evaluate(x_vec=self.sample_test, y_vec=self.sample_train)

            svc = SVC(kernel="precomputed")
            svc.fit(kernel_train, self.label_train)
            score = svc.score(kernel_test, self.label_test)

            self.assertEqual(score, 0.5)


class TestQuantumKernelEvaluate(QiskitMachineLearningTestCase):
    """Test QuantumKernel primitives Evaluate Method"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.sampler_factory = functools.partial(Sampler)
        self.fidelity_factory = functools.partial(Fidelity)

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

        self.sample_train = np.asarray(
            [
                [2.95309709, 2.51327412],
                [3.14159265, 4.08407045],
                [4.08407045, 2.26194671],
                [4.46106157, 2.38761042],
            ]
        )

        self.sample_test = np.asarray([[3.83274304, 2.45044227], [3.89557489, 0.31415927]])

        self.sample_feature_dim = np.asarray([[1, 2, 3], [4, 5, 6]])
        self.sample_more_dim = np.asarray([[[0, 0], [1, 1]]])

        self.ref_kernel_train = np.array(
            [
                [1.00000000, 0.85342280, 0.12267747, 0.36334379],
                [0.85342280, 1.00000000, 0.11529017, 0.45246347],
                [0.12267747, 0.11529017, 1.00000000, 0.67137258],
                [0.36334379, 0.45246347, 0.67137258, 1.00000000],
            ]
        )

        self.ref_kernel_test = np.array(
            [
                [0.14439530, 0.33041779],
                [0.18170069, 0.37663733],
                [0.47479649, 0.02115561],
                [0.14691763, 0.16106199],
            ]
        )

    def test_symmetric(self):
        """Test symmetric matrix evaluation"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as qkclass:
            kernel = qkclass.evaluate(x_vec=self.sample_train)

            np.testing.assert_allclose(kernel, self.ref_kernel_train, rtol=1e-4)

    def test_unsymmetric(self):
        """Test unsymmetric matrix evaluation"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as qkclass:
            kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)

            np.testing.assert_allclose(kernel, self.ref_kernel_test, rtol=1e-4)

    def test_x_one_dim(self):
        """Test one x_vec dimension"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as qkclass:
            kernel = qkclass.evaluate(x_vec=self.sample_train[0])

            np.testing.assert_allclose(kernel, np.array([[1.0]]), rtol=1e-4)

    def test_y_one_dim(self):
        """Test one y_vec dimension"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as qkclass:

            kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test[0])

            np.testing.assert_allclose(
                kernel, self.ref_kernel_test[:, 0].reshape((-1, 1)), rtol=1e-4
            )

    def test_xy_one_dim(self):
        """Test one y_vec dimension"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map
        ) as qkclass:
            kernel = qkclass.evaluate(x_vec=self.sample_train[0], y_vec=self.sample_test[0])

            np.testing.assert_allclose(kernel, self.ref_kernel_test[0, 0], rtol=1e-4)

    def test_custom_fidelity_string(self):
        """Test symmetric matrix evaluation"""
        fidelity = "zero_prob"
        with QuantumKernel(
            sampler_factory=self.sampler_factory, feature_map=self.feature_map, fidelity=fidelity
        ) as qkclass:
            kernel = qkclass.evaluate(x_vec=self.sample_train)

            np.testing.assert_allclose(kernel, self.ref_kernel_train, rtol=1e-4)

    def test_custom_fidelity_factory(self):
        """Test symmetric matrix evaluation"""
        with QuantumKernel(
            sampler_factory=self.sampler_factory,
            feature_map=self.feature_map,
            fidelity=self.fidelity_factory,
        ) as qkclass:
            kernel = qkclass.evaluate(x_vec=self.sample_train)

            np.testing.assert_allclose(kernel, self.ref_kernel_train, rtol=1e-4)


# class TestQuantumKernelTrainingParameters(QiskitMachineLearningTestCase):
#     """Test QuantumKernel training parameter support"""

#     def setUp(self):
#         super().setUp()

#         # Create an arbitrary 3-qubit feature map circuit
#         circ1 = ZZFeatureMap(3)
#         circ2 = ZZFeatureMap(3)
#         training_params = circ2.parameters
#         for i, training_param in enumerate(training_params):
#             training_param._name = f"θ[{i}]"

#         self.feature_map = circ1.compose(circ2).compose(circ1)
#         self.training_parameters = training_params

#     def test_training_parameters(self):
#         """Test assigning/re-assigning user parameters"""

#         with self.subTest("check basic instantiation"):
#             # Ensure we can instantiate a QuantumKernel with user parameters
#             qkclass = QuantumKernel(
#                 feature_map=self.feature_map, training_parameters=self.training_parameters
#             )
#             self.assertEqual(qkclass.training_parameters, self.training_parameters)

#         with self.subTest("test invalid parameter assignment"):
#             # Instantiate a QuantumKernel
#             qkclass = QuantumKernel(
#                 feature_map=self.feature_map, training_parameters=self.training_parameters
#             )

#             # Try to set the user parameters using an incorrect number of values
#             training_param_values = [2.0, 4.0, 6.0, 8.0]
#             with self.assertRaises(ValueError):
#                 qkclass.assign_training_parameters(training_param_values)

#             self.assertEqual(qkclass.get_unbound_training_parameters(), qkclass.training_parameters)

#         with self.subTest("test parameter assignment"):
#             # Assign params to some new values, and also test the bind_training_parameters interface
#             param_binds = {
#                 self.training_parameters[0]: 0.1,
#                 self.training_parameters[1]: 0.2,
#                 self.training_parameters[2]: 0.3,
#             }
#             qkclass.bind_training_parameters(param_binds)

#             # Ensure the values are properly bound
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.values()), list(param_binds.values())
#             )
#             self.assertEqual(qkclass.get_unbound_training_parameters(), [])
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )

#         with self.subTest("test partial parameter assignment"):
#             # Assign params to some new values, and also test the bind_training_parameters interface
#             param_binds = {self.training_parameters[0]: 0.5, self.training_parameters[1]: 0.4}
#             qkclass.bind_training_parameters(param_binds)

#             # Ensure values were properly bound and param 2 was unchanged
#             self.assertEqual(list(qkclass.training_parameter_binds.values()), [0.5, 0.4, 0.3])
#             self.assertEqual(qkclass.get_unbound_training_parameters(), [])
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )

#         with self.subTest("test unassign and assign to parameter expression"):
#             param_binds = {
#                 self.training_parameters[0]: self.training_parameters[0],
#                 self.training_parameters[1]: self.training_parameters[0]
#                 + self.training_parameters[2],
#                 self.training_parameters[2]: self.training_parameters[2],
#             }
#             qkclass.assign_training_parameters(param_binds)

#             # Ensure quantum kernel forgets unused param 1 and unbinds param 0 and 2
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 [self.training_parameters[0], self.training_parameters[2]],
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 list(qkclass.training_parameter_binds.values()),
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )

#         with self.subTest("test immediate reassignment to parameter expression"):
#             # Create a new quantum kernel
#             qkclass = QuantumKernel(
#                 feature_map=self.feature_map, training_parameters=self.training_parameters
#             )
#             # Create a new parameter
#             new_param = Parameter("0[n]")

#             # Create partial param binds with immediate reassignments to param expressions
#             param_binds = {
#                 self.training_parameters[0]: new_param,
#                 self.training_parameters[1]: self.training_parameters[0]
#                 + self.training_parameters[2],
#             }
#             qkclass.assign_training_parameters(param_binds)

#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 [new_param, self.training_parameters[0], self.training_parameters[2]],
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 list(qkclass.training_parameter_binds.values()),
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )

#         with self.subTest("test bringing back old parameters"):
#             param_binds = {
#                 new_param: self.training_parameters[1] * self.training_parameters[0]
#                 + self.training_parameters[2]
#             }
#             qkclass.assign_training_parameters(param_binds)
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 [
#                     self.training_parameters[0],
#                     self.training_parameters[1],
#                     self.training_parameters[2],
#                 ],
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 list(qkclass.training_parameter_binds.values()),
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )

#         with self.subTest("test assign with immediate reassign"):
#             # Create a new quantum kernel
#             qkclass = QuantumKernel(
#                 feature_map=self.feature_map, training_parameters=self.training_parameters
#             )
#             param_binds = {
#                 self.training_parameters[0]: 0.9,
#                 self.training_parameters[1]: self.training_parameters[0],
#                 self.training_parameters[2]: self.training_parameters[1],
#             }
#             qkclass.assign_training_parameters(param_binds)
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()),
#                 [self.training_parameters[0], self.training_parameters[1]],
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.values()), [0.9, self.training_parameters[1]]
#             )
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )

#         with self.subTest("test unordered assigns"):
#             # Create a new quantum kernel
#             qkclass = QuantumKernel(
#                 feature_map=self.feature_map, training_parameters=self.training_parameters
#             )
#             param_binds = {
#                 self.training_parameters[2]: self.training_parameters[1],
#                 self.training_parameters[1]: self.training_parameters[0],
#                 self.training_parameters[0]: 1.7,
#             }
#             qkclass.assign_training_parameters(param_binds)
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), [self.training_parameters[0]]
#             )
#             self.assertEqual(list(qkclass.training_parameter_binds.values()), [1.7])
#             self.assertEqual(
#                 list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
#             )


# class TestQuantumKernelBatching(QiskitMachineLearningTestCase):
#     """Test QuantumKernel circuit batching

#     Checks batching with both statevector simulator and QASM simulator.
#     Checks that the actual number of circuits being passed
#     to execute does not exceed the batch_size requested by the Quantum Kernel.
#     Checks that the sum of the batch sizes matches the total number of expected
#     circuits.
#     """

#     def count_circuits(self, func):
#         """Wrapper to record the number of circuits passed to QuantumInstance.execute.

#         Args:
#             func (Callable): execute function to be wrapped

#         Returns:
#             Callable: function wrapper
#         """

#         @functools.wraps(func)
#         def wrapper(*args, **kwds):
#             self.circuit_counts.append(len(args[0]))
#             return func(*args, **kwds)

#         return wrapper

#     def setUp(self):
#         super().setUp()

#         algorithm_globals.random_seed = 10598

#         self.shots = 12000
#         self.batch_size = 3
#         self.circuit_counts = []

#         self.statevector_simulator = QuantumInstance(
#             BasicAer.get_backend("statevector_simulator"),
#             shots=1,
#             seed_simulator=algorithm_globals.random_seed,
#             seed_transpiler=algorithm_globals.random_seed,
#         )

#         # monkey patch the statevector simulator
#         self.statevector_simulator.execute = self.count_circuits(self.statevector_simulator.execute)

#         self.qasm_simulator = QuantumInstance(
#             BasicAer.get_backend("qasm_simulator"),
#             shots=self.shots,
#             seed_simulator=algorithm_globals.random_seed,
#             seed_transpiler=algorithm_globals.random_seed,
#         )

#         # monkey patch the qasm simulator
#         self.qasm_simulator.execute = self.count_circuits(self.qasm_simulator.execute)

#         self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

#         # data generated using
#         # sample_train, label_train, _, _ = ad_hoc_data(training_size=4, test_size=2, n=2, gap=0.3)
#         self.sample_train = np.asarray(
#             [
#                 [5.90619419, 1.25663706],
#                 [2.32477856, 0.9424778],
#                 [4.52389342, 5.0893801],
#                 [3.58141563, 0.9424778],
#                 [0.31415927, 3.45575192],
#                 [4.83805269, 3.70707933],
#                 [5.65486678, 6.09468975],
#                 [5.46637122, 4.52389342],
#             ]
#         )
#         self.label_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

#     def test_statevector_batching(self):
#         """Test batching when using the statevector simulator"""

#         self.circuit_counts = []

#         kernel = QuantumKernel(
#             feature_map=self.feature_map,
#             batch_size=self.batch_size,
#             quantum_instance=self.statevector_simulator,
#         )

#         svc = SVC(kernel=kernel.evaluate)

#         svc.fit(self.sample_train, self.label_train)

#         for circuit_count in self.circuit_counts:
#             self.assertLessEqual(circuit_count, self.batch_size)

#         self.assertEqual(sum(self.circuit_counts), len(self.sample_train))

#     def test_qasm_batching(self):
#         """Test batching when using the QASM simulator"""

#         self.circuit_counts = []

#         kernel = QuantumKernel(
#             feature_map=self.feature_map,
#             batch_size=self.batch_size,
#             quantum_instance=self.qasm_simulator,
#         )

#         svc = SVC(kernel=kernel.evaluate)
#         svc.fit(self.sample_train, self.label_train)

#         for circuit_count in self.circuit_counts:
#             self.assertLessEqual(circuit_count, self.batch_size)

#         num_train = len(self.sample_train)
#         num_circuits = num_train * (num_train - 1) / 2

#         self.assertEqual(sum(self.circuit_counts), num_circuits)


if __name__ == "__main__":
    unittest.main()
