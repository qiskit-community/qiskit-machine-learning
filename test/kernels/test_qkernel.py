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

""" Test QuantumKernel """

import functools
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
import qiskit
from ddt import data, ddt, idata, unpack
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.transpiler import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from qiskit.utils import QuantumInstance, algorithm_globals, optionals
from sklearn.svm import SVC

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.kernels import QuantumKernel


@ddt
class TestQuantumKernelClassify(QiskitMachineLearningTestCase):
    """Test QuantumKernel for Classification using SKLearn"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

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

    def test_callable(self):
        """Test callable kernel in sklearn"""
        kernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        score = svc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_precomputed(self):
        """Test precomputed kernel in sklearn"""
        kernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        kernel_train = kernel.evaluate(x_vec=self.sample_train)
        kernel_test = kernel.evaluate(x_vec=self.sample_test, y_vec=self.sample_train)

        svc = SVC(kernel="precomputed")
        svc.fit(kernel_train, self.label_train)
        score = svc.score(kernel_test, self.label_test)

        self.assertEqual(score, 0.5)

    # pylint: disable=no-member
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @data(qiskit.providers.aer.AerSimulator(), BasicAer.get_backend("statevector_simulator"))
    def test_custom_pass_manager(self, backend):
        """Test quantum kernel with a custom pass manager."""

        quantum_instance = QuantumInstance(
            backend,
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
            pass_manager=level_1_pass_manager(PassManagerConfig(basis_gates=["u3", "cx"])),
            bound_pass_manager=level_1_pass_manager(PassManagerConfig(basis_gates=["u3", "cx"])),
        )

        kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=quantum_instance)

        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        score = svc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)


class TestQuantumKernelEvaluate(QiskitMachineLearningTestCase):
    """Test QuantumKernel Evaluate Method"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598
        self.shots = 12000

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=self.shots,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        self.qasm_sample = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=10,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

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

        self.ref_kernel_train = {
            "one_dim": np.array([[1.0]]),
            "qasm": np.array(
                [
                    [1.000000, 0.856583, 0.120417, 0.358833],
                    [0.856583, 1.000000, 0.113167, 0.449250],
                    [0.120417, 0.113167, 1.000000, 0.671500],
                    [0.358833, 0.449250, 0.671500, 1.000000],
                ]
            ),
            "statevector": np.array(
                [
                    [1.00000000, 0.85342280, 0.12267747, 0.36334379],
                    [0.85342280, 1.00000000, 0.11529017, 0.45246347],
                    [0.12267747, 0.11529017, 1.00000000, 0.67137258],
                    [0.36334379, 0.45246347, 0.67137258, 1.00000000],
                ]
            ),
            "qasm_sample": np.array(
                [
                    [1.0, 0.9, 0.1, 0.4],
                    [0.9, 1.0, 0.1, 0.6],
                    [0.1, 0.1, 1.0, 0.9],
                    [0.4, 0.6, 0.9, 1.0],
                ]
            ),
            "qasm_sample_psd": np.array(
                [
                    [1.004036, 0.891664, 0.091883, 0.410062],
                    [0.891664, 1.017215, 0.116764, 0.579220],
                    [0.091883, 0.116764, 1.016324, 0.879765],
                    [0.410062, 0.579220, 0.879765, 1.025083],
                ]
            ),
        }

        self.ref_kernel_test = {
            "one_y_dim": np.array([[0.144395], [0.181701], [0.474796], [0.146918]]),
            "one_xy_dim": np.array([[0.144395]]),
            "qasm": np.array(
                [
                    [0.140667, 0.327833],
                    [0.177750, 0.371750],
                    [0.467833, 0.018417],
                    [0.143333, 0.156750],
                ]
            ),
            "statevector": np.array(
                [
                    [0.14439530, 0.33041779],
                    [0.18170069, 0.37663733],
                    [0.47479649, 0.02115561],
                    [0.14691763, 0.16106199],
                ]
            ),
        }

    def test_qasm_symmetric(self):
        """Test symmetric matrix evaluation using qasm simulator"""
        qkclass = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.qasm_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train["qasm"], rtol=1e-4)

    def test_qasm_unsymmetric(self):
        """Test unsymmetric matrix evaluation using qasm simulator"""
        qkclass = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.qasm_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)

        np.testing.assert_allclose(kernel, self.ref_kernel_test["qasm"], rtol=1e-4)

    def test_sv_symmetric(self):
        """Test symmetric matrix evaluation using state vector simulator"""
        qkclass = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train["statevector"], rtol=1e-4)

    def test_sv_unsymmetric(self):
        """Test unsymmetric matrix evaluation using state vector simulator"""
        qkclass = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)

        np.testing.assert_allclose(kernel, self.ref_kernel_test["statevector"], rtol=1e-4)

    def test_qasm_nopsd(self):
        """Test symmetric matrix qasm sample no positive semi-definite enforcement"""
        qkclass = QuantumKernel(
            feature_map=self.feature_map,
            quantum_instance=self.qasm_sample,
            enforce_psd=False,
        )

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train["qasm_sample"], rtol=1e-4)

    def test_qasm_psd(self):
        """Test symmetric matrix positive semi-definite enforcement qasm sample"""
        qkclass = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.qasm_sample)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train["qasm_sample_psd"], rtol=1e-4)

    def test_x_one_dim(self):
        """Test one x_vec dimension"""
        qkclass = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        kernel = qkclass.evaluate(x_vec=self.sample_train[0])

        np.testing.assert_allclose(kernel, self.ref_kernel_train["one_dim"], rtol=1e-4)

    def test_y_one_dim(self):
        """Test one y_vec dimension"""
        qkclass = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test[0])

        np.testing.assert_allclose(kernel, self.ref_kernel_test["one_y_dim"], rtol=1e-4)

    def test_xy_one_dim(self):
        """Test one y_vec dimension"""
        qkclass = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        kernel = qkclass.evaluate(x_vec=self.sample_train[0], y_vec=self.sample_test[0])

        np.testing.assert_allclose(kernel, self.ref_kernel_test["one_xy_dim"], rtol=1e-4)

    def test_no_backend(self):
        """Test no backend provided"""
        qkclass = QuantumKernel(feature_map=self.feature_map)

        with self.assertRaises(QiskitMachineLearningError):
            _ = qkclass.evaluate(x_vec=self.sample_train)

    def test_x_more_dim(self):
        """Test incorrect x_vec dimension"""
        qkclass = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.qasm_simulator)

        with self.assertRaises(ValueError):
            _ = qkclass.evaluate(x_vec=self.sample_more_dim)

    def test_y_more_dim(self):
        """Test incorrect y_vec dimension"""
        qkclass = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.qasm_simulator)

        with self.assertRaises(ValueError):
            _ = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_more_dim)

    def test_y_feature_dim(self):
        """Test incorrect y_vec feature dimension"""
        qkclass = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.qasm_simulator)

        with self.assertRaises(ValueError):
            _ = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_feature_dim)

    def test_adjust_feature_map(self):
        """Test adjust feature map"""
        qkclass = QuantumKernel(
            feature_map=ZZFeatureMap(feature_dimension=3), quantum_instance=self.qasm_simulator
        )
        _ = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)

    def test_fail_adjust_feature_map(self):
        """Test feature map adjustment failed"""
        feature_map = QuantumCircuit(3)
        qkclass = QuantumKernel(feature_map=feature_map, quantum_instance=self.qasm_simulator)
        with self.assertRaises(ValueError):
            _ = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)


class TestQuantumKernelConstructCircuit(QiskitMachineLearningTestCase):
    """Test QuantumKernel ConstructCircuit Method"""

    def setUp(self):
        super().setUp()

        self.x = [1, 1]
        self.y = [2, 2]
        self.z = [3]

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=1)

    def _check_circuit(self, qc: QuantumCircuit, check_measurements: bool, check_inverse: bool):
        self.assertEqual(qc.num_qubits, self.feature_map.num_qubits)

        # check that there are two feature maps
        self.assertTrue(qc.data[0][0].name.startswith(self.feature_map.name))
        self.assertTrue(qc.data[1][0].name.startswith(self.feature_map.name))

        # check that there are measurement operations in the circuit
        if check_measurements:
            original_depth = qc.depth()
            qc.remove_final_measurements()
            self.assertNotEqual(original_depth, qc.depth())

        # check that there are two feature maps: plain and plain dagger (inverse)
        if check_inverse:
            self.assertEqual(qc.data[1][0].definition, qc.data[0][0].definition.inverse())

    def test_innerproduct(self):
        """Test inner product"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, self.y)
        self._check_circuit(qc, check_measurements=True, check_inverse=False)

    def test_selfinnerproduct(self):
        """Test self inner product"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x)
        self._check_circuit(qc, check_measurements=True, check_inverse=True)

    def test_innerproduct_nomeasurement(self):
        """Test inner product no measurement"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, self.y, measurement=False)
        self._check_circuit(qc, check_measurements=False, check_inverse=False)

    def test_selfinnerprodect_nomeasurement(self):
        """Test self inner product no measurement"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, measurement=False)
        self._check_circuit(qc, check_measurements=False, check_inverse=True)

    def test_statevector(self):
        """Test state vector simulator"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, is_statevector_sim=True)

        self.assertEqual(qc.num_qubits, self.feature_map.num_qubits)

        # check that there's a feature map in the circuit
        self.assertTrue(qc.data[0][0].name.startswith(self.feature_map.name))

    def test_xdim(self):
        """Test incorrect x dimension"""
        qkclass = QuantumKernel(feature_map=self.feature_map)

        with self.assertRaises(ValueError):
            _ = qkclass.construct_circuit(self.z)

    def test_ydim(self):
        """Test incorrect y dimension"""
        qkclass = QuantumKernel(feature_map=self.feature_map)

        with self.assertRaises(ValueError):
            _ = qkclass.construct_circuit(self.x, self.z)


class TestQuantumKernelTrainingParameters(QiskitMachineLearningTestCase):
    """Test QuantumKernel training parameter support"""

    def setUp(self):
        super().setUp()

        # Create an arbitrary 3-qubit feature map circuit
        circ1 = ZZFeatureMap(3)
        circ2 = ZZFeatureMap(3, parameter_prefix="θ")
        training_params = circ2.parameters

        self.feature_map = circ1.compose(circ2).compose(circ1)
        self.training_parameters = training_params

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

    def test_training_parameters(self):
        """Test assigning/re-assigning user parameters"""

        with self.subTest("check basic instantiation"):
            # Ensure we can instantiate a QuantumKernel with user parameters
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.training_parameters
            )
            self.assertEqual(qkclass.training_parameters, self.training_parameters)

        with self.subTest("test invalid parameter assignment"):
            # Instantiate a QuantumKernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.training_parameters
            )

            # Try to set the user parameters using an incorrect number of values
            training_param_values = [2.0, 4.0, 6.0, 8.0]
            with self.assertRaises(ValueError):
                qkclass.assign_training_parameters(training_param_values)

            self.assertEqual(qkclass.get_unbound_training_parameters(), qkclass.training_parameters)

        with self.subTest("test parameter assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = {
                self.training_parameters[0]: 0.1,
                self.training_parameters[1]: 0.2,
                self.training_parameters[2]: 0.3,
            }
            qkclass.bind_training_parameters(param_binds)

            # Ensure the values are properly bound
            self.assertEqual(
                list(qkclass.training_parameter_binds.values()), list(param_binds.values())
            )
            self.assertEqual(qkclass.get_unbound_training_parameters(), [])
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

        with self.subTest("test partial parameter assignment"):
            # Assign params to some new values, and also test the bind_training_parameters interface
            param_binds = {self.training_parameters[0]: 0.5, self.training_parameters[1]: 0.4}
            qkclass.bind_training_parameters(param_binds)

            # Ensure values were properly bound and param 2 was unchanged
            self.assertEqual(list(qkclass.training_parameter_binds.values()), [0.5, 0.4, 0.3])
            self.assertEqual(qkclass.get_unbound_training_parameters(), [])
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

        with self.subTest("test unassign and assign to parameter expression"):
            param_binds = {
                self.training_parameters[0]: self.training_parameters[0],
                self.training_parameters[1]: self.training_parameters[0]
                + self.training_parameters[2],
                self.training_parameters[2]: self.training_parameters[2],
            }
            qkclass.assign_training_parameters(param_binds)

            # Ensure quantum kernel forgets unused param 1 and unbinds param 0 and 2
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                [self.training_parameters[0], self.training_parameters[2]],
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                list(qkclass.training_parameter_binds.values()),
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

        with self.subTest("test immediate reassignment to parameter expression"):
            # Create a new quantum kernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.training_parameters
            )
            # Create a new parameter
            new_param = Parameter("0[n]")

            # Create partial param binds with immediate reassignments to param expressions
            param_binds = {
                self.training_parameters[0]: new_param,
                self.training_parameters[1]: self.training_parameters[0]
                + self.training_parameters[2],
            }
            qkclass.assign_training_parameters(param_binds)

            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                [new_param, self.training_parameters[0], self.training_parameters[2]],
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                list(qkclass.training_parameter_binds.values()),
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

        with self.subTest("test bringing back old parameters"):
            param_binds = {
                new_param: self.training_parameters[1] * self.training_parameters[0]
                + self.training_parameters[2]
            }
            qkclass.assign_training_parameters(param_binds)
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                [
                    self.training_parameters[0],
                    self.training_parameters[1],
                    self.training_parameters[2],
                ],
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                list(qkclass.training_parameter_binds.values()),
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

        with self.subTest("test assign with immediate reassign"):
            # Create a new quantum kernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.training_parameters
            )
            param_binds = {
                self.training_parameters[0]: 0.9,
                self.training_parameters[1]: self.training_parameters[0],
                self.training_parameters[2]: self.training_parameters[1],
            }
            qkclass.assign_training_parameters(param_binds)
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()),
                [self.training_parameters[0], self.training_parameters[1]],
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.values()), [0.9, self.training_parameters[1]]
            )
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

        with self.subTest("test unordered assigns"):
            # Create a new quantum kernel
            qkclass = QuantumKernel(
                feature_map=self.feature_map, training_parameters=self.training_parameters
            )
            param_binds = {
                self.training_parameters[2]: self.training_parameters[1],
                self.training_parameters[1]: self.training_parameters[0],
                self.training_parameters[0]: 1.7,
            }
            qkclass.assign_training_parameters(param_binds)
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), [self.training_parameters[0]]
            )
            self.assertEqual(list(qkclass.training_parameter_binds.values()), [1.7])
            self.assertEqual(
                list(qkclass.training_parameter_binds.keys()), qkclass.training_parameters
            )

    def test_unbound_parameters(self):
        """Test unbound parameters."""
        qc = QuantumCircuit(2)
        parameters = [Parameter("x")]
        qc.ry(parameters[0], 0)

        qkernel = QuantumKernel(
            qc, training_parameters=parameters, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC(quantum_kernel=qkernel)
        self.assertRaises(ValueError, qsvc.fit, self.sample_train, self.label_train)


class TestQuantumKernelBatching(QiskitMachineLearningTestCase):
    """Test QuantumKernel circuit batching

    Checks batching with both statevector simulator and QASM simulator.
    Checks that the actual number of circuits being passed
    to execute does not exceed the batch_size requested by the Quantum Kernel.
    Checks that the sum of the batch sizes matches the total number of expected
    circuits.
    """

    def count_circuits(self, func):
        """Wrapper to record the number of circuits passed to QuantumInstance.execute.

        Args:
            func (Callable): execute function to be wrapped

        Returns:
            Callable: function wrapper
        """

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.circuit_counts.append(len(args[0]))
            return func(*args, **kwds)

        return wrapper

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.shots = 12000
        self.batch_size = 3
        self.circuit_counts = []

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # monkey patch the statevector simulator
        self.statevector_simulator.execute = self.count_circuits(self.statevector_simulator.execute)

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=self.shots,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # monkey patch the qasm simulator
        self.qasm_simulator.execute = self.count_circuits(self.qasm_simulator.execute)

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

        # data generated using
        # sample_train, label_train, _, _ = ad_hoc_data(training_size=4, test_size=2, n=2, gap=0.3)
        self.sample_train = np.asarray(
            [
                [5.90619419, 1.25663706],
                [2.32477856, 0.9424778],
                [4.52389342, 5.0893801],
                [3.58141563, 0.9424778],
                [0.31415927, 3.45575192],
                [4.83805269, 3.70707933],
                [5.65486678, 6.09468975],
                [5.46637122, 4.52389342],
            ]
        )
        self.label_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    def test_statevector_batching(self):
        """Test batching when using the statevector simulator"""

        self.circuit_counts = []

        kernel = QuantumKernel(
            feature_map=self.feature_map,
            batch_size=self.batch_size,
            quantum_instance=self.statevector_simulator,
        )

        svc = SVC(kernel=kernel.evaluate)

        svc.fit(self.sample_train, self.label_train)

        for circuit_count in self.circuit_counts:
            self.assertLessEqual(circuit_count, self.batch_size)

        self.assertEqual(sum(self.circuit_counts), len(self.sample_train))

    def test_qasm_batching(self):
        """Test batching when using the QASM simulator"""

        self.circuit_counts = []

        kernel = QuantumKernel(
            feature_map=self.feature_map,
            batch_size=self.batch_size,
            quantum_instance=self.qasm_simulator,
        )

        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)

        for circuit_count in self.circuit_counts:
            self.assertLessEqual(circuit_count, self.batch_size)

        num_train = len(self.sample_train)
        num_circuits = num_train * (num_train - 1) / 2

        self.assertEqual(sum(self.circuit_counts), num_circuits)


@ddt
class TestQuantumKernelEvaluateDuplicates(QiskitMachineLearningTestCase):
    """Test QuantumKernel for duplicate evaluation."""

    def count_circuits(self, func):
        """Wrapper to record the number of circuits passed to QuantumInstance.execute.

        Args:
            func (Callable): execute function to be wrapped

        Returns:
            Callable: function wrapper
        """

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.circuit_counts += len(args[0])
            return func(*args, **kwds)

        return wrapper

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598
        self.circuit_counts = 0

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # monkey patch the qasm simulator
        self.qasm_simulator.execute = self.count_circuits(self.qasm_simulator.execute)

        self.feature_map = ZFeatureMap(feature_dimension=2, reps=1)

        self.properties = {
            "no_dups": np.array([[1, 2], [2, 3], [3, 4]]),
            "dups": np.array([[1, 2], [1, 2], [3, 4]]),
            "y_vec": np.array([[0, 1], [1, 2]]),
        }

    @idata(
        [
            ("no_dups", "all", 6),
            ("no_dups", "off_diagonal", 3),
            ("no_dups", "none", 3),
            ("dups", "all", 6),
            ("dups", "off_diagonal", 3),
            ("dups", "none", 2),
        ]
    )
    @unpack
    def test_evaluate_duplicates(self, dataset_name, evaluate_duplicates, expected_num_circuits):
        """Tests symmetric quantum kernel evaluation with duplicate samples."""
        self.circuit_counts = 0
        qkernel = QuantumKernel(
            feature_map=self.feature_map,
            evaluate_duplicates=evaluate_duplicates,
            quantum_instance=self.qasm_simulator,
        )
        qkernel.evaluate(self.properties.get(dataset_name))
        self.assertEqual(self.circuit_counts, expected_num_circuits)

    @idata(
        [
            ("no_dups", "all", 6),
            ("no_dups", "off_diagonal", 6),
            ("no_dups", "none", 5),
        ]
    )
    @unpack
    def test_evaluate_duplicates_not_symmetric(
        self, dataset_name, evaluate_duplicates, expected_num_circuits
    ):
        """Tests non-symmetric quantum kernel evaluation with duplicate samples."""
        self.circuit_counts = 0
        qkernel = QuantumKernel(
            feature_map=self.feature_map,
            evaluate_duplicates=evaluate_duplicates,
            quantum_instance=self.qasm_simulator,
        )
        qkernel.evaluate(self.properties.get(dataset_name), self.properties.get("y_vec"))
        self.assertEqual(self.circuit_counts, expected_num_circuits)


if __name__ == "__main__":
    unittest.main()
