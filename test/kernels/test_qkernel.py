# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QuantumKernel """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np

from sklearn.svm import SVC

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


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


class TestQuantumKernelConstructCircuit(QiskitMachineLearningTestCase):
    """Test QuantumKernel ConstructCircuit Method"""

    def setUp(self):
        super().setUp()

        self.x = [1, 1]
        self.y = [2, 2]
        self.z = [3]

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=1)

    def test_innerproduct(self):
        """Test inner product"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, self.y)
        self.assertEqual(qc.decompose().size(), 4)

    def test_selfinnerproduct(self):
        """Test self inner product"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x)
        self.assertEqual(qc.decompose().size(), 4)

    def test_innerproduct_nomeasurement(self):
        """Test inner product no measurement"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, self.y, measurement=False)
        self.assertEqual(qc.decompose().size(), 2)

    def test_selfinnerprodect_nomeasurement(self):
        """Test self inner product no measurement"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, measurement=False)
        self.assertEqual(qc.decompose().size(), 2)

    def test_statevector(self):
        """Test state vector simulator"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, is_statevector_sim=True)
        self.assertEqual(qc.decompose().size(), 1)

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


if __name__ == "__main__":
    unittest.main()
