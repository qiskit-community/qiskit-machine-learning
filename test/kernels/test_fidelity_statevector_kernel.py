# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test FidelityStatevectorKernel."""

from __future__ import annotations

import functools
import itertools
import pickle
import sys
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, idata, unpack
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import optionals
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.kernels import FidelityStatevectorKernel


@ddt
class TestStatevectorKernel(QiskitMachineLearningTestCase):
    """Test FidelityStatevectorKernel."""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.feature_map = ZFeatureMap(feature_dimension=2, reps=2)

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

        self.properties = {
            "samples_1": self.sample_train[0],
            "samples_4": self.sample_train,
            "samples_test": self.sample_test,
            "z_fm": self.feature_map,
            "no_fm": None,
        }

    def test_svc_callable(self):
        """Test callable kernel in sklearn."""
        kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        score = svc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_svc_precomputed(self):
        """Test precomputed kernel in sklearn."""
        kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
        kernel_train = kernel.evaluate(x_vec=self.sample_train)
        kernel_test = kernel.evaluate(x_vec=self.sample_test, y_vec=self.sample_train)

        svc = SVC(kernel="precomputed")
        svc.fit(kernel_train, self.label_train)
        score = svc.score(kernel_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_defaults(self):
        """Test statevector kernel with all default values."""
        features = algorithm_globals.random.random((10, 2)) - 0.5
        labels = np.sign(features[:, 0])

        kernel = FidelityStatevectorKernel()
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(features, labels)
        score = svc.score(features, labels)

        self.assertGreaterEqual(score, 0.5)

    def test_with_shot_noise(self):
        """Test statevector kernel with shot noise emulation."""
        features = algorithm_globals.random.random((3, 2)) - 0.5
        kernel = FidelityStatevectorKernel(
            feature_map=self.feature_map, shots=10, enforce_psd=False
        )
        kernel_train = kernel.evaluate(x_vec=features)
        np.testing.assert_array_almost_equal(
            kernel_train, [[1, 0.9, 0.9], [0.4, 1, 1], [0.7, 0.8, 1]]
        )

    def test_enforce_psd(self):
        """Test enforce_psd"""

        with self.subTest("No PSD enforcement"):
            kernel = FidelityStatevectorKernel(enforce_psd=False, shots=1)
            kernel._add_shot_noise = lambda *args, **kwargs: -1
            matrix = kernel.evaluate(self.sample_train)
            w = np.linalg.eigvals(matrix)
            # there's a negative eigenvalue
            self.assertFalse(np.all(np.greater_equal(w, -1e-10)))

        with self.subTest("PSD enforced"):
            kernel = FidelityStatevectorKernel(enforce_psd=True, shots=1)
            kernel._add_shot_noise = lambda *args, **kwargs: -1
            matrix = kernel.evaluate(self.sample_train)
            w = np.linalg.eigvals(matrix)
            # all eigenvalues are non-negative with some tolerance
            self.assertTrue(np.all(np.greater_equal(w, -1e-10)))

    # todo: enable the test on macOS when fixed: https://github.com/Qiskit/qiskit-aer/issues/1886
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @unittest.skipIf(sys.platform.startswith("darwin"), "macOS is not supported")
    def test_aer_statevector(self):
        """Test statevector kernel when using AerStatevector type statevectors."""
        from qiskit_aer.quantum_info import AerStatevector

        features = algorithm_globals.random.random((10, 2)) - 0.5
        labels = np.sign(features[:, 0])

        kernel = FidelityStatevectorKernel(statevector_type=AerStatevector)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(features, labels)
        score = svc.score(features, labels)

        self.assertGreaterEqual(score, 0.5)

    def test_statevector_cache(self):
        """Test filling and clearing the statevector cache."""
        kernel = FidelityStatevectorKernel(auto_clear_cache=False)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        with self.subTest("Check cache fills correctly."):
            # pylint: disable=no-member
            self.assertEqual(kernel._get_statevector.cache_info().currsize, len(self.sample_train))

        svc.fit(self.sample_test, self.label_test)
        with self.subTest("Check no auto_clear_cache."):
            # pylint: disable=no-member
            self.assertEqual(
                kernel._get_statevector.cache_info().currsize,
                len(self.sample_train) + len(self.sample_test),
            )

        kernel = FidelityStatevectorKernel(cache_size=3, auto_clear_cache=False)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        with self.subTest("Check cache limit respected."):
            # pylint: disable=no-member
            self.assertEqual(kernel._get_statevector.cache_info().currsize, 3)

        kernel.clear_cache()
        with self.subTest("Check cache clears correctly"):
            # pylint: disable=no-member
            self.assertEqual(kernel._get_statevector.cache_info().currsize, 0)

    @idata(
        # params, feature map, duplicate
        itertools.product(
            ["samples_1", "samples_4"],
            ["no_fm", "z_fm"],
        )
    )
    @unpack
    def test_evaluate_symmetric(self, params, feature_map):
        """Test QuantumKernel.evaluate(x) for a symmetric kernel."""
        solution = self._get_symmetric_solution(params, feature_map)

        x_vec = self.properties[params]
        feature_map = self.properties[feature_map]
        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
        )

        kernel_matrix = kernel.evaluate(x_vec)

        np.testing.assert_allclose(kernel_matrix, solution, rtol=1e-4, atol=1e-10)

    @idata(
        itertools.product(
            ["samples_1", "samples_4"],
            ["samples_1", "samples_4", "samples_test"],
            ["no_fm", "z_fm"],
        )
    )
    @unpack
    def test_evaluate_asymmetric(self, params_x, params_y, feature_map):
        """Test QuantumKernel.evaluate(x,y) for an asymmetric kernel."""
        solution = self._get_asymmetric_solution(params_x, params_y, feature_map)

        x_vec = self.properties[params_x]
        y_vec = self.properties[params_y]
        feature_map = self.properties[feature_map]
        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
        )

        if isinstance(solution, str) and solution == "wrong":
            with self.assertRaises(ValueError):
                _ = kernel.evaluate(x_vec, y_vec)
        else:
            kernel_matrix = kernel.evaluate(x_vec, y_vec)
            np.testing.assert_allclose(kernel_matrix, solution, rtol=1e-4, atol=1e-10)

    def _get_symmetric_solution(self, params, feature_map):
        if params == "samples_1":
            solution = np.array([[1.0]])

        elif params == "samples_4" and feature_map == "z_fm":
            solution = np.array(
                [
                    [1.0, 0.78883982, 0.15984355, 0.06203766],
                    [0.78883982, 1.0, 0.49363215, 0.32128356],
                    [0.15984355, 0.49363215, 1.0, 0.91953051],
                    [0.06203766, 0.32128356, 0.91953051, 1.0],
                ]
            )
        else:
            # ZZFeatureMap with 4 params
            solution = np.array(
                [
                    [1.0, 0.81376617, 0.05102078, 0.06033439],
                    [0.81376617, 1.0, 0.14750292, 0.09980414],
                    [0.05102078, 0.14750292, 1.0, 0.26196463],
                    [0.06033439, 0.09980414, 0.26196463, 1.0],
                ]
            )
        return solution

    def _get_asymmetric_solution(self, params_x, params_y, feature_map):
        if params_x == "wrong" or params_y == "wrong":
            return "wrong"
        # check if hidden symmetric case
        if params_x == params_y:
            return self._get_symmetric_solution(params_x, feature_map)

        if feature_map == "z_fm":
            if params_x == "samples_1" and params_y == "samples_4":
                solution = np.array([[1.0, 0.78883982, 0.15984355, 0.06203766]])
            elif params_x == "samples_1" and params_y == "samples_test":
                solution = np.array([[0.30890363, 0.04543022]])
            elif params_x == "samples_4" and params_y == "samples_1":
                solution = np.array([[1.0, 0.78883982, 0.15984355, 0.06203766]]).T
            else:
                # 4_param and 2_param
                solution = np.array(
                    [
                        [0.30890363, 0.04543022],
                        [0.39666513, 0.23044328],
                        [0.11826802, 0.58742761],
                        [0.10665779, 0.7650088],
                    ]
                )
        else:
            # ZZFeatureMap
            if params_x == "samples_1" and params_y == "samples_4":
                solution = np.array([[1.0, 0.81376617, 0.05102078, 0.06033439]])
            elif params_x == "samples_1" and params_y == "samples_test":
                solution = np.array([[0.24610242, 0.17510262]])
            elif params_x == "samples_4" and params_y == "samples_1":
                solution = np.array([[1.0, 0.81376617, 0.05102078, 0.06033439]]).T
            else:
                # 4_param and 2_param
                solution = np.array(
                    [
                        [0.24610242, 0.17510262],
                        [0.36660828, 0.06476594],
                        [0.13924611, 0.48450828],
                        [0.24435258, 0.31099496],
                    ]
                )
        return solution

    def test_validate_input(self):
        """Test validation of input data in the base (abstract) class."""
        with self.subTest("Incorrect size of x_vec"):
            kernel = FidelityStatevectorKernel()

            x_vec = np.asarray([[[0]]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec)

            x_vec = np.asarray([])
            self.assertRaises(ValueError, kernel.evaluate, x_vec)

        with self.subTest("Adjust the number of qubits in the feature map"):
            kernel = FidelityStatevectorKernel()

            x_vec = np.asarray([[1, 2, 3]])
            kernel.evaluate(x_vec)
            self.assertEqual(kernel.feature_map.num_qubits, 3)

        with self.subTest("Fail to adjust the number of qubits in the feature map"):
            qc = QuantumCircuit(1)
            kernel = FidelityStatevectorKernel(feature_map=qc)

            x_vec = np.asarray([[1, 2]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec)

        with self.subTest("Incorrect size of y_vec"):
            kernel = FidelityStatevectorKernel()

            x_vec = np.asarray([[1, 2]])
            y_vec = np.asarray([[[0]]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec, y_vec)

            x_vec = np.asarray([[1, 2]])
            y_vec = np.asarray([])
            self.assertRaises(ValueError, kernel.evaluate, x_vec, y_vec)

        with self.subTest("Fail when x_vec and y_vec have different shapes"):
            kernel = FidelityStatevectorKernel()

            x_vec = np.asarray([[1, 2]])
            y_vec = np.asarray([[1, 2, 3]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec, y_vec)

    def test_properties(self):
        """Test properties of the base (abstract) class and statevector based kernel."""
        qc = QuantumCircuit(1)
        qc.ry(Parameter("w"), 0)
        kernel = FidelityStatevectorKernel(feature_map=qc)

        self.assertEqual(qc, kernel.feature_map)
        self.assertEqual(1, kernel.num_features)

    def test_pickling(self):
        """Test that the kernel can be pickled correctly and without error."""
        # Compares original kernel with copies made using pickle module and get & set state directly
        qc = QuantumCircuit(1)
        qc.ry(Parameter("w"), 0)
        kernel1 = FidelityStatevectorKernel(feature_map=qc)

        pickled_obj = pickle.dumps(kernel1)
        kernel2 = pickle.loads(pickled_obj)

        kernel3 = FidelityStatevectorKernel()
        kernel3.__setstate__(kernel1.__getstate__())

        with self.subTest("Pickle fail, kernels are not the same type"):
            self.assertEqual(type(kernel1), type(kernel2))

        with self.subTest("Pickle fail, kernels are not the same type"):
            self.assertEqual(type(kernel1), type(kernel3))

        with self.subTest("Pickle fail, kernels are not unique objects"):
            self.assertNotEqual(kernel1, kernel2)

        with self.subTest("Pickle fail, kernels are not unique objects"):
            self.assertNotEqual(kernel1, kernel3)

        with self.subTest("Pickle fail, caches are not the same type"):
            self.assertEqual(type(kernel1._get_statevector), type(kernel2._get_statevector))

        with self.subTest("Pickle fail, caches are not the same type"):
            self.assertEqual(type(kernel1._get_statevector), type(kernel3._get_statevector))

        # Remove cache to check dict properties are otherwise identical.
        # - caches are never identical as they have different RAM locations.
        kernel1.__dict__["_get_statevector"] = None
        kernel2.__dict__["_get_statevector"] = None
        kernel3.__dict__["_get_statevector"] = None

        # Confirm changes were made.
        with self.subTest("Pickle fail, caches have not been removed from kernels"):
            self.assertEqual(kernel1._get_statevector, None)
            self.assertEqual(kernel2._get_statevector, None)
            self.assertEqual(kernel3._get_statevector, None)

        with self.subTest("Pickle fail, properties of kernels (bar cache) are not identical"):
            self.assertEqual(kernel1.__dict__, kernel2.__dict__)

        with self.subTest("Pickle fail, properties of kernels (bar cache) are not identical"):
            self.assertEqual(kernel1.__dict__, kernel3.__dict__)


@ddt
class TestStatevectorKernelDuplicates(QiskitMachineLearningTestCase):
    """Test statevector kernel with duplicate entries."""

    def setUp(self) -> None:
        super().setUp()

        self.feature_map = ZFeatureMap(feature_dimension=2, reps=1)

        self.properties = {
            "no_dups": np.array([[1, 2], [2, 3], [3, 4]]),
            "dups": np.array([[1, 2], [1, 2], [3, 4]]),
            "y_vec": np.array([[0, 1], [1, 2]]),
        }

        self.computation_counts = 0

    def count_computations(self, func):
        """Wrapper to record the number of computed kernel entries.

        Args:
            func (Callable): execute function to be wrapped

        Returns:
            Callable: function wrapper
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.computation_counts += 1
            return func(*args, **kwargs)

        return wrapper

    @idata(
        [
            ("no_dups", 6),
            ("dups", 4),
        ]
    )
    @unpack
    def test_with_duplicates(self, dataset_name, expected_computations):
        """Tests statevector kernel evaluation with duplicate samples."""
        self.computation_counts = 0
        kernel = FidelityStatevectorKernel(
            feature_map=self.feature_map,
        )
        kernel._compute_kernel_entry = self.count_computations(kernel._compute_kernel_entry)
        kernel.evaluate(self.properties.get(dataset_name))

        self.assertEqual(self.computation_counts, expected_computations)

    @idata(
        [
            ("no_dups", 5),
            ("dups", 4),
        ]
    )
    @unpack
    def test_with_duplicates_asymmetric(self, dataset_name, expected_computations):
        """Tests asymmetric statevector kernel evaluation with duplicate samples."""
        self.computation_counts = 0
        kernel = FidelityStatevectorKernel(
            feature_map=self.feature_map,
        )
        kernel._compute_kernel_entry = self.count_computations(kernel._compute_kernel_entry)
        kernel.evaluate(self.properties.get(dataset_name), self.properties.get("y_vec"))
        self.assertEqual(self.computation_counts, expected_computations)


if __name__ == "__main__":
    unittest.main()
