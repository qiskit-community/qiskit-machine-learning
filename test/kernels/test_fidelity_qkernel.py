# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test FidelityQuantumKernel."""

from __future__ import annotations

import functools
import itertools
import unittest
from typing import Sequence

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, idata, unpack
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Sampler

from qiskit_machine_learning.algorithm_job import AlgorithmJob
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.state_fidelities import (
    ComputeUncompute,
    BaseStateFidelity,
    StateFidelityResult,
)
from qiskit_machine_learning.kernels import FidelityQuantumKernel


@ddt
class TestFidelityQuantumKernel(QiskitMachineLearningTestCase):
    """Test FidelityQuantumKernel."""

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

        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(self.sampler)

        self.properties = {
            "samples_1": self.sample_train[0],
            "samples_4": self.sample_train,
            "samples_test": self.sample_test,
            "z_fm": self.feature_map,
            "no_fm": None,
        }

    def test_svc_callable(self):
        """Test callable kernel in sklearn."""
        kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        score = svc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_svc_precomputed(self):
        """Test precomputed kernel in sklearn."""
        kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        kernel_train = kernel.evaluate(x_vec=self.sample_train)
        kernel_test = kernel.evaluate(x_vec=self.sample_test, y_vec=self.sample_train)

        svc = SVC(kernel="precomputed")
        svc.fit(kernel_train, self.label_train)
        score = svc.score(kernel_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_defaults(self):
        """Test quantum kernel with all default values."""
        features = algorithm_globals.random.random((10, 2)) - 0.5
        labels = np.sign(features[:, 0])

        kernel = FidelityQuantumKernel()
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(features, labels)
        score = svc.score(features, labels)

        self.assertGreaterEqual(score, 0.5)

    def test_max_circuits_per_job(self):
        """Test max_circuits_per_job parameters."""
        kernel_all = FidelityQuantumKernel(feature_map=self.feature_map, max_circuits_per_job=None)
        kernel_matrix_all = kernel_all.evaluate(x_vec=self.sample_train)
        with self.subTest("Check when max_circuits_per_job > left_parameters"):
            kernel_more = FidelityQuantumKernel(
                feature_map=self.feature_map, max_circuits_per_job=20
            )
            kernel_matrix_more = kernel_more.evaluate(x_vec=self.sample_train)
            np.testing.assert_equal(kernel_matrix_all, kernel_matrix_more)
        with self.subTest("Check when max_circuits_per_job = 1"):
            kernel_1 = FidelityQuantumKernel(feature_map=self.feature_map, max_circuits_per_job=1)
            kernel_matrix_1 = kernel_1.evaluate(x_vec=self.sample_train)
            np.testing.assert_equal(kernel_matrix_all, kernel_matrix_1)

    def test_exceptions(self):
        """Test quantum kernel raises exceptions and warnings."""
        with self.assertRaises(ValueError, msg="Unsupported value of 'evaluate_duplicates'."):
            _ = FidelityQuantumKernel(evaluate_duplicates="wrong")
        with self.assertRaises(ValueError, msg="Unsupported value of 'max_circuits_per_job'."):
            _ = FidelityQuantumKernel(max_circuits_per_job=-1)

    # pylint: disable=too-many-positional-arguments
    @idata(
        # params, fidelity, feature map, enforce_psd, duplicate
        itertools.product(
            ["samples_1", "samples_4"],
            ["default_fidelity", "fidelity_instance"],
            ["no_fm", "z_fm"],
            [True, False],
            ["none", "off_diagonal", "all"],
        )
    )
    @unpack
    def test_evaluate_symmetric(self, params, fidelity, feature_map, enforce_psd, duplicates):
        """Test QuantumKernel.evaluate(x) for a symmetric kernel."""
        solution = self._get_symmetric_solution(params, feature_map)

        x_vec = self.properties[params]
        feature_map = self.properties[feature_map]
        kernel = self._create_kernel(fidelity, feature_map, enforce_psd, duplicates)

        kernel_matrix = kernel.evaluate(x_vec)

        np.testing.assert_allclose(kernel_matrix, solution, rtol=1e-4, atol=1e-10)

    # pylint: disable=too-many-positional-arguments
    @idata(
        itertools.product(
            ["samples_1", "samples_4"],
            ["samples_1", "samples_4", "samples_test"],
            ["default_fidelity", "fidelity_instance"],
            ["no_fm", "z_fm"],
            [True, False],
            ["none", "off_diagonal", "all"],
        )
    )
    @unpack
    def test_evaluate_asymmetric(
        self, params_x, params_y, fidelity, feature_map, enforce_psd, duplicates
    ):
        """Test QuantumKernel.evaluate(x,y) for an asymmetric kernel."""
        solution = self._get_asymmetric_solution(params_x, params_y, feature_map)

        x_vec = self.properties[params_x]
        y_vec = self.properties[params_y]
        feature_map = self.properties[feature_map]
        kernel = self._create_kernel(fidelity, feature_map, enforce_psd, duplicates)

        if isinstance(solution, str) and solution == "wrong":
            with self.assertRaises(ValueError):
                _ = kernel.evaluate(x_vec, y_vec)
        else:
            kernel_matrix = kernel.evaluate(x_vec, y_vec)
            np.testing.assert_allclose(kernel_matrix, solution, rtol=1e-4, atol=1e-10)

    def _create_kernel(self, fidelity, feature_map, enforce_psd, duplicates):
        if fidelity == "default_fidelity":
            kernel = FidelityQuantumKernel(
                feature_map=feature_map,
                enforce_psd=enforce_psd,
                evaluate_duplicates=duplicates,
            )
        elif fidelity == "fidelity_instance":
            kernel = FidelityQuantumKernel(
                feature_map=feature_map,
                fidelity=self.fidelity,
                enforce_psd=enforce_psd,
                evaluate_duplicates=duplicates,
            )
        else:
            raise ValueError("Unsupported configuration!")
        return kernel

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

    def test_enforce_psd(self):
        """Test enforce_psd"""

        class MockFidelity(BaseStateFidelity):
            """Custom fidelity that returns -0.5 for any input."""

            def create_fidelity_circuit(
                self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
            ) -> QuantumCircuit:
                raise NotImplementedError()

            def _run(  # type: ignore[override]
                self,
                circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
                circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
                values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
                values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
                **options,
            ) -> StateFidelityResult | AlgorithmJob:
                values = np.asarray(values_1)
                fidelities = np.full(values.shape[0], -0.5)

                return AlgorithmJob(MockFidelity._call, fidelities, options)

            @staticmethod
            def _call(fidelities, options) -> StateFidelityResult:
                return StateFidelityResult(fidelities, [], {}, options)  # type: ignore[arg-type]

        with self.subTest("No PSD enforcement"):
            kernel = FidelityQuantumKernel(fidelity=MockFidelity(), enforce_psd=False)
            matrix = kernel.evaluate(self.sample_train)
            eigen_values = np.linalg.eigvals(matrix)
            # there's a negative eigenvalue
            self.assertFalse(np.all(np.greater_equal(eigen_values, -1e-10)))

        with self.subTest("PSD enforced"):
            kernel = FidelityQuantumKernel(fidelity=MockFidelity(), enforce_psd=True)
            matrix = kernel.evaluate(self.sample_train)
            eigen_values = np.linalg.eigvals(matrix)
            # all eigenvalues are non-negative with some tolerance
            self.assertTrue(np.all(np.greater_equal(eigen_values, -1e-10)))

    def test_validate_input(self):
        """Test validation of input data in the base (abstract) class."""
        with self.subTest("Incorrect size of x_vec"):
            kernel = FidelityQuantumKernel()

            x_vec = np.asarray([[[0]]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec)

            x_vec = np.asarray([])
            self.assertRaises(ValueError, kernel.evaluate, x_vec)

        with self.subTest("Adjust the number of qubits in the feature map"):
            kernel = FidelityQuantumKernel()

            x_vec = np.asarray([[1, 2, 3]])
            kernel.evaluate(x_vec)
            self.assertEqual(kernel.feature_map.num_qubits, 3)

        with self.subTest("Fail to adjust the number of qubits in the feature map"):
            qc = QuantumCircuit(1)
            kernel = FidelityQuantumKernel(feature_map=qc)

            x_vec = np.asarray([[1, 2]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec)

        with self.subTest("Incorrect size of y_vec"):
            kernel = FidelityQuantumKernel()

            x_vec = np.asarray([[1, 2]])
            y_vec = np.asarray([[[0]]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec, y_vec)

            x_vec = np.asarray([[1, 2]])
            y_vec = np.asarray([])
            self.assertRaises(ValueError, kernel.evaluate, x_vec, y_vec)

        with self.subTest("Fail when x_vec and y_vec have different shapes"):
            kernel = FidelityQuantumKernel()

            x_vec = np.asarray([[1, 2]])
            y_vec = np.asarray([[1, 2, 3]])
            self.assertRaises(ValueError, kernel.evaluate, x_vec, y_vec)

    def test_properties(self):
        """Test properties of the base (abstract) class and fidelity based kernel."""
        qc = QuantumCircuit(1)
        qc.ry(Parameter("w"), 0)
        fidelity = ComputeUncompute(sampler=Sampler())
        kernel = FidelityQuantumKernel(
            feature_map=qc, fidelity=fidelity, enforce_psd=False, evaluate_duplicates="none"
        )

        self.assertEqual(qc, kernel.feature_map)
        self.assertEqual(fidelity, kernel.fidelity)
        self.assertEqual(False, kernel.enforce_psd)
        self.assertEqual("none", kernel.evaluate_duplicates)
        self.assertEqual(1, kernel.num_features)


@ddt
class TestDuplicates(QiskitMachineLearningTestCase):
    """Test quantum kernel with duplicate entries."""

    def setUp(self) -> None:
        super().setUp()

        self.feature_map = ZFeatureMap(feature_dimension=2, reps=1)

        self.properties = {
            "no_dups": np.array([[1, 2], [2, 3], [3, 4]]),
            "dups": np.array([[1, 2], [1, 2], [3, 4]]),
            "y_vec": np.array([[0, 1], [1, 2]]),
        }

        counting_sampler = Sampler()
        counting_sampler.run = self.count_circuits(counting_sampler.run)
        self.counting_sampler = counting_sampler
        self.circuit_counts = 0

    def count_circuits(self, func):
        """Wrapper to record the number of circuits passed to QuantumInstance.execute.

        Args:
            func (Callable): execute function to be wrapped

        Returns:
            Callable: function wrapper
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.circuit_counts += len(kwargs["circuits"])
            return func(*args, **kwargs)

        return wrapper

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
        """Tests quantum kernel evaluation with duplicate samples."""
        self.circuit_counts = 0
        kernel = FidelityQuantumKernel(
            fidelity=ComputeUncompute(sampler=self.counting_sampler),
            feature_map=self.feature_map,
            evaluate_duplicates=evaluate_duplicates,
        )
        kernel.evaluate(self.properties.get(dataset_name))

        self.assertEqual(self.circuit_counts, expected_num_circuits)

    @idata(
        [
            ("no_dups", "all", 6),
            ("no_dups", "off_diagonal", 6),
            ("no_dups", "none", 5),
        ]
    )
    @unpack
    def test_evaluate_duplicates_asymmetric(
        self, dataset_name, evaluate_duplicates, expected_num_circuits
    ):
        """Tests asymmetric quantum kernel evaluation with duplicate samples."""
        self.circuit_counts = 0
        kernel = FidelityQuantumKernel(
            fidelity=ComputeUncompute(sampler=self.counting_sampler),
            feature_map=self.feature_map,
            evaluate_duplicates=evaluate_duplicates,
        )
        kernel.evaluate(self.properties.get(dataset_name), self.properties.get("y_vec"))
        self.assertEqual(self.circuit_counts, expected_num_circuits)


if __name__ == "__main__":
    unittest.main()
