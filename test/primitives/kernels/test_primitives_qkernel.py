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
import itertools
import unittest
import re

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, idata, unpack
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import Fidelity, BaseFidelity
from sklearn.svm import SVC

from qiskit_machine_learning.primitives.kernels import QuantumKernel


class MockFidelity(BaseFidelity):
    """Custom fidelity that returns -0.5 for any input."""

    def __call__(self, values_left, values_right) -> np.ndarray:
        return np.zeros(values_left.shape[0]) - 0.5


@ddt
class TestPrimitivesQuantumKernel(QiskitMachineLearningTestCase):
    """Test QuantumKernel primitives."""

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

        self.sampler_factory = functools.partial(Sampler)
        self.fidelity = Fidelity(self.sampler_factory)

    def test_svc_callable(self):
        """Test callable kernel in sklearn."""
        kernel = QuantumKernel(sampler=self.sampler_factory, feature_map=self.feature_map)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        score = svc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_svc_precomputed(self):
        """Test precomputed kernel in sklearn."""
        kernel = QuantumKernel(sampler=self.sampler_factory, feature_map=self.feature_map)
        kernel_train = kernel.evaluate(x_vec=self.sample_train)
        kernel_test = kernel.evaluate(x_vec=self.sample_test, y_vec=self.sample_train)

        svc = SVC(kernel="precomputed")
        svc.fit(kernel_train, self.label_train)
        score = svc.score(kernel_test, self.label_test)

        self.assertEqual(score, 1.0)

    @idata(
        itertools.product(
            ["1_param", "4_params"],
            ["default", "zero_prob", "fidelity_instance", "mock_fidelity"],
            ["ZZ", "Z"],
            [True, False],
        )
    )
    @unpack
    def test_evaluate_symmetric(self, params, fidelity, feature_map, enforce_psd):
        """Test QuantumKernel.evaluate(x) for a symmetric kernel."""
        solution = self.get_symmetric_solution(params, fidelity, feature_map, enforce_psd)

        if params == "1_param":
            x_vec = self.sample_train[0]
        else:
            x_vec = self.sample_train

        if feature_map == "Z":
            feature_map = self.feature_map
        else:
            feature_map = None

        if fidelity == "default":
            kernel = QuantumKernel(
                sampler=self.sampler_factory, feature_map=feature_map, enforce_psd=enforce_psd
            )
        elif fidelity == "zero_prob":
            kernel = QuantumKernel(
                sampler=self.sampler_factory,
                feature_map=feature_map,
                fidelity="zero_prob",
                enforce_psd=enforce_psd,
            )
        elif fidelity == "fidelity_instance":
            kernel = QuantumKernel(
                feature_map=feature_map, fidelity=self.fidelity, enforce_psd=enforce_psd
            )
        else:
            kernel = QuantumKernel(
                feature_map=feature_map, fidelity=MockFidelity(), enforce_psd=enforce_psd
            )

        kernel_matrix = kernel.evaluate(x_vec)

        np.testing.assert_allclose(kernel_matrix, solution, rtol=1e-4, atol=1e-10)

    @idata(
        itertools.product(
            ["1_param", "4_params", "wrong"],
            ["1_param", "4_params", "2_params", "wrong"],
            ["default", "zero_prob", "fidelity_instance", "mock_fidelity"],
            ["ZZ", "Z"],
            [True, False],
        )
    )
    @unpack
    def test_evaluate_asymmetric(self, params_x, params_y, fidelity, feature_map, enforce_psd):
        """Test QuantumKernel.evaluate(x,y) for an asymmetric kernel."""
        solution = self.get_asymmetric_solution(
            params_x, params_y, fidelity, feature_map, enforce_psd
        )

        if params_x == "1_param":
            x_vec = self.sample_train[0]
        elif params_x == "4_params":
            x_vec = self.sample_train
        else:
            x_vec = np.zeros((2, 3))

        if params_y == "1_param":
            y_vec = self.sample_train[0]
        elif params_y == "4_params":
            y_vec = self.sample_train
        elif params_y == "2_params":
            y_vec = self.sample_test
        else:
            y_vec = np.zeros((2, 3))

        if feature_map == "Z":
            feature_map = self.feature_map
        else:
            feature_map = None

        if fidelity == "default":
            kernel = QuantumKernel(
                sampler=self.sampler_factory, feature_map=feature_map, enforce_psd=enforce_psd
            )
        elif fidelity == "zero_prob":
            kernel = QuantumKernel(
                sampler=self.sampler_factory,
                feature_map=feature_map,
                fidelity="zero_prob",
                enforce_psd=enforce_psd,
            )
        elif fidelity == "fidelity_instance":
            kernel = QuantumKernel(
                feature_map=feature_map, fidelity=self.fidelity, enforce_psd=enforce_psd
            )
        else:
            kernel = QuantumKernel(
                feature_map=feature_map, fidelity=MockFidelity(), enforce_psd=enforce_psd
            )

        if solution == "wrong":
            with self.assertRaises(ValueError):
                _ = kernel.evaluate(x_vec, y_vec)
        else:
            kernel_matrix = kernel.evaluate(x_vec, y_vec)
            np.testing.assert_allclose(kernel_matrix, solution, rtol=1e-4, atol=1e-10)

    def get_symmetric_solution(self, params, fidelity, feature_map, enforce_psd):
        if fidelity == "mock_fidelity":
            if params == "1_param":
                if enforce_psd:
                    return np.array([[0.0]])
                return np.array([[-0.5]])
            if params == "4_params" and enforce_psd:
                return np.zeros((4, 4))
            return np.zeros((4, 4)) - 0.5

        # all other fidelities have the same result
        if params == "1_param":
            return np.array([[1.0]])

        if params == "4_params" and feature_map == "Z":
            return np.array(
                [
                    [1.0, 0.78883982, 0.15984355, 0.06203766],
                    [0.78883982, 1.0, 0.49363215, 0.32128356],
                    [0.15984355, 0.49363215, 1.0, 0.91953051],
                    [0.06203766, 0.32128356, 0.91953051, 1.0],
                ]
            )
        # ZZFeatureMap with 4 params
        return np.array(
            [
                [1.0, 0.81376617, 0.05102078, 0.06033439],
                [0.81376617, 1.0, 0.14750292, 0.09980414],
                [0.05102078, 0.14750292, 1.0, 0.26196463],
                [0.06033439, 0.09980414, 0.26196463, 1.0],
            ]
        )

    def get_asymmetric_solution(self, params_x, params_y, fidelity, feature_map, enforce_psd):
        if params_x == "wrong" or params_y == "wrong":
            return "wrong"
        # check if hidden symmetric case
        if params_x == params_y:
            return self.get_symmetric_solution(params_x, fidelity, feature_map, enforce_psd)

        if fidelity == "mock_fidelity":
            len_x = int(re.search(r"\d+", params_x).group())
            len_y = int(re.search(r"\d+", params_y).group())
            return np.zeros((len_x, len_y)) - 0.5

        # all other fidelities have the same result
        if feature_map == "Z":
            if params_x == "1_param" and params_y == "4_params":
                return np.array([[1.0, 0.78883982, 0.15984355, 0.06203766]])
            elif params_x == "1_param" and params_y == "2_params":
                return np.array([[0.30890363, 0.04543022]])
            elif params_x == "4_params" and params_y == "1_param":
                return np.array([[1.0, 0.78883982, 0.15984355, 0.06203766]]).T
            # 4_param and 2_param
            return np.array(
                [
                    [0.30890363, 0.04543022],
                    [0.39666513, 0.23044328],
                    [0.11826802, 0.58742761],
                    [0.10665779, 0.7650088],
                ]
            )
        # ZZFeatureMap
        if params_x == "1_param" and params_y == "4_params":
            return np.array([[1.0, 0.81376617, 0.05102078, 0.06033439]])
        elif params_x == "1_param" and params_y == "2_params":
            return np.array([[0.24610242, 0.17510262]])
        elif params_x == "4_params" and params_y == "1_param":
            return np.array([[1.0, 0.81376617, 0.05102078, 0.06033439]]).T
        # 4_param and 2_param
        return np.array(
            [
                [0.24610242, 0.17510262],
                [0.36660828, 0.06476594],
                [0.13924611, 0.48450828],
                [0.24435258, 0.31099496],
            ]
        )


if __name__ == "__main__":
    unittest.main()
