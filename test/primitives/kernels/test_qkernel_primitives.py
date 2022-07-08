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
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals
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


if __name__ == "__main__":
    unittest.main()
