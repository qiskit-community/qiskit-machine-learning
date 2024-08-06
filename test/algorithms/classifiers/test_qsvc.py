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

""" Test QSVC """
import os
import tempfile
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, SerializableModelMixin
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.exceptions import (
    QiskitMachineLearningWarning,
)


class TestQSVC(QiskitMachineLearningTestCase):
    """Test QSVC Algorithm"""

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

    def test_qsvc(self):
        """Test QSVC"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_change_kernel(self):
        """Test QSVC with FidelityQuantumKernel later"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvc = QSVC()
        qsvc.quantum_kernel = qkernel
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_qsvc_parameters(self):
        """Test QSVC with extra constructor parameters"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvc = QSVC(quantum_kernel=qkernel, tol=1e-4, C=0.5)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_qsvc_to_string(self):
        """Test QSVC print works when no *args passed in"""
        qsvc = QSVC()
        _ = str(qsvc)

    def test_with_kernel_parameter(self):
        """Test QSVC with the `kernel` argument."""
        with self.assertWarns(QiskitMachineLearningWarning):
            QSVC(kernel=1)

    def test_save_load(self):
        """Tests save and load models."""
        features = np.array([[0, 0], [0.1, 0.2], [1, 1], [0.9, 0.8]])
        labels = np.array([0, 0, 1, 1])

        quantum_kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(2))
        classifier = QSVC(quantum_kernel=quantum_kernel)
        classifier.fit(features, labels)

        # predicted labels from the newly trained model
        test_features = np.array([[0.2, 0.1], [0.8, 0.9]])
        original_predicts = classifier.predict(test_features)

        # save/load, change the quantum instance and check if predicted values are the same
        file_name = os.path.join(tempfile.gettempdir(), "qsvc.model")
        classifier.save(file_name)
        try:
            classifier_load = QSVC.load(file_name)
            loaded_model_predicts = classifier_load.predict(test_features)

            np.testing.assert_array_almost_equal(original_predicts, loaded_model_predicts)

            # test loading warning
            class FakeModel(SerializableModelMixin):
                """Fake model class for test purposes."""

                pass

            with self.assertRaises(TypeError):
                FakeModel.load(file_name)

        finally:
            os.remove(file_name)


if __name__ == "__main__":
    unittest.main()
