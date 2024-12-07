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

""" Test Pegasos QSVC """
import os
import tempfile
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZFeatureMap

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import PegasosQSVC, SerializableModelMixin
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class TestPegasosQSVC(QiskitMachineLearningTestCase):
    """Test Pegasos QSVC Algorithm"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598

        # number of qubits is equal to the number of features
        self.q = 2
        # number of steps performed during the training procedure
        self.tau = 100

        self.feature_map = ZFeatureMap(feature_dimension=self.q, reps=1)

        sample, label = make_blobs(
            n_samples=20, n_features=2, centers=2, random_state=3, shuffle=True
        )
        sample = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(sample)

        # split into train and test set
        self.sample_train = sample[:15]
        self.label_train = label[:15]
        self.sample_test = sample[15:]
        self.label_test = label[15:]

        # The same for a 4-dimensional example
        # number of qubits is equal to the number of features
        self.q_4d = 4
        self.feature_map_4d = ZFeatureMap(feature_dimension=self.q_4d, reps=1)

        sample_4d, label_4d = make_blobs(
            n_samples=20, n_features=self.q_4d, centers=2, random_state=3, shuffle=True
        )
        sample_4d = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(sample_4d)

        # split into train and test set
        self.sample_train_4d = sample_4d[:15]
        self.label_train_4d = label_4d[:15]
        self.sample_test_4d = sample_4d[15:]
        self.label_test_4d = label_4d[15:]

    def test_qsvc(self):
        """
        Test the Pegasos QSVC algorithm.
        """
        quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        classifier = PegasosQSVC(quantum_kernel=quantum_kernel, C=1000, num_steps=self.tau)
        classifier.fit(self.sample_train, self.label_train)

        # Evaluate the model on the test data
        test_score = classifier.score(self.sample_test, self.label_test)
        self.assertEqual(test_score, 1.0)

        # Expected predictions for the given test data
        predicted_labels = classifier.predict(self.sample_test)
        self.assertTrue(np.array_equal(predicted_labels, self.label_test))

        # Test predict_proba method (normalization is imposed by definition)
        probas = classifier.predict_proba(self.sample_test)
        expected_probas = np.array(
            [
                [0.67722117, 0.32277883],
                [0.35775209, 0.64224791],
                [0.36540916, 0.63459084],
                [0.64419096, 0.35580904],
                [0.35864466, 0.64135534],
            ]
        )
        self.assertEqual(probas.shape, (self.label_test.shape[0], 2))
        np.testing.assert_array_almost_equal(probas, expected_probas, decimal=5)

    def test_decision_function(self):
        """Test PegasosQSVC."""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=1000, num_steps=self.tau)

        pegasos_qsvc.fit(self.sample_train, self.label_train)
        decision_function = pegasos_qsvc.decision_function(self.sample_test)

        self.assertTrue(np.all((decision_function > 0) == (self.label_test == 0)))

    def test_qsvc_4d(self):
        """Test PegasosQSVC with 4-dimensional input data"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map_4d)

        pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=1000, num_steps=self.tau)

        pegasos_qsvc.fit(self.sample_train_4d, self.label_train_4d)
        score = pegasos_qsvc.score(self.sample_test_4d, self.label_test_4d)
        self.assertEqual(score, 1.0)

    def test_precomputed_kernel(self):
        """Test PegasosQSVC with a precomputed kernel matrix"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        pegasos_qsvc = PegasosQSVC(C=1000, num_steps=self.tau, precomputed=True)

        # training
        kernel_matrix_train = qkernel.evaluate(self.sample_train, self.sample_train)
        pegasos_qsvc.fit(kernel_matrix_train, self.label_train)

        # testing
        kernel_matrix_test = qkernel.evaluate(self.sample_test, self.sample_train)
        score = pegasos_qsvc.score(kernel_matrix_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_change_kernel(self):
        """Test QSVC with FidelityQuantumKernel later"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        pegasos_qsvc = PegasosQSVC(C=1000, num_steps=self.tau)
        pegasos_qsvc.quantum_kernel = qkernel
        pegasos_qsvc.fit(self.sample_train, self.label_train)
        score = pegasos_qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1)

    def test_labels(self):
        """Test PegasosQSVC with different integer labels than {0, 1}"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=1000, num_steps=self.tau)

        label_train_temp = self.label_train.copy()
        label_train_temp[self.label_train == 0] = 2
        label_train_temp[self.label_train == 1] = 3

        label_test_temp = self.label_test.copy()
        label_test_temp[self.label_test == 0] = 2
        label_test_temp[self.label_test == 1] = 3

        pegasos_qsvc.fit(self.sample_train, label_train_temp)
        score = pegasos_qsvc.score(self.sample_test, label_test_temp)

        self.assertEqual(score, 1.0)

    def test_constructor(self):
        """Tests properties of PegasosQSVC"""
        with self.subTest("Default parameters"):
            pegasos_qsvc = PegasosQSVC()
            self.assertIsInstance(pegasos_qsvc.quantum_kernel, FidelityQuantumKernel)
            self.assertFalse(pegasos_qsvc.precomputed)
            self.assertEqual(pegasos_qsvc.num_steps, 1000)

        with self.subTest("PegasosQSVC with QuantumKernel"):
            qkernel = FidelityQuantumKernel(feature_map=self.feature_map)
            pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel)
            self.assertIsInstance(pegasos_qsvc.quantum_kernel, FidelityQuantumKernel)
            self.assertFalse(pegasos_qsvc.precomputed)

        with self.subTest("PegasosQSVC with precomputed kernel"):
            pegasos_qsvc = PegasosQSVC(precomputed=True)
            self.assertIsNone(pegasos_qsvc.quantum_kernel)
            self.assertTrue(pegasos_qsvc.precomputed)

        with self.subTest("PegasosQSVC with wrong parameters"):
            qkernel = FidelityQuantumKernel(feature_map=self.feature_map)
            with self.assertRaises(ValueError):
                _ = PegasosQSVC(quantum_kernel=qkernel, precomputed=True)

        with self.subTest("Both kernel and precomputed are passed"):
            qkernel = FidelityQuantumKernel(feature_map=self.feature_map)
            self.assertRaises(ValueError, PegasosQSVC, quantum_kernel=qkernel, precomputed=True)

    def test_change_kernel_types(self):
        """Test PegasosQSVC with a precomputed kernel matrix"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        pegasos_qsvc = PegasosQSVC(C=1000, num_steps=self.tau, precomputed=True)

        # train precomputed
        kernel_matrix_train = qkernel.evaluate(self.sample_train, self.sample_train)
        pegasos_qsvc.fit(kernel_matrix_train, self.label_train)

        # now train the same instance, but with a new quantum kernel
        pegasos_qsvc.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        pegasos_qsvc.fit(self.sample_train, self.label_train)
        score = pegasos_qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_save_load(self):
        """Tests save and load models."""
        features = np.array([[0, 0], [0.1, 0.2], [1, 1], [0.9, 0.8]])
        labels = np.array([0, 0, 1, 1])

        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        regressor = PegasosQSVC(quantum_kernel=qkernel, C=1000, num_steps=self.tau)
        regressor.fit(features, labels)

        # predicted labels from the newly trained model
        test_features = np.array([[0.5, 0.5]])
        original_predicts = regressor.predict(test_features)

        # save/load, change the quantum instance and check if predicted values are the same
        file_name = os.path.join(tempfile.gettempdir(), "pegasos.model")
        regressor.save(file_name)
        try:
            regressor_load = PegasosQSVC.load(file_name)
            loaded_model_predicts = regressor_load.predict(test_features)

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
