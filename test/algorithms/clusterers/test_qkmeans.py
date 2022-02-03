# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Quantum K-Means Clustering"""
import unittest

import numpy as np
from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms.clusterers import QKMeans
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from test import QiskitMachineLearningTestCase


class TestQKMeans(QiskitMachineLearningTestCase):
    """Test Quantum K-Means Clustering"""

    # pylint: disable=invalid-name
    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.X = np.array([[1, 1], [2, 1], [100, 90], [120, 140]])
        self.init_centers = "k-means++"

        self.expected_labels = np.array([0, 0, 1, 1])
        self.expected_centers = np.array([[1.5, 1], [110, 115]])

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=1024,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

    def test_parameters(self):
        """Test proper errors are raised on invalid classical parameters"""
        # Test invalid number of clusters - <= 0
        with self.assertRaises(ValueError):
            QKMeans(n_clusters=0, quantum_instance=self.qasm_simulator).fit(self.X)

        # Test invalid number of clusters - > n_samples
        with self.assertRaises(ValueError):
            QKMeans(n_clusters=6, quantum_instance=self.qasm_simulator).fit(self.X)

        # Test invalid initialization - invalid string
        with self.assertRaises(ValueError):
            QKMeans(n_clusters=2, init="invalid", quantum_instance=self.qasm_simulator).fit(self.X)

        # Test invalid initialization - invalid shape
        with self.assertRaises(ValueError):
            QKMeans(
                n_clusters=2, init=np.array([[0], [1]]), quantum_instance=self.qasm_simulator
            ).fit(self.X)

        # Test invalid initialization - invalid number of clusters
        with self.assertRaises(ValueError):
            QKMeans(
                n_clusters=2,
                init=np.array([[0, 0], [1, 1], [2, 2]]),
                quantum_instance=self.qasm_simulator,
            ).fit(self.X)

        # Test invalid initialization - 0 vector as initial centroid
        with self.assertRaises(QiskitMachineLearningError):
            QKMeans(
                n_clusters=2,
                init=np.array([[0, 1], [0, 0]]),
                quantum_instance=self.qasm_simulator,
            ).fit(self.X)

        # Test max_iter - <= 0
        with self.assertRaises(ValueError):
            QKMeans(n_clusters=2, max_iter=0, quantum_instance=self.qasm_simulator).fit(self.X)

        # Test invalid dataset - 0 vector in dataset
        with self.assertRaises(QiskitMachineLearningError):
            QKMeans(n_clusters=2, quantum_instance=self.qasm_simulator).fit(
                np.array([[2, 0], [0, 0]])
            )

    # pylint: disable=invalid-name
    def test_quantum_instance(self):
        """Test proper errors are raised on invalid quantum_instance"""
        # Test missing quantum_instance
        with self.assertRaises(ValueError):
            QKMeans(n_clusters=2).fit(self.X)

        # very high dimensional data
        X = np.random.rand(5, 40000)

        # Test invalid quantum_instance
        with self.assertRaises(QiskitMachineLearningError):
            QKMeans(n_clusters=2, quantum_instance=self.qasm_simulator).fit(X)

    def test_fit(self):
        """Test algorithm on a simple dataset"""
        qkmeans = QKMeans(
            n_clusters=2, init=self.init_centers, quantum_instance=self.qasm_simulator
        )
        qkmeans.fit(self.X)

        # Check that the labels are correct
        self.assertTrue(np.array_equal(qkmeans.labels_, self.expected_labels))

        # Check that the centers are almost correct
        self.assertTrue(np.allclose(qkmeans.cluster_centers_, self.expected_centers, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
