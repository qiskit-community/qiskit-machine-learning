# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QSVR"""
import os
import tempfile
import unittest
from test import QiskitMachineLearningTestCase

import numpy as np
from sklearn.metrics import mean_squared_error
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVR, SerializableModelMixin
from qiskit_machine_learning.exceptions import QiskitMachineLearningWarning
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.utils import algorithm_globals


class TestQSVR(QiskitMachineLearningTestCase):
    """Test QSVR Algorithm"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.feature_map = zz_feature_map(feature_dimension=2, reps=2)

        self.sample_train = np.asarray(
            [
                [-0.36572221, 0.90579879],
                [-0.41816432, 0.03011426],
                [-0.48806982, 0.87208714],
                [-0.67078436, -0.91017876],
                [-0.12980588, 0.98475113],
                [0.78335453, 0.49721604],
                [0.78158498, 0.78689328],
                [0.03771672, -0.3681419],
                [0.54402486, 0.32332253],
                [-0.25268454, -0.81106666],
            ]
        )
        self.label_train = np.asarray(
            [
                0.07045477,
                0.80047778,
                0.04493319,
                -0.30427998,
                -0.02430856,
                0.17224315,
                -0.26474769,
                0.83097582,
                0.60943777,
                0.31577759,
            ]
        )

        self.sample_test = np.asarray(
            [
                [-0.60713067, -0.37935265],
                [0.55480968, 0.94365285],
                [0.00148237, -0.71220499],
                [-0.97212742, -0.54068794],
            ]
        )
        self.label_test = np.asarray([0.45066614, -0.18052862, 0.4549451, -0.23674218])

    def test_qsvr(self):
        """Test QSVR"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map, enforce_psd=False)

        qsvr = QSVR(quantum_kernel=qkernel)
        qsvr.fit(self.sample_train, self.label_train)
        predictions = qsvr.predict(self.sample_test)
        mse = mean_squared_error(self.label_test, predictions)
        self.assertAlmostEqual(mse, 0.04964456790383482, places=4)

    def test_change_kernel(self):
        """Test QSVR with QuantumKernel set later"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map, enforce_psd=False)

        qsvr = QSVR(feature_map=QuantumCircuit(2))
        qsvr.quantum_kernel = qkernel
        qsvr.fit(self.sample_train, self.label_train)
        predictions = qsvr.predict(self.sample_test)
        mse = mean_squared_error(self.label_test, predictions)
        self.assertAlmostEqual(mse, 0.04964456790383482, places=4)

    def test_qsvr_parameters(self):
        """Test QSVR with extra constructor parameters"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvr = QSVR(quantum_kernel=qkernel, tol=1e-3, C=1.0)
        qsvr.fit(self.sample_train, self.label_train)
        predictions = qsvr.predict(self.sample_test)
        mse = mean_squared_error(self.label_test, predictions)
        self.assertAlmostEqual(mse, 0.04964456790383482, places=4)

    def test_qsvr_to_string(self):
        """Test QSVR string representation"""
        qsvr = QSVR(feature_map=self.feature_map)
        _ = str(qsvr)

    def test_with_kernel_parameter(self):
        """Test QSVR with the `kernel` argument"""
        with self.assertWarns(QiskitMachineLearningWarning):
            QSVR(feature_map=self.feature_map, kernel=1)

    def test_precomputed(self):
        """Test QSVC with the precomputed option."""
        features = np.array([[0, 0], [0.1, 0.1], [0.4, 0.4], [1, 1]])
        labels = np.array([0, 0.1, 0.4, 1])

        quantum_kernel = FidelityQuantumKernel(feature_map=zz_feature_map(2))
        evaluated_kernel = quantum_kernel.evaluate(features)
        classifier = QSVR(quantum_kernel="precomputed")
        classifier.fit(evaluated_kernel, labels)

    def test_save_load(self):
        """Tests save and load functionality"""
        features = np.array([[0, 0], [0.1, 0.1], [0.4, 0.4], [1, 1]])
        labels = np.array([0, 0.1, 0.4, 1])

        quantum_kernel = FidelityQuantumKernel(feature_map=zz_feature_map(2))
        regressor = QSVR(quantum_kernel=quantum_kernel)
        regressor.fit(features, labels)

        test_features = np.array([[0.5, 0.5]])
        original_predicts = regressor.predict(test_features)

        with tempfile.TemporaryDirectory() as dir_name:
            file_name = os.path.join(dir_name, "qsvr.model")
            regressor.to_dill(file_name)

            regressor_load = QSVR.from_dill(file_name)
            loaded_model_predicts = regressor_load.predict(test_features)

            np.testing.assert_array_almost_equal(original_predicts, loaded_model_predicts)

            class FakeModel(SerializableModelMixin):
                """Fake model class for test purposes"""

                pass

            with self.assertRaises(TypeError):
                FakeModel.from_dill(file_name)


if __name__ == "__main__":
    unittest.main()
