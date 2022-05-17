# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Neural Network Classifier """

from __future__ import annotations
from dataclasses import dataclass

from test import QiskitMachineLearningTestCase
import functools
import itertools
import unittest

import numpy as np
import scipy

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


@dataclass
class _Dataset:
    name: str | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None


def _one_hot_encode(y: np.ndarray) -> np.ndarray:
    y_one_hot = np.zeros((y.size, int(y.max() + 1)), dtype=int)
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


class TestVQC(QiskitMachineLearningTestCase):
    """VQC Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 1111111

        # Set-up the quantum instances.
        self.sv_quantum_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        self.qasm_quantum_instance = QuantumInstance(
            Aer.get_backend("aer_simulator"),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        self.quantum_instances = [self.sv_quantum_instance, self.qasm_quantum_instance]

        # Set-up the numbers of qubits.
        self.num_qubits_list = [2, None]

        # Set-up the feature maps.
        self.num_features = 2
        self.zz_feature_map = ZZFeatureMap(self.num_features)
        self.feature_maps = [self.zz_feature_map, None]

        # Set-up the ansatzes
        self.real_amplitudes = RealAmplitudes(self.num_features, reps=1)
        self.ansatzes = [self.real_amplitudes, None]

        # Set-up the optimizers.
        bfgs = L_BFGS_B(maxiter=3)
        cobyla = COBYLA(maxiter=15)
        self.optimizers = [bfgs, cobyla]

        # Set-up the loss functions.
        self.losses = ["squared_error", "absolute_error", "cross_entropy"]

        # Set-up the datasets.
        binary_x, binary_y = make_classification(
            n_samples=6,
            n_features=self.num_features,
            n_classes=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=5.0,
            random_state=algorithm_globals.random_seed,
        )
        self.binary_dataset = _Dataset(
            "binary", MinMaxScaler().fit_transform(binary_x), _one_hot_encode(binary_y)
        )

        multiclass_x, multiclass_y = make_classification(
            n_samples=9,
            n_features=self.num_features,
            n_classes=3,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=5.0,
            random_state=algorithm_globals.random_seed,
        )
        self.multiclass_dataset = _Dataset(
            "multiclass", MinMaxScaler().fit_transform(multiclass_x), _one_hot_encode(multiclass_y)
        )

        self.datasets = [self.binary_dataset, self.multiclass_dataset]
        self.num_classes_by_batch = []

    def test_VQC(self):
        """
        Test VQC with binary and multiclass data using a range of quantum
        instances, numbers of qubits, feature maps, and optimizers.
        """

        for (
            quantum_instance,
            num_qubits,
            feature_map,
            ansatz,
            optimizer,
            dataset,
        ) in itertools.product(
            self.quantum_instances,
            self.num_qubits_list,
            self.feature_maps,
            self.ansatzes,
            self.optimizers,
            self.datasets,
        ):
            subtest_name = (
                f"{quantum_instance.backend_name}, {num_qubits}, "
                f"{type(feature_map).__name__}, {type(optimizer).__name__}, "
                f"{dataset.name}"
            )
            with self.subTest(subtest_name):
                self.setUp()
                if num_qubits is None and feature_map is None and ansatz is None:
                    self.skipTest(
                        "At least one of num_qubits, feature_map, or ansatz must be set by the user."
                    )

                initial_point = (
                    np.array([0.5] * ansatz.num_parameters) if ansatz is not None else None
                )

                # construct classifier - note: CrossEntropy requires eval_probabilities=True!
                classifier = VQC(
                    num_qubits=num_qubits,
                    feature_map=feature_map,
                    ansatz=ansatz,
                    optimizer=optimizer,
                    quantum_instance=quantum_instance,
                    initial_point=initial_point,
                )

                classifier.fit(dataset.x, dataset.y)
                score = classifier.score(dataset.x, dataset.y)
                self.assertGreater(score, 0.5)
                self.tearDown()

    def test_VQC_non_parameterized(self):
        """
        Test VQC with binary data
        """
        classifier = VQC(
            num_qubits=2,
            optimizer=None,
            quantum_instance=self.sv_quantum_instance,
        )
        classifier.fit(self.binary_dataset.x, self.binary_dataset.y)
        score = classifier.score(self.binary_dataset.x, self.binary_dataset.y)
        self.assertGreater(score, 0.5)

    def test_warm_start(self):
        """Test VQC when training from a warm start."""

        for dataset in self.datasets:
            with self.subTest(dataset.name):
                self.setUp()
                classifier = VQC(
                    feature_map=self.zz_feature_map,
                    quantum_instance=self.sv_quantum_instance,
                    warm_start=True,
                )

                # Fit the VQC to the first half of the data.
                num_start = len(dataset.y) // 2
                classifier.fit(dataset.x[:num_start, :], dataset.y[:num_start])
                first_fit_final_point = classifier._fit_result.x

                # Fit the VQC to the second half of the data with a warm start.
                classifier.fit(dataset.x[num_start:, :], dataset.y[num_start:])
                second_fit_initial_point = classifier._initial_point

                # Check the final optimization point from the first fit was used to start the second fit.
                np.testing.assert_allclose(first_fit_final_point, second_fit_initial_point)

                score = classifier.score(dataset.x, dataset.y)
                self.assertGreater(score, 0.5)
                self.tearDown()

    def _get_num_classes(self, func):
        """Wrapper to record the number of classes assumed when building CircuitQNN."""

        @functools.wraps(func)
        def wrapper(num_classes):
            self.num_classes_by_batch.append(num_classes)
            return func(num_classes)

        return wrapper

    def test_batches_with_incomplete_labels(self):
        """Test VQC when targets are one-hot and some batches don't have all possible labels."""

        # Generate data with batches that have incomplete labels.
        x = algorithm_globals.random.random((6, 2))
        y = np.asarray([0, 0, 1, 1, 2, 2])
        y_one_hot = _one_hot_encode(y)

        classifier = VQC(
            feature_map=self.zz_feature_map,
            ansatz=self.real_amplitudes,
            warm_start=True,
            quantum_instance=self.sv_quantum_instance,
        )

        classifier._get_interpret = self._get_num_classes(classifier._get_interpret)

        num_classes_list = []
        classifier.fit(x[:2, :], y_one_hot[:2])
        num_classes_list.append(classifier.num_classes)
        classifier.fit(x[2:4, :], y_one_hot[2:4])
        num_classes_list.append(classifier.num_classes)
        classifier.fit(x[4:, :], y_one_hot[4:])
        num_classes_list.append(classifier.num_classes)

        with self.subTest("Test all batches assume the correct number of classes."):
            self.assertTrue((np.asarray(num_classes_list) == 3).all())

        with self.subTest("Check correct number of classes is used to build CircuitQNN."):
            self.assertTrue((np.asarray(self.num_classes_by_batch) == 3).all())

    def test_multilabel_targets_raise_an_error(self):
        """Tests VQC multi-label input raises an error."""

        # Generate multi-label data.
        x = algorithm_globals.random.random((3, 2))
        y = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])

        classifier = VQC(num_qubits=2, quantum_instance=self.sv_quantum_instance)
        with self.assertRaises(QiskitMachineLearningError):
            classifier.fit(x, y)

    def test_changing_classes_raises_error(self):
        """Tests VQC raises an error when fitting new data with a different number of classes."""

        targets1 = np.asarray([[0, 0, 1], [0, 1, 0]])
        targets2 = np.asarray([[0, 1], [1, 0]])
        features1 = algorithm_globals.random.random((len(targets1), 2))
        features2 = algorithm_globals.random.random((len(targets2), 2))

        classifier = VQC(num_qubits=2, warm_start=True, quantum_instance=self.sv_quantum_instance)
        classifier.fit(features1, targets1)
        with self.assertRaises(QiskitMachineLearningError):
            classifier.fit(features2, targets2)

    def test_sparse_arrays(self):
        """Tests VQC on sparse features and labels."""
        for quantum_instance, loss in itertools.product(self.quantum_instances, self.losses):
            classifier = VQC(num_qubits=2, loss=loss, quantum_instance=quantum_instance)
            x = scipy.sparse.csr_matrix([[0, 0], [1, 1]])
            y = scipy.sparse.csr_matrix([[1, 0], [0, 1]])

            classifier.fit(x, y)

            score = classifier.score(x, y)
            self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
