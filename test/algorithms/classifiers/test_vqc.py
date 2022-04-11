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

from test import QiskitMachineLearningTestCase
import functools
import itertools
import unittest

import numpy as np
import scipy
from ddt import ddt, idata, data, unpack

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

RANDOM_SEED = 1111111
algorithm_globals.random_seed = RANDOM_SEED

# Set-up the quantum instances.
SV_QUANTUM_INSTANCE_ = QuantumInstance(
    Aer.get_backend("aer_simulator_statevector"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)
QASM_QUANTUM_INSTANCE_ = QuantumInstance(
    Aer.get_backend("aer_simulator"),
    shots=100,
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)
QUANTUM_INSTANCES_ = [SV_QUANTUM_INSTANCE_, QASM_QUANTUM_INSTANCE_]

# Set-up the numbers of qubits.
NUM_QUBITS_LIST = [2, None]

# Set-up the feature maps.
NUM_FEATURES_ = 2
ZZ_FEATURE_MAP_ = ZZFeatureMap(NUM_FEATURES_)
FEATURE_MAPS_ = [ZZ_FEATURE_MAP_, None]

# Set-up the ansatzes
REAL_AMPLITUDES_ = RealAmplitudes(NUM_FEATURES_, reps=1)
ANSATZES_ = [REAL_AMPLITUDES_, None]

# Set-up the optimizers.
BFGS_ = L_BFGS_B(maxiter=3)
COBYLA_ = COBYLA(maxiter=15)
OPTIMIZERS_ = [BFGS_, COBYLA_]

# Set-up the loss functions.
LOSSES = ["squared_error", "absolute_error", "cross_entropy"]


def _one_hot_encode(y: np.ndarray) -> np.ndarray:
    y_one_hot = np.zeros((y.size, int(y.max() + 1)), dtype=int)
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


# Set-up the datasets.
BINARY_X, BINARY_Y = make_classification(
    n_samples=6,
    n_features=NUM_FEATURES_,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=5.0,
    random_state=algorithm_globals.random_seed,
)
BINARY_X = MinMaxScaler().fit_transform(BINARY_X)
BINARY_Y = _one_hot_encode(BINARY_Y)
BINARY_DATASET = (BINARY_X, BINARY_Y)

MULTICLASS_X, MULTICLASS_Y = make_classification(
    n_samples=9,
    n_features=NUM_FEATURES_,
    n_classes=3,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=5.0,
    random_state=algorithm_globals.random_seed,
)
MULTICLASS_X = MinMaxScaler().fit_transform(MULTICLASS_X)
MULTICLASS_Y = _one_hot_encode(MULTICLASS_Y)
MULTICLASS_DATASET = (MULTICLASS_X, MULTICLASS_Y)

DATASETS_ = [BINARY_DATASET, MULTICLASS_DATASET]


@ddt
class TestVQC(QiskitMachineLearningTestCase):
    """VQC Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = RANDOM_SEED
        self.num_classes_by_batch = []

    @idata(
        itertools.product(
            QUANTUM_INSTANCES_, NUM_QUBITS_LIST, FEATURE_MAPS_, ANSATZES_, OPTIMIZERS_, DATASETS_
        )
    )
    @unpack
    def test_VQC(self, quantum_instance, num_qubits, feature_map, ansatz, optimizer, dataset):
        """
        Test VQC with binary and multiclass data using a range of quantum
        instances, numbers of qubits, feature maps, and OPTIMIZERS.
        """
        if num_qubits is None and feature_map is None and ansatz is None:
            self.skipTest(
                "At least one of num_qubits, feature_map, or ansatz must be set by the user."
            )

        initial_point = np.array([0.5] * ansatz.num_parameters) if ansatz is not None else None

        # construct classifier - note: CrossEntropy requires eval_probabilities=True!
        classifier = VQC(
            num_qubits=num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )

        x, y = dataset
        classifier.fit(x, y)
        score = classifier.score(x, y)
        self.assertGreater(score, 0.5)

    def test_VQC_non_parameterized(self):
        """
        Test VQC with binary data
        """
        classifier = VQC(
            num_qubits=2,
            optimizer=None,
            quantum_instance=SV_QUANTUM_INSTANCE_,
        )
        classifier.fit(BINARY_X, BINARY_Y)
        score = classifier.score(BINARY_X, BINARY_Y)
        self.assertGreater(score, 0.5)

    @idata(
        itertools.product(
            [SV_QUANTUM_INSTANCE_],
            [ZZ_FEATURE_MAP_],
            DATASETS_,
        )
    )
    @unpack
    def test_warm_start(self, quantum_instance, feature_map, dataset):
        """Test VQC when training from a warm start."""

        classifier = VQC(
            feature_map=feature_map,
            quantum_instance=quantum_instance,
            warm_start=True,
        )

        x, y = dataset

        # Fit the VQC to the first half of the data.
        num_start = len(y) // 2
        classifier.fit(x[:num_start, :], y[:num_start])
        first_fit_final_point = classifier._fit_result.x

        # Fit the VQC to the second half of the data with a warm start.
        classifier.fit(x[num_start:, :], y[num_start:])
        second_fit_initial_point = classifier._initial_point

        # Check the final optimization point from the first fit was used to start the second fit.
        np.testing.assert_allclose(first_fit_final_point, second_fit_initial_point)

        score = classifier.score(x, y)
        self.assertGreater(score, 0.5)

    def _get_num_classes(self, func):
        """Wrapper to record the number of classes assumed when building CircuitQNN."""

        @functools.wraps(func)
        def wrapper(num_classes):
            self.num_classes_by_batch.append(num_classes)
            return func(num_classes)

        return wrapper

    @data(
        (ZZ_FEATURE_MAP_, REAL_AMPLITUDES_, SV_QUANTUM_INSTANCE_),
    )
    @unpack
    def test_batches_with_incomplete_labels(self, feature_map, ansatz, quantum_instance):
        """Test VQC when targets are one-hot and some batches don't have all possible labels."""

        # Generate data with batches that have incomplete labels.
        x = algorithm_globals.random.random((6, 2))
        y = np.asarray([0, 0, 1, 1, 2, 2])
        y_one_hot = _one_hot_encode(y)

        classifier = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            warm_start=True,
            quantum_instance=quantum_instance,
        )

        classifier._get_interpret = self._get_num_classes(classifier._get_interpret)

        classifier.fit(x[:2, :], y_one_hot[:2])
        classifier.fit(x[2:4, :], y_one_hot[2:4])
        classifier.fit(x[4:, :], y_one_hot[4:])

        # Check all batches assume the correct number of classes.
        self.assertTrue((np.asarray(self.num_classes_by_batch) == 3).all())

    @data(SV_QUANTUM_INSTANCE_)
    def test_multilabel_targets_raise_an_error(self, quantum_instance):
        """Tests VQC multi-label input raises an error."""

        # Generate multi-label data.
        x = algorithm_globals.random.random((3, 2))
        y = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])

        classifier = VQC(num_qubits=2, quantum_instance=quantum_instance)
        with self.assertRaises(QiskitMachineLearningError):
            classifier.fit(x, y)

    @data(SV_QUANTUM_INSTANCE_)
    def test_changing_classes_raises_error(self, quantum_instance):
        """Tests VQC raises an error when fitting new data with a different number of classes."""

        targets1 = np.asarray([[0, 0, 1], [0, 1, 0]])
        targets2 = np.asarray([[0, 1], [1, 0]])
        features1 = algorithm_globals.random.random((len(targets1), 2))
        features2 = algorithm_globals.random.random((len(targets2), 2))

        classifier = VQC(num_qubits=2, warm_start=True, quantum_instance=quantum_instance)
        classifier.fit(features1, targets1)
        with self.assertRaises(QiskitMachineLearningError):
            classifier.fit(features2, targets2)

    @idata(itertools.product(QUANTUM_INSTANCES_, LOSSES))
    @unpack
    def test_sparse_arrays(self, quantum_instance, loss):
        """Tests VQC on sparse features and labels."""
        classifier = VQC(num_qubits=2, loss=loss, quantum_instance=quantum_instance)
        x = scipy.sparse.csr_matrix([[0, 0], [1, 1]])
        y = scipy.sparse.csr_matrix([[1, 0], [0, 1]])

        classifier.fit(x, y)

        score = classifier.score(x, y)
        self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
