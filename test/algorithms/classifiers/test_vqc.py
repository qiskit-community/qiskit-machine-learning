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

from ddt import ddt, idata, unpack
import numpy as np
import scipy

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals, optionals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

QUANTUM_INSTANCES = ["statevector", "qasm"]
NUM_QUBITS_LIST = [2, None]
FEATURE_MAPS = ["zz_feature_map", None]
ANSATZES = ["real_amplitudes", None]
OPTIMIZERS = ["cobyla", "bfgs", None]
DATASETS = ["binary", "multiclass", "no_one_hot"]
LOSSES = ["squared_error", "absolute_error", "cross_entropy"]


@dataclass(frozen=True)
class _Dataset:
    x: np.ndarray | None = None
    y: np.ndarray | None = None


def _create_dataset(n_samples: int, n_classes: int, one_hot=True) -> _Dataset:
    x, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_classes,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=5.0,
        random_state=algorithm_globals.random_seed,
    )
    x = MinMaxScaler().fit_transform(x)
    if one_hot:
        y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    return _Dataset(x, y)


@ddt
class TestVQC(QiskitMachineLearningTestCase):
    """VQC Tests."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 1111111
        self.num_classes_by_batch = []
        from qiskit_aer import Aer

        # Set-up the quantum instances.
        statevector = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        qasm = QuantumInstance(
            Aer.get_backend("aer_simulator"),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # We want string keys to ensure DDT-generated tests have meaningful names.
        self.properties = {
            "statevector": statevector,
            "qasm": qasm,
            "bfgs": L_BFGS_B(maxiter=5),
            "cobyla": COBYLA(maxiter=25),
            "real_amplitudes": RealAmplitudes(num_qubits=2, reps=1),
            "zz_feature_map": ZZFeatureMap(2),
            "binary": _create_dataset(6, 2),
            "multiclass": _create_dataset(10, 3),
            "no_one_hot": _create_dataset(6, 2, one_hot=False),
        }

    @idata(
        itertools.product(
            QUANTUM_INSTANCES, NUM_QUBITS_LIST, FEATURE_MAPS, ANSATZES, OPTIMIZERS, DATASETS
        )
    )
    @unpack
    def test_VQC(self, q_i, num_qubits, f_m, ans, opt, d_s):
        """
        Test VQC with binary and multiclass data using a range of quantum
        instances, numbers of qubits, feature maps, and optimizers.
        """
        if num_qubits is None and f_m is None and ans is None:
            self.skipTest(
                "At least one of num_qubits, feature_map, or ansatz must be set by the user."
            )

        quantum_instance = self.properties.get(q_i)
        feature_map = self.properties.get(f_m)
        optimizer = self.properties.get(opt)
        ansatz = self.properties.get(ans)
        dataset = self.properties.get(d_s)

        initial_point = np.array([0.5] * ansatz.num_parameters) if ansatz is not None else None

        classifier = VQC(
            quantum_instance=quantum_instance,
            num_qubits=num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
        )
        classifier.fit(dataset.x, dataset.y)
        score = classifier.score(dataset.x, dataset.y)
        self.assertGreater(score, 0.5)

        predict = classifier.predict(dataset.x[0, :])
        unique_labels = np.unique(dataset.y, axis=0)
        # we want to have labels as a column array, either 1D or 2D(one hot)
        # thus, the assert works with plain and one hot labels
        unique_labels = unique_labels.reshape(len(unique_labels), -1)
        # the predicted value should be in the labels
        self.assertTrue(np.all(predict == unique_labels, axis=1).any())

    def test_VQC_non_parameterized(self):
        """
        Test VQC without an optimizer set.
        """
        classifier = VQC(
            num_qubits=2,
            optimizer=None,
            quantum_instance=self.properties.get("statevector"),
        )
        dataset = self.properties.get("binary")
        classifier.fit(dataset.x, dataset.y)
        score = classifier.score(dataset.x, dataset.y)
        self.assertGreater(score, 0.5)

    @idata(DATASETS)
    def test_warm_start(self, d_s):
        """Test VQC when training from a warm start."""

        classifier = VQC(
            feature_map=self.properties.get("zz_feature_map"),
            quantum_instance=self.properties.get("statevector"),
            warm_start=True,
        )
        dataset = self.properties.get(d_s)

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
        y_one_hot = OneHotEncoder().fit_transform(y.reshape(-1, 1))

        classifier = VQC(
            feature_map=self.properties.get("zz_feature_map"),
            ansatz=self.properties.get("real_amplitudes"),
            warm_start=True,
            quantum_instance=self.properties.get("statevector"),
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

        classifier = VQC(num_qubits=2, quantum_instance=self.properties.get("statevector"))
        with self.assertRaises(QiskitMachineLearningError):
            classifier.fit(x, y)

    def test_changing_classes_raises_error(self):
        """Tests VQC raises an error when fitting new data with a different number of classes."""

        targets1 = np.asarray([[0, 0, 1], [0, 1, 0]])
        targets2 = np.asarray([[0, 1], [1, 0]])
        features1 = algorithm_globals.random.random((len(targets1), 2))
        features2 = algorithm_globals.random.random((len(targets2), 2))

        classifier = VQC(
            num_qubits=2,
            warm_start=True,
            quantum_instance=self.properties.get("statevector"),
        )
        classifier.fit(features1, targets1)
        with self.assertRaises(QiskitMachineLearningError):
            classifier.fit(features2, targets2)

    @idata(itertools.product(QUANTUM_INSTANCES, LOSSES))
    @unpack
    def test_sparse_arrays(self, q_i, loss):
        """Tests VQC on sparse features and labels."""
        quantum_instance = self.properties.get(q_i)
        classifier = VQC(num_qubits=2, loss=loss, quantum_instance=quantum_instance)
        x = scipy.sparse.csr_matrix([[0, 0], [1, 1]])
        y = scipy.sparse.csr_matrix([[1, 0], [0, 1]])

        classifier.fit(x, y)

        score = classifier.score(x, y)
        self.assertGreaterEqual(score, 0.5)

    def test_categorical(self):
        """Test VQC on categorical labels."""
        classifier = VQC(
            num_qubits=2,
            optimizer=COBYLA(25),
            quantum_instance=self.properties.get("statevector"),
        )
        dataset = self.properties.get("no_one_hot")
        features = dataset.x
        labels = np.empty(dataset.y.shape, dtype=str)
        labels[dataset.y == 0] = "A"
        labels[dataset.y == 1] = "B"

        classifier.fit(features, labels)
        score = classifier.score(features, labels)
        self.assertGreater(score, 0.5)

        predict = classifier.predict(features[0, :])
        self.assertIn(predict, ["A", "B"])

    def test_circuit_extensions(self):
        """Test VQC when the number of qubits is different compared to the feature map/ansatz."""
        num_qubits = 2
        classifier = VQC(
            num_qubits=num_qubits,
            feature_map=ZFeatureMap(1),
            ansatz=RealAmplitudes(1),
            quantum_instance=self.properties.get("statevector"),
        )
        self.assertEqual(classifier.feature_map.num_qubits, num_qubits)
        self.assertEqual(classifier.ansatz.num_qubits, num_qubits)

        qc = QuantumCircuit(1)
        with self.assertRaises(QiskitMachineLearningError):
            _ = VQC(
                num_qubits=num_qubits,
                feature_map=qc,
                ansatz=qc,
                quantum_instance=self.properties.get("statevector"),
            )


if __name__ == "__main__":
    unittest.main()
