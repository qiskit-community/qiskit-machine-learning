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

import functools
import unittest
from test import QiskitMachineLearningTestCase

from typing import Callable

import numpy as np
import scipy
from ddt import ddt, data
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms import VQC


@ddt
class TestVQC(QiskitMachineLearningTestCase):
    """VQC Tests."""

    def setUp(self):
        super().setUp()

        self.num_classes_by_batch = []

        # specify quantum instances
        algorithm_globals.random_seed = 12345
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

    @data(
        # optimizer, quantum instance
        ("cobyla", "statevector"),
        ("cobyla", "qasm"),
        ("bfgs", "statevector"),
        ("bfgs", "qasm"),
        (None, "statevector"),
        (None, "qasm"),
    )
    def test_vqc(self, config):
        """Test VQC."""

        opt, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        elif q_i == "qasm":
            quantum_instance = self.qasm_quantum_instance
        else:
            quantum_instance = None

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)
        # fix the initial point
        initial_point = np.array([0.5] * ansatz.num_parameters)

        # construct classifier - note: CrossEntropy requires eval_probabilities=True!
        classifier = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )

        # construct data
        num_samples = 5
        # pylint: disable=invalid-name
        X = algorithm_globals.random.random((num_samples, num_inputs))
        y = 1.0 * (np.sum(X, axis=1) <= 1)
        while len(np.unique(y)) == 1:
            X = algorithm_globals.random.random((num_samples, num_inputs))
            y = 1.0 * (np.sum(X, axis=1) <= 1)
        y = np.array([y, 1 - y]).transpose()  # VQC requires one-hot encoded input

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)

    @data(
        # num_qubits, feature_map, ansatz
        (True, False, False),
        (True, True, False),
        (True, True, True),
        (False, True, True),
        (False, False, True),
        (True, False, True),
        (False, True, False),
    )
    def test_default_parameters(self, config):
        """Test VQC instantiation with default parameters."""

        provide_num_qubits, provide_feature_map, provide_ansatz = config
        num_inputs = 2

        num_qubits, feature_map, ansatz = None, None, None

        if provide_num_qubits:
            num_qubits = num_inputs
        if provide_feature_map:
            feature_map = ZZFeatureMap(num_inputs)
        if provide_ansatz:
            ansatz = RealAmplitudes(num_inputs, reps=1)

        classifier = VQC(
            num_qubits=num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=self.qasm_quantum_instance,
        )

        # construct data
        num_samples = 5
        # pylint: disable=invalid-name
        X = algorithm_globals.random.random((num_samples, num_inputs))
        y = 1.0 * (np.sum(X, axis=1) <= 1)
        while len(np.unique(y)) == 1:
            X = algorithm_globals.random.random((num_samples, num_inputs))
            y = 1.0 * (np.sum(X, axis=1) <= 1)
        y = np.array([y, 1 - y]).transpose()  # VQC requires one-hot encoded input

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)

    @data(
        # optimizer, quantum instance
        ("cobyla", "statevector"),
        ("cobyla", "qasm"),
        ("bfgs", "statevector"),
        ("bfgs", "qasm"),
        (None, "statevector"),
        (None, "qasm"),
    )
    def test_multiclass(self, config):
        """Test multiclass VQC."""
        opt, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        elif q_i == "qasm":
            quantum_instance = self.qasm_quantum_instance
        else:
            quantum_instance = None

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)
        # fix the initial point
        initial_point = np.array([0.5] * ansatz.num_parameters)

        # construct classifier - note: CrossEntropy requires eval_probabilities=True!
        classifier = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )

        # construct data
        num_samples = 5
        num_classes = 5
        # pylint: disable=invalid-name

        # We create a dataset that is random, but has some training signal, as follows:
        # First, we create a random feature matrix X, but sort it by the row-wise sum in ascending
        # order.
        X = algorithm_globals.random.random((num_samples, num_inputs))
        X = X[X.sum(1).argsort()]

        # Next we create an array which contains all class labels, multiple times if num_samples <
        # num_classes, and in ascending order (e.g. [0, 0, 1, 1, 2]). So now we have a dataset
        # where the row-sum of X is correlated with the class label (i.e. smaller row-sum is more
        # likely to belong to class 0, and big row-sum is more likely to belong to class >0)
        y_indices = (
            np.digitize(np.arange(0, 1, 1 / num_samples), np.arange(0, 1, 1 / num_classes)) - 1
        )

        # Third, we random shuffle both X and y_indices
        permutation = np.random.permutation(np.arange(num_samples))
        X = X[permutation]
        y_indices = y_indices[permutation]

        # Lastly we create a 1-hot label matrix y
        y = np.zeros((num_samples, num_classes))
        for e, index in enumerate(y_indices):
            y[e, index] = 1

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 1 / num_classes)

    @data(
        # optimizer, quantum instance
        ("cobyla", "statevector"),
        ("cobyla", "qasm"),
        ("bfgs", "statevector"),
        ("bfgs", "qasm"),
        (None, "statevector"),
        (None, "qasm"),
    )
    def test_warm_start(self, config):
        """Test VQC with warm_start=True."""
        opt, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        elif q_i == "qasm":
            quantum_instance = self.qasm_quantum_instance
        else:
            quantum_instance = None

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # Construct the data.
        num_samples = 10
        # pylint: disable=invalid-name
        X = algorithm_globals.random.random((num_samples, num_inputs))
        y = 1.0 * (np.sum(X, axis=1) <= 1)
        while len(np.unique(y)) == 1:
            X = algorithm_globals.random.random((num_samples, num_inputs))
            y = 1.0 * (np.sum(X, axis=1) <= 1)
        y = np.array([y, 1 - y]).transpose()  # VQC requires one-hot encoded input.

        # Initialize the VQC.
        classifier = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            warm_start=True,
            quantum_instance=quantum_instance,
        )

        # Fit the VQC to the first half of the data.
        num_start = num_samples // 2
        classifier.fit(X[:num_start, :], y[:num_start])
        first_fit_final_point = classifier._fit_result.x

        # Fit the VQC to the second half of the data with a warm start.
        classifier.fit(X[num_start:, :], y[num_start:])
        second_fit_initial_point = classifier._initial_point

        # Check the final optimization point from the first fit was used to start the second fit.
        np.testing.assert_allclose(first_fit_final_point, second_fit_initial_point)

        # Check score.
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)

    def get_num_classes(self, func: Callable) -> Callable:
        """Wrapper to record the number of classes assumed when building CircuitQNN."""

        @functools.wraps(func)
        def wrapper(num_classes: int):
            self.num_classes_by_batch.append(num_classes)
            return func(num_classes)

        return wrapper

    @data(
        # optimizer, quantum instance
        ("cobyla", "statevector"),
        ("cobyla", "qasm"),
        ("bfgs", "statevector"),
        ("bfgs", "qasm"),
        (None, "statevector"),
        (None, "qasm"),
    )
    def test_batches_with_incomplete_labels(self, config):
        """Test VQC when some batches do not include all possible labels."""
        opt, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        elif q_i == "qasm":
            quantum_instance = self.qasm_quantum_instance
        else:
            quantum_instance = None

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # Construct the data.
        features = algorithm_globals.random.random((15, num_inputs))
        target = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        num_classes = len(np.unique(target))

        # One-hot encode the target.
        target_onehot = np.zeros((target.size, int(target.max() + 1)))
        target_onehot[np.arange(target.size), target.astype(int)] = 1

        # Initialize the VQC.
        classifier = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            warm_start=True,
            quantum_instance=quantum_instance,
        )

        classifier._get_interpret = self.get_num_classes(classifier._get_interpret)

        # Fit the VQC to the first third of the data.
        classifier.fit(features[:5, :], target_onehot[:5])

        # Fit the VQC to the second third of the data with a warm start.
        classifier.fit(features[5:10, :], target_onehot[5:10])

        # Fit the VQC to the third third of the data with a warm start.
        classifier.fit(features[10:, :], target_onehot[10:])

        # Check all batches assume the correct number of classes
        self.assertTrue((np.asarray(self.num_classes_by_batch) == num_classes).all())

    def test_sparse_arrays(self):
        """Tests VQC on sparse features and labels."""
        for quantum_instance in [self.sv_quantum_instance, self.qasm_quantum_instance]:
            for loss in ["squared_error", "absolute_error", "cross_entropy"]:
                with self.subTest(f"quantum_instance: {quantum_instance}, loss: {loss}"):
                    self._test_sparse_arrays(quantum_instance, loss)

    def _test_sparse_arrays(self, quantum_instance: QuantumInstance, loss: str):
        classifier = VQC(num_qubits=2, loss=loss, quantum_instance=quantum_instance)
        features = scipy.sparse.csr_matrix([[0, 0], [1, 1]])
        labels = scipy.sparse.csr_matrix([[1, 0], [0, 1]])

        # fit to data
        classifier.fit(features, labels)

        # score
        score = classifier.score(features, labels)
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
