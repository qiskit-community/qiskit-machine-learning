# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Neural Network Classifier """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss


@ddt
class TestNeuralNetworkClassifier(QiskitMachineLearningTestCase):
    """Neural Network Classifier Tests."""

    def setUp(self):
        super().setUp()

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
        # optimizer, loss, quantum instance
        ("cobyla", "absolute_error", "statevector"),
        ("cobyla", "absolute_error", "qasm"),
        ("cobyla", "squared_error", "statevector"),
        ("cobyla", "squared_error", "qasm"),
        ("bfgs", "absolute_error", "statevector"),
        ("bfgs", "absolute_error", "qasm"),
        ("bfgs", "squared_error", "statevector"),
        ("bfgs", "squared_error", "qasm"),
        (None, "absolute_error", "statevector"),
        (None, "absolute_error", "qasm"),
        (None, "squared_error", "statevector"),
        (None, "squared_error", "qasm"),
    )
    def test_classifier_with_opflow_qnn(self, config):
        """Test Neural Network Classifier with Opflow QNN (Two Layer QNN)."""

        opt, loss, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_inputs = 2
        ansatz = RealAmplitudes(num_inputs, reps=1)
        qnn = TwoLayerQNN(num_inputs, ansatz=ansatz, quantum_instance=quantum_instance)
        initial_point = np.array([0.5] * ansatz.num_parameters)

        classifier = NeuralNetworkClassifier(
            qnn, optimizer=optimizer, loss=loss, initial_point=initial_point
        )

        # construct data
        num_samples = 5
        X = algorithm_globals.random.random(  # pylint: disable=invalid-name
            (num_samples, num_inputs)
        )
        y = 2.0 * (np.sum(X, axis=1) <= 1) - 1.0

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)

    @data(
        # optimizer, loss, quantum instance
        ("cobyla", "absolute_error", "statevector"),
        ("cobyla", "absolute_error", "qasm"),
        ("cobyla", "squared_error", "statevector"),
        ("cobyla", "squared_error", "qasm"),
        ("bfgs", "absolute_error", "statevector"),
        ("bfgs", "absolute_error", "qasm"),
        ("bfgs", "squared_error", "statevector"),
        ("bfgs", "squared_error", "qasm"),
        (None, "absolute_error", "statevector"),
        (None, "absolute_error", "qasm"),
        (None, "squared_error", "statevector"),
        (None, "squared_error", "qasm"),
    )
    def test_classifier_with_circuit_qnn(self, config):
        """Test Neural Network Classifier with Circuit QNN."""

        opt, loss, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(2))
        qc.append(ansatz, range(2))

        # construct qnn
        def parity(x):
            return "{:b}".format(x).count("1") % 2

        output_shape = 2
        qnn = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
        )
        initial_point = np.array([0.5] * ansatz.num_parameters)

        # construct classifier
        classifier = NeuralNetworkClassifier(
            qnn, optimizer=optimizer, loss=loss, initial_point=initial_point
        )

        # construct data
        num_samples = 5
        X = algorithm_globals.random.random(  # pylint: disable=invalid-name
            (num_samples, num_inputs)
        )
        y = 1.0 * (np.sum(X, axis=1) <= 1)

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
    def test_classifier_with_circuit_qnn_and_cross_entropy(self, config):
        """Test Neural Network Classifier with Circuit QNN and Cross Entropy loss."""

        opt, q_i = config

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        loss = CrossEntropyLoss()

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(2))
        qc.append(ansatz, range(2))

        # construct qnn
        def parity(x):
            return "{:b}".format(x).count("1") % 2

        output_shape = 2
        qnn = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
        )
        # classification may fail sometimes, so let's fix initial point
        initial_point = np.array([0.5] * ansatz.num_parameters)
        # construct classifier - note: CrossEntropy requires eval_probabilities=True!
        classifier = NeuralNetworkClassifier(
            qnn, optimizer=optimizer, loss=loss, one_hot=True, initial_point=initial_point
        )

        # construct data
        num_samples = 5
        X = algorithm_globals.random.random(  # pylint: disable=invalid-name
            (num_samples, num_inputs)
        )
        y = 1.0 * (np.sum(X, axis=1) <= 1)
        y = np.array([y, 1 - y]).transpose()

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
