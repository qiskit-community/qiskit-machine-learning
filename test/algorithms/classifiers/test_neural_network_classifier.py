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
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss


@ddt
class TestNeuralNetworkClassifier(QiskitMachineLearningTestCase):
    """Neural Network Classifier Tests."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        self.random_seed = 12345
        self.sv_quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'),
                                                   seed_simulator=self.random_seed,
                                                   seed_transpiler=self.random_seed)
        self.qasm_quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100,
                                                     seed_simulator=self.random_seed,
                                                     seed_transpiler=self.random_seed)
        np.random.seed(self.random_seed)

    @data(
        # optimizer, loss, quantum instance
        ('cobyla', 'l1', 'statevector'),
        ('cobyla', 'l1', 'qasm'),

        ('cobyla', 'l2', 'statevector'),
        ('cobyla', 'l2', 'qasm'),

        ('bfgs', 'l1', 'statevector'),
        ('bfgs', 'l1', 'qasm'),

        ('bfgs', 'l2', 'statevector'),
        ('bfgs', 'l2', 'qasm'),
    )
    def test_classifier_with_opflow_qnn(self, config):
        """ Test Neural Network Classifier with Opflow QNN (Two Layer QNN)."""

        opt, loss, q_i = config

        if q_i == 'statevector':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == 'bfgs':
            optimizer = L_BFGS_B(maxiter=5)
        else:
            optimizer = COBYLA(maxiter=25)

        num_inputs = 2
        ansatz = RealAmplitudes(num_inputs, reps=1)
        qnn = TwoLayerQNN(num_inputs, ansatz=ansatz, quantum_instance=quantum_instance)

        classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss=loss)

        # construct data
        num_samples = 5
        X = np.random.rand(num_samples, num_inputs)  # pylint: disable=invalid-name
        y = 2.0 * (np.sum(X, axis=1) <= 1) - 1.0

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)

    @data(
        # optimizer, loss, quantum instance
        ('cobyla', 'l1', 'statevector'),
        ('cobyla', 'l1', 'qasm'),

        ('cobyla', 'l2', 'statevector'),
        ('cobyla', 'l2', 'qasm'),

        ('bfgs', 'l1', 'statevector'),
        ('bfgs', 'l1', 'qasm'),

        ('bfgs', 'l2', 'statevector'),
        ('bfgs', 'l2', 'qasm'),
    )
    def test_classifier_with_circuit_qnn(self, config):
        """ Test Neural Network Classifier with Circuit QNN."""

        opt, loss, q_i = config

        if q_i == 'statevector':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == 'bfgs':
            optimizer = L_BFGS_B(maxiter=5)
        else:
            optimizer = COBYLA(maxiter=25)

        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(2))
        qc.append(ansatz, range(2))

        # construct qnn
        def parity(x):
            return '{:b}'.format(x).count('1') % 2

        output_shape = 2
        qnn = CircuitQNN(qc, input_params=feature_map.parameters,
                         weight_params=ansatz.parameters,
                         sparse=False,
                         interpret=parity,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)

        # construct classifier
        classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss=loss)

        # construct data
        num_samples = 5
        X = np.random.rand(num_samples, num_inputs)  # pylint: disable=invalid-name
        y = 1.0 * (np.sum(X, axis=1) <= 1)

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)

    @data(
        # optimizer, quantum instance
        ('cobyla', 'statevector'),
        ('cobyla', 'qasm'),

        ('bfgs', 'statevector'),
        ('bfgs', 'qasm'),
    )
    def test_classifier_with_circuit_qnn_and_cross_entropy(self, config):
        """ Test Neural Network Classifier with Circuit QNN and Cross Entropy loss."""

        opt, q_i = config

        if q_i == 'statevector':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == 'bfgs':
            optimizer = L_BFGS_B(maxiter=5)
        else:
            optimizer = COBYLA(maxiter=25)

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
            return '{:b}'.format(x).count('1') % 2

        output_shape = 2
        qnn = CircuitQNN(qc, input_params=feature_map.parameters,
                         weight_params=ansatz.parameters,
                         sparse=False,
                         interpret=parity,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)

        # construct classifier - note: CrossEntropy requires eval_probabilities=True!
        classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss=loss, one_hot=True)

        # construct data
        num_samples = 5
        X = np.random.rand(num_samples, num_inputs)  # pylint: disable=invalid-name
        y = 1.0 * (np.sum(X, axis=1) <= 1)
        y = np.array([y, 1 - y]).transpose()

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        self.assertGreater(score, 0.5)


if __name__ == '__main__':
    unittest.main()
