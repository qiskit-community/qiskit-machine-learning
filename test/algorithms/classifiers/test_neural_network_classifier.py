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

from test import QiskitMachineLearningTestCase

import unittest
from ddt import ddt, data

import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions.loss import CrossEntropyLoss


@ddt
class TestNeuralNetworkClassifier(QiskitMachineLearningTestCase):
    """Opflow QNN Tests."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        self.sv_quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.qasm_quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100)

    @data(
        # optimizer, loss, warm start, quantum instance
        ('cobyla', 'l1', True, 'statevector'),
        ('cobyla', 'l1', True, 'qasm'),
        ('cobyla', 'l1', True, 'statevector'),
        ('cobyla', 'l1', True, 'qasm'),

        ('cobyla', 'l2', True, 'statevector'),
        ('cobyla', 'l2', True, 'qasm'),
        ('cobyla', 'l2', True, 'statevector'),
        ('cobyla', 'l2', True, 'qasm'),

        ('bfgs', 'l1', True, 'statevector'),
        ('bfgs', 'l1', True, 'qasm'),
        ('bfgs', 'l1', True, 'statevector'),
        ('bfgs', 'l1', True, 'qasm'),

        ('bfgs', 'l2', True, 'statevector'),
        ('bfgs', 'l2', True, 'qasm'),
        ('bfgs', 'l2', True, 'statevector'),
        ('bfgs', 'l2', True, 'qasm'),
    )
    def test_classifier_with_opflow_qnn(self, config):
        """ Test Neural Network Classifier with Opflow QNN (Two Layer QNN)."""

        opt, loss, warm_start, q_i = config

        if q_i == 'statevector':
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == 'bfgs':
            optimizer = L_BFGS_B(maxiter=5)
        else:
            optimizer = COBYLA(maxiter=25)

        num_inputs = 2
        var_form = RealAmplitudes(num_inputs, reps=1)
        qnn = TwoLayerQNN(num_inputs, var_form=var_form, quantum_instance=quantum_instance)

        classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss=loss,
                                             warm_start=warm_start)

        # construct data
        num_samples = 5
        X = np.random.rand(num_samples, num_inputs)  # pylint: disable=invalid-name
        y = 2.0*(np.sum(X, axis=1) <= 1) - 1.0

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        print(score)  # TODO: should involve some criterion (like greater than threshold)

        if warm_start:
            # fit again to data
            classifier.fit(X, y)

    @data(
        # optimizer, loss, warm start, quantum instance
        ('cobyla', 'l1', True, 'statevector'),
        ('cobyla', 'l1', True, 'qasm'),
        ('cobyla', 'l1', True, 'statevector'),
        ('cobyla', 'l1', True, 'qasm'),

        ('cobyla', 'l2', True, 'statevector'),
        ('cobyla', 'l2', True, 'qasm'),
        ('cobyla', 'l2', True, 'statevector'),
        ('cobyla', 'l2', True, 'qasm'),

        ('bfgs', 'l1', True, 'statevector'),
        ('bfgs', 'l1', True, 'qasm'),
        ('bfgs', 'l1', True, 'statevector'),
        ('bfgs', 'l1', True, 'qasm'),

        ('bfgs', 'l2', True, 'statevector'),
        ('bfgs', 'l2', True, 'qasm'),
        ('bfgs', 'l2', True, 'statevector'),
        ('bfgs', 'l2', True, 'qasm'),
    )
    def test_classifier_with_circuit_qnn(self, config):
        """ Test Neural Network Classifier with Opflow QNN (Two Layer QNN)."""

        opt, loss, warm_start, q_i = config

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
        var_form = RealAmplitudes(num_inputs, reps=1)

        # construct circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(2))
        qc.append(var_form, range(2))

        # construct qnn
        def parity(x):
            return '{:b}'.format(x).count('1') % 2
        output_shape = 2
        qnn = CircuitQNN(qc, input_params=feature_map.parameters,
                         weight_params=var_form.parameters,
                         sparse=False,
                         interpret=parity,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)

        # construct classifier
        classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss=loss,
                                             warm_start=warm_start)

        # construct data
        num_samples = 5
        X = np.random.rand(num_samples, num_inputs)  # pylint: disable=invalid-name
        y = 1.0*(np.sum(X, axis=1) <= 1)

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        print(score)  # TODO: should involve some criterion (like greater than threshold)

        if warm_start:
            # fit again to data
            classifier.fit(X, y)

    @data(
        # optimizer, loss, warm start, quantum instance
        ('cobyla', 'statevector'),
        ('cobyla', 'qasm'),

        ('bfgs', 'statevector'),
        ('bfgs', 'qasm'),
    )
    def test_classifier_with_circuit_qnn_and_cross_entropy(self, config):
        """ Test Neural Network Classifier with Opflow QNN (Two Layer QNN)."""

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
        var_form = RealAmplitudes(num_inputs, reps=1)

        # construct circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(2))
        qc.append(var_form, range(2))

        # construct qnn
        def parity(x):
            return '{:b}'.format(x).count('1') % 2
        output_shape = 2
        qnn = CircuitQNN(qc, input_params=feature_map.parameters,
                         weight_params=var_form.parameters,
                         sparse=False,
                         interpret=parity,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)

        # construct classifier - note: CrossEntropy requires eval_probabilities=True!
        classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss=loss,
                                             eval_probabilities=True)

        # construct data
        num_samples = 5
        X = np.random.rand(num_samples, num_inputs)  # pylint: disable=invalid-name
        y = 1.0*(np.sum(X, axis=1) <= 1)
        y = np.array([y, 1-y]).transpose()

        # fit to data
        classifier.fit(X, y)

        # score
        score = classifier.score(X, y)
        print(score)  # TODO: should involve some criterion (like greater than threshold)


if __name__ == '__main__':
    unittest.main()
