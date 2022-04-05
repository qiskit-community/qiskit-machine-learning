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

import os
import tempfile
import unittest
from test import QiskitMachineLearningTestCase

from typing import Tuple, Optional, Callable

import numpy as np
import scipy.sparse
from ddt import ddt, data, idata
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, Optimizer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms import SerializableModelMixin
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN, NeuralNetwork
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
        ("cobyla", "absolute_error", "statevector", False),
        ("cobyla", "absolute_error", "qasm", False),
        ("cobyla", "squared_error", "statevector", False),
        ("cobyla", "squared_error", "qasm", False),
        ("bfgs", "absolute_error", "statevector", False),
        ("bfgs", "absolute_error", "qasm", False),
        ("bfgs", "squared_error", "statevector", False),
        ("bfgs", "squared_error", "qasm", False),
        (None, "absolute_error", "statevector", False),
        (None, "absolute_error", "qasm", False),
        (None, "squared_error", "statevector", False),
        (None, "squared_error", "qasm", False),
        ("cobyla", "absolute_error", "statevector", True),
        ("cobyla", "absolute_error", "qasm", True),
        ("cobyla", "squared_error", "statevector", True),
        ("cobyla", "squared_error", "qasm", True),
        ("bfgs", "absolute_error", "statevector", True),
        ("bfgs", "absolute_error", "qasm", True),
        ("bfgs", "squared_error", "statevector", True),
        ("bfgs", "squared_error", "qasm", True),
        (None, "absolute_error", "statevector", True),
        (None, "absolute_error", "qasm", True),
        (None, "squared_error", "statevector", True),
        (None, "squared_error", "qasm", True),
    )
    def test_classifier_with_opflow_qnn(self, config):
        """Test Neural Network Classifier with Opflow QNN (Two Layer QNN)."""

        opt, loss, q_i, cb_flag = config

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

        if cb_flag is True:
            history = {"weights": [], "values": []}

            def callback(objective_weights, objective_value):
                history["weights"].append(objective_weights)
                history["values"].append(objective_value)

        else:
            callback = None

        num_inputs = 2
        ansatz = RealAmplitudes(num_inputs, reps=1)
        qnn = TwoLayerQNN(num_inputs, ansatz=ansatz, quantum_instance=quantum_instance)
        initial_point = np.array([0.5] * ansatz.num_parameters)

        classifier = NeuralNetworkClassifier(
            qnn, optimizer=optimizer, loss=loss, initial_point=initial_point, callback=callback
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

        # callback
        if callback is not None:
            self.assertTrue(all(isinstance(value, float) for value in history["values"]))
            for weights in history["weights"]:
                self.assertEqual(len(weights), qnn.num_weights)
                self.assertTrue(all(isinstance(weight, float) for weight in weights))

    def _create_circuit_qnn(self, quantum_instance: QuantumInstance) -> Tuple[CircuitQNN, int, int]:
        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(2))
        qc.append(ansatz, range(2))

        # construct qnn
        def parity(x):
            return f"{x:b}".count("1") % 2

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

        return qnn, num_inputs, ansatz.num_parameters

    def _generate_data(self, num_inputs: int) -> Tuple[np.ndarray, np.ndarray]:
        # construct data
        num_samples = 5
        features = algorithm_globals.random.random((num_samples, num_inputs))
        labels = 1.0 * (np.sum(features, axis=1) <= 1)

        return features, labels

    def _create_classifier(
        self,
        qnn: NeuralNetwork,
        num_parameters: int,
        optimizer: Optimizer,
        loss: str,
        callback: Optional[Callable[[np.ndarray, float], None]] = None,
        one_hot: bool = False,
    ):
        initial_point = np.array([0.5] * num_parameters)

        # construct classifier
        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=optimizer,
            loss=loss,
            one_hot=one_hot,
            initial_point=initial_point,
            callback=callback,
        )
        return classifier

    @data(
        # optimizer, loss, quantum instance
        ("cobyla", "absolute_error", "statevector", False),
        ("cobyla", "absolute_error", "qasm", False),
        ("cobyla", "squared_error", "statevector", False),
        ("cobyla", "squared_error", "qasm", False),
        ("bfgs", "absolute_error", "statevector", False),
        ("bfgs", "absolute_error", "qasm", False),
        ("bfgs", "squared_error", "statevector", False),
        ("bfgs", "squared_error", "qasm", False),
        (None, "absolute_error", "statevector", False),
        (None, "absolute_error", "qasm", False),
        (None, "squared_error", "statevector", False),
        (None, "squared_error", "qasm", False),
        ("cobyla", "absolute_error", "statevector", True),
        ("cobyla", "absolute_error", "qasm", True),
        ("cobyla", "squared_error", "statevector", True),
        ("cobyla", "squared_error", "qasm", True),
        ("bfgs", "absolute_error", "statevector", True),
        ("bfgs", "absolute_error", "qasm", True),
        ("bfgs", "squared_error", "statevector", True),
        ("bfgs", "squared_error", "qasm", True),
        (None, "absolute_error", "statevector", True),
        (None, "absolute_error", "qasm", True),
        (None, "squared_error", "statevector", True),
        (None, "squared_error", "qasm", True),
    )
    def test_classifier_with_circuit_qnn(self, config):
        """Test Neural Network Classifier with Circuit QNN."""

        opt, loss, q_i, cb_flag = config

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

        if cb_flag is True:
            history = {"weights": [], "values": []}

            def callback(objective_weights, objective_value):
                history["weights"].append(objective_weights)
                history["values"].append(objective_value)

        else:
            callback = None

        qnn, num_inputs, num_parameters = self._create_circuit_qnn(quantum_instance)

        classifier = self._create_classifier(qnn, num_parameters, optimizer, loss, callback)

        features, labels = self._generate_data(num_inputs)

        # fit to data
        classifier.fit(features, labels)

        # score
        score = classifier.score(features, labels)
        self.assertGreater(score, 0.5)

        # callback
        if callback is not None:
            self.assertTrue(all(isinstance(value, float) for value in history["values"]))
            for weights in history["weights"]:
                self.assertEqual(len(weights), qnn.num_weights)
                self.assertTrue(all(isinstance(weight, float) for weight in weights))

    @data(
        # optimizer, quantum instance
        ("cobyla", "statevector", False),
        ("cobyla", "qasm", False),
        ("bfgs", "statevector", False),
        ("bfgs", "qasm", False),
        (None, "statevector", False),
        (None, "qasm", False),
    )
    def test_classifier_with_circuit_qnn_and_cross_entropy(self, config):
        """Test Neural Network Classifier with Circuit QNN and Cross Entropy loss."""

        opt, q_i, cb_flag = config

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

        if cb_flag is True:
            history = {"weights": [], "values": []}

            def callback(objective_weights, objective_value):
                history["weights"].append(objective_weights)
                history["values"].append(objective_value)

        else:
            callback = None

        loss = CrossEntropyLoss()

        qnn, num_inputs, num_parameters = self._create_circuit_qnn(quantum_instance)

        classifier = self._create_classifier(qnn, num_parameters, optimizer, loss, callback, True)

        features, labels = self._generate_data(num_inputs)
        labels = np.array([labels, 1 - labels]).transpose()

        # fit to data
        classifier.fit(features, labels)

        # score
        score = classifier.score(features, labels)
        self.assertGreater(score, 0.5)

        # callback
        if callback is not None:
            self.assertTrue(all(isinstance(value, float) for value in history["values"]))
            for weights in history["weights"]:
                self.assertEqual(len(weights), qnn.num_weights)
                self.assertTrue(all(isinstance(weight, float) for weight in weights))

    @data(
        # one-hot, loss
        (True, "absolute_error"),
        (True, "squared_error"),
        (True, "cross_entropy"),
        (False, "absolute_error"),
        (False, "squared_error"),
    )
    def test_categorical_data(self, config):
        """Test categorical labels using QNN"""

        one_hot, loss = config

        quantum_instance = self.sv_quantum_instance
        optimizer = L_BFGS_B(maxiter=5)

        qnn, num_inputs, num_parameters = self._create_circuit_qnn(quantum_instance)

        classifier = self._create_classifier(qnn, num_parameters, optimizer, loss, one_hot=one_hot)

        features, labels = self._generate_data(num_inputs)
        labels = labels.astype(str)
        # convert to categorical
        labels[labels == "0.0"] = "A"
        labels[labels == "1.0"] = "B"

        # fit to data
        classifier.fit(features, labels)

        # score
        score = classifier.score(features, labels)
        self.assertGreater(score, 0.5)

    def test_sparse_arrays(self):
        """Tests classifier with sparse arrays as features and labels."""
        for quantum_instance in [self.sv_quantum_instance, self.qasm_quantum_instance]:
            for loss in ["squared_error", "absolute_error", "cross_entropy"]:
                with self.subTest(f"quantum_instance: {quantum_instance}, loss: {loss}"):
                    self._test_sparse_arrays(quantum_instance, loss)

    def _test_sparse_arrays(self, quantum_instance: QuantumInstance, loss: str):
        optimizer = L_BFGS_B(maxiter=5)

        qnn, _, num_parameters = self._create_circuit_qnn(quantum_instance)

        classifier = self._create_classifier(qnn, num_parameters, optimizer, loss, one_hot=True)

        features = scipy.sparse.csr_matrix([[0, 0], [1, 1]])
        labels = scipy.sparse.csr_matrix([[1, 0], [0, 1]])

        # fit to data
        classifier.fit(features, labels)

        # score
        score = classifier.score(features, labels)
        self.assertGreater(score, 0.5)

    @idata(["opflow", "circuit_qnn"])
    def test_save_load(self, qnn_type):
        """Tests save and load models."""
        features = np.array([[0, 0], [0.1, 0.2], [1, 1], [0.9, 0.8]])

        if qnn_type == "opflow":
            labels = np.array([-1, -1, 1, 1])

            num_qubits = 2
            ansatz = RealAmplitudes(num_qubits, reps=1)
            qnn = TwoLayerQNN(
                num_qubits, ansatz=ansatz, quantum_instance=self.qasm_quantum_instance
            )
            num_parameters = ansatz.num_parameters
        elif qnn_type == "circuit_qnn":
            labels = np.array([0, 0, 1, 1])
            qnn, _, num_parameters = self._create_circuit_qnn(self.qasm_quantum_instance)
        else:
            raise ValueError(f"Unsupported QNN type: {qnn_type}")

        classifier = self._create_classifier(
            qnn, num_parameters=num_parameters, optimizer=COBYLA(), loss="squared_error"
        )
        classifier.fit(features, labels)

        # predicted labels from the newly trained model
        test_features = np.array([[0.2, 0.1], [0.8, 0.9]])
        original_predicts = classifier.predict(test_features)

        # save/load, change the quantum instance and check if predicted values are the same
        file_name = os.path.join(tempfile.gettempdir(), "classifier.model")
        classifier.save(file_name)
        try:
            classifier_load = NeuralNetworkClassifier.load(file_name)
            loaded_model_predicts = classifier_load.predict(test_features)

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
