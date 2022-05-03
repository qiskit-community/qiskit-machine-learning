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
""" Test Neural Network Regressor """

import os
import tempfile

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms import SerializableModelMixin
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import TwoLayerQNN


@ddt
class TestNeuralNetworkRegressor(QiskitMachineLearningTestCase):
    """Test Neural Network Regressor."""

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

        num_samples = 20
        eps = 0.2

        # pylint: disable=invalid-name
        lb, ub = -np.pi, np.pi
        self.X = (ub - lb) * np.random.rand(num_samples, 1) + lb
        self.y = np.sin(self.X[:, 0]) + eps * (2 * np.random.rand(num_samples) - 1)

    @data(
        # optimizer, loss, quantum instance
        ("cobyla", "statevector", False),
        ("cobyla", "qasm", False),
        ("bfgs", "statevector", False),
        ("bfgs", "qasm", False),
        (None, "statevector", False),
        (None, "qasm", False),
    )
    def test_regressor_with_opflow_qnn(self, config):
        """Test Neural Network Regressor with Opflow QNN (Two Layer QNN)."""
        opt, q_i, cb_flag = config

        num_qubits = 1
        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(num_qubits, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter("y")
        ansatz = QuantumCircuit(num_qubits, name="vf")
        ansatz.ry(param_y, 0)

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

        # construct QNN
        regression_opflow_qnn = TwoLayerQNN(
            num_qubits, feature_map, ansatz, quantum_instance=quantum_instance
        )

        initial_point = np.zeros(ansatz.num_parameters)

        # construct the regressor from the neural network
        regressor = NeuralNetworkRegressor(
            neural_network=regression_opflow_qnn,
            loss="squared_error",
            optimizer=optimizer,
            initial_point=initial_point,
            callback=callback,
        )

        # fit to data
        regressor.fit(self.X, self.y)

        # score the result
        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)

        # callback
        if callback is not None:
            self.assertTrue(all(isinstance(value, float) for value in history["values"]))
            for weights in history["weights"]:
                self.assertEqual(len(weights), regression_opflow_qnn.num_weights)
                self.assertTrue(all(isinstance(weight, float) for weight in weights))

    def test_save_load(self):
        """Tests save and load models."""
        features = np.array([[0, 0], [0.1, 0.1], [0.4, 0.4], [1, 1]])
        labels = np.array([0, 0.1, 0.4, 1])
        num_inputs = 2
        qnn = TwoLayerQNN(
            num_inputs,
            feature_map=ZZFeatureMap(num_inputs),
            ansatz=RealAmplitudes(num_inputs),
            observable=PauliSumOp.from_list([("Z" * num_inputs, 1)]),
            quantum_instance=self.qasm_quantum_instance,
        )
        regressor = NeuralNetworkRegressor(qnn, optimizer=COBYLA())
        regressor.fit(features, labels)

        # predicted labels from the newly trained model
        test_features = np.array([[0.5, 0.5]])
        original_predicts = regressor.predict(test_features)

        # save/load, change the quantum instance and check if predicted values are the same
        file_name = os.path.join(
            tempfile.gettempdir(), "regressor" + str(np.random.randint(1000)) + ".model"
        )
        regressor.save(file_name)
        try:
            regressor_load = NeuralNetworkRegressor.load(file_name)
            loaded_model_predicts = regressor_load.predict(test_features)

            np.testing.assert_array_almost_equal(original_predicts, loaded_model_predicts)

            # test loading warning
            class FakeModel(SerializableModelMixin):
                """Fake model class for test purposes."""

                pass

            with self.assertRaises(TypeError):
                FakeModel.load(file_name)

        finally:
            os.remove(file_name)
