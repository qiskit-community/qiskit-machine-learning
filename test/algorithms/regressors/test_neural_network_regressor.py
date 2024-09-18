# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
""" Test Neural Network Regressor """
from __future__ import annotations

import itertools
import os
import tempfile
import unittest
from functools import partial

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, unpack, idata
from scipy.optimize import minimize

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.algorithms import SerializableModelMixin
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import EstimatorQNN

QUANTUM_INSTANCES = ["statevector", "qasm"]
OPTIMIZERS = ["cobyla", "bfgs", "callable", None]
CALLBACKS = [True, False]


@ddt
class TestNeuralNetworkRegressor(QiskitMachineLearningTestCase):
    """Test Neural Network Regressor."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        algorithm_globals.random_seed = 12345

        num_samples = 20
        eps = 0.2

        # pylint: disable=invalid-name
        lb, ub = -np.pi, np.pi
        rng = np.random.default_rng(101)
        self.X = (ub - lb) * rng.random((num_samples, 1)) + lb
        self.y = np.sin(self.X[:, 0]) + eps * (2 * rng.random(num_samples) - 1)

    def _create_regressor(
        self, opt, callback=None
    ) -> tuple[NeuralNetworkRegressor, EstimatorQNN, QuantumCircuit]:
        num_qubits = 1

        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(num_qubits, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter("y")
        ansatz = QuantumCircuit(num_qubits, name="vf")
        ansatz.ry(param_y, 0)

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)  # type: ignore[assignment]
        elif opt == "callable":
            optimizer = partial(
                minimize, method="COBYLA", options={"maxiter": 25}  # type: ignore[assignment]
            )
        else:
            optimizer = None

        # construct QNN
        regression_estimator_qnn = EstimatorQNN(
            circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
        )

        initial_point = np.zeros(ansatz.num_parameters)

        # construct the regressor from the neural network
        regressor = NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=optimizer,
            initial_point=initial_point,
            callback=callback,
        )

        return regressor, regression_estimator_qnn, ansatz

    @idata(itertools.product(OPTIMIZERS, CALLBACKS))
    @unpack
    def test_regressor_with_estimator_qnn(self, opt, cb_flag):
        """Test Neural Network Regressor with Estimator QNN."""
        if cb_flag:
            history = {"weights": [], "values": []}

            def callback(objective_weights, objective_value):
                history["weights"].append(objective_weights)
                history["values"].append(objective_value)

        else:
            callback = None

        regressor, qnn, ansatz = self._create_regressor(opt, callback)

        # fit to data
        regressor.fit(self.X, self.y)

        # score the result
        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)

        # callback
        if callback is not None:
            self.assertTrue(all(isinstance(value, float) for value in history["values"]))
            for weights in history["weights"]:
                self.assertEqual(len(weights), qnn.num_weights)
                self.assertTrue(all(isinstance(weight, float) for weight in weights))

        self.assertIsNotNone(regressor.fit_result)
        self.assertIsNotNone(regressor.weights)
        np.testing.assert_array_equal(regressor.fit_result.x, regressor.weights)
        self.assertEqual(len(regressor.weights), ansatz.num_parameters)

    @idata(OPTIMIZERS)
    def test_warm_start(self, opt):
        """Test VQC when training from a warm start."""
        regressor, _, _ = self._create_regressor(opt)
        regressor.warm_start = True

        # Fit the regressor to the first half of the data.
        num_start = len(self.y) // 2
        regressor.fit(self.X[:num_start, :], self.y[:num_start])
        first_fit_final_point = regressor.weights

        # Fit the regressor to the second half of the data with a warm start.
        regressor.fit(self.X[num_start:, :], self.y[num_start:])
        second_fit_initial_point = regressor._initial_point

        # Check the final optimization point from the first fit was used to start the second fit.
        np.testing.assert_allclose(first_fit_final_point, second_fit_initial_point)

        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)

    def test_save_load(self):
        """Tests save and load models."""
        features = np.array([[0, 0], [0.1, 0.1], [0.4, 0.4], [1, 1]])
        labels = np.array([0, 0.1, 0.4, 1])
        num_inputs = 2

        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)
        qc = QuantumCircuit(num_inputs)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
        regressor = NeuralNetworkRegressor(qnn, optimizer=COBYLA())
        regressor.fit(features, labels)

        # predicted labels from the newly trained model
        test_features = np.array([[0.5, 0.5]])
        original_predicts = regressor.predict(test_features)

        # save/load, change the quantum instance and check if predicted values are the same
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = os.path.join(dir_name, "regressor.model")
            regressor.save(file_name)

            regressor_load = NeuralNetworkRegressor.load(file_name)
            loaded_model_predicts = regressor_load.predict(test_features)

            np.testing.assert_array_almost_equal(original_predicts, loaded_model_predicts)

            # test loading warning
            class FakeModel(SerializableModelMixin):
                """Fake model class for test purposes."""

                pass

            with self.assertRaises(TypeError):
                FakeModel.load(file_name)

    def test_untrained(self):
        """Test untrained regressor."""
        qnn = EstimatorQNN(circuit=QuantumCircuit(2))

        regressor = NeuralNetworkRegressor(qnn)
        with self.assertRaises(QiskitMachineLearningError, msg="regressor.predict()"):
            regressor.predict(np.asarray([]))

        with self.assertRaises(QiskitMachineLearningError, msg="regressor.fit_result"):
            _ = regressor.fit_result

        with self.assertRaises(QiskitMachineLearningError, msg="regressor.weights"):
            _ = regressor.weights

    def test_callback_setter(self):
        """Test the callback setter."""
        num_inputs = 2
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)
        qc = QuantumCircuit(num_inputs)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )

        single_step_opt = SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1)
        regressor = NeuralNetworkRegressor(qnn, optimizer=single_step_opt)

        loss_history = []

        def store_loss(_, loss):
            loss_history.append(loss)

        # use setter for the callback instead of providing in the initialize method
        regressor.callback = store_loss

        features = np.array([[0, 0], [0.1, 0.1], [0.4, 0.4], [1, 1]])
        labels = np.array([0, 0.1, 0.4, 1])
        regressor.fit(features, labels)

        self.assertEqual(len(loss_history), 3)


if __name__ == "__main__":
    unittest.main()
