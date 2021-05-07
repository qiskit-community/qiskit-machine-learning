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

""" Test Neural Network Regressor """
from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import data, ddt

from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import TwoLayerQNN


@ddt
class TestNeuralNetworkRegressor(QiskitMachineLearningTestCase):
    """Test Neural Network Regressor."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        self.random_seed = 12345
        self.sv_quantum_instance = QuantumInstance(
            Aer.get_backend("statevector_simulator"),
            seed_simulator=self.random_seed,
            seed_transpiler=self.random_seed,
        )
        self.qasm_quantum_instance = QuantumInstance(
            Aer.get_backend("qasm_simulator"),
            shots=100,
            seed_simulator=self.random_seed,
            seed_transpiler=self.random_seed,
        )
        np.random.seed(self.random_seed)

        num_samples = 20
        eps = 0.2

        # pylint: disable=invalid-name
        lb, ub = -np.pi, np.pi
        self.X = (ub - lb) * np.random.rand(num_samples, 1) + lb
        self.y = np.sin(self.X[:, 0]) + eps * (2 * np.random.rand(num_samples) - 1)

    @data(
        # optimizer, loss, quantum instance
        ("cobyla", "statevector"),
        ("cobyla", "qasm"),
        ("bfgs", "statevector"),
        ("bfgs", "qasm"),
    )
    def test_regressor_with_opflow_qnn(self, config):
        """Test Neural Network Regressor with Opflow QNN (Two Layer QNN)."""
        opt, q_i = config

        num_qubits = 1
        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(num_qubits, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple feature map
        param_y = Parameter("y")
        var_form = QuantumCircuit(num_qubits, name="vf")
        var_form.ry(param_y, 0)

        if q_i == "statevector":
            quantum_instance = self.sv_quantum_instance
        else:
            quantum_instance = self.qasm_quantum_instance

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        else:
            optimizer = COBYLA(maxiter=25)

        # construct QNN
        regression_opflow_qnn = TwoLayerQNN(
            num_qubits, feature_map, var_form, quantum_instance=quantum_instance
        )

        # construct the regressor from the neural network
        regressor = NeuralNetworkRegressor(
            neural_network=regression_opflow_qnn, loss="l2", optimizer=optimizer
        )

        # fit to data
        regressor.fit(self.X, self.y)

        # score the result
        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)
