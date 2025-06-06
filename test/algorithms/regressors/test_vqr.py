# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
""" Test Neural Network Regressor with EstimatorQNN."""

import unittest
from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import data, ddt
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.algorithms import VQR


@ddt
class TestVQR(QiskitMachineLearningTestCase):
    """VQR Tests."""

    def setUp(self):
        super().setUp()

        # specify quantum instances
        algorithm_globals.random_seed = 12345

        self.estimator = Estimator()

        num_samples = 20
        eps = 0.2

        # pylint: disable=invalid-name
        lb, ub = -np.pi, np.pi
        rng = np.random.default_rng(101)
        self.X = (ub - lb) * rng.random((num_samples, 1)) + lb
        self.y = np.sin(self.X[:, 0]) + eps * (2 * rng.random(num_samples) - 1)

    @data(
        # optimizer, has ansatz
        ("cobyla", True),
        ("cobyla", False),
        ("bfgs", True),
        ("bfgs", False),
        (None, True),
        (None, False),
    )
    def test_vqr(self, config):
        """Test VQR."""

        opt, has_ansatz = config

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        num_qubits = 1
        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(num_qubits, name="fm")
        feature_map.ry(param_x, 0)

        if has_ansatz:
            param_y = Parameter("y")
            ansatz = QuantumCircuit(num_qubits, name="vf")
            ansatz.ry(param_y, 0)
            initial_point = np.zeros(ansatz.num_parameters)
        else:
            ansatz = None
            # we know it will be RealAmplitudes
            initial_point = np.zeros(4)

        # construct regressor
        regressor = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            estimator=self.estimator,
        )

        # fit to data
        regressor.fit(self.X, self.y)

        # score
        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)

    def test_properties(self):
        """Test properties of VQR."""
        vqr = VQR(num_qubits=2)
        self.assertIsNotNone(vqr.feature_map)
        self.assertIsInstance(vqr.feature_map, QuantumCircuit)
        self.assertEqual(vqr.feature_map.num_qubits, 2)

        self.assertIsNotNone(vqr.ansatz)
        self.assertIsInstance(vqr.ansatz, QuantumCircuit)
        self.assertEqual(vqr.ansatz.num_qubits, 2)

        self.assertEqual(vqr.num_qubits, 2)

    def test_incorrect_observable(self):
        """Test VQR with a wrong observable."""
        with self.assertRaises(ValueError):
            _ = VQR(num_qubits=2, observable=QuantumCircuit(2))

    @data(
        # optimizer, has ansatz
        ("cobyla", True),
        ("cobyla", False),
        ("bfgs", True),
        ("bfgs", False),
        (None, True),
        (None, False),
    )
    def test_vqr_v2(self, config):
        """Test VQR."""

        opt, has_ansatz = config

        if opt == "bfgs":
            optimizer = L_BFGS_B(maxiter=5)
        elif opt == "cobyla":
            optimizer = COBYLA(maxiter=25)
        else:
            optimizer = None

        backend = GenericBackendV2(
            num_qubits=2,
            calibrate_instructions=None,
            pulse_channels=False,
            noise_info=False,
            seed=123,
        )
        session = Session(backend=backend)
        _estimator = EstimatorV2(mode=session)
        pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)

        num_qubits = 1
        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(num_qubits, name="fm")
        feature_map.ry(param_x, 0)

        if has_ansatz:
            param_y = Parameter("y")
            ansatz = QuantumCircuit(num_qubits, name="vf")
            ansatz.ry(param_y, 0)
            initial_point = np.zeros(ansatz.num_parameters)
        else:
            ansatz = None
            # we know it will be RealAmplitudes
            initial_point = np.zeros(4)

        # construct regressor
        regressor = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            estimator=_estimator,
            pass_manager=pass_manager,
        )

        # fit to data
        regressor.fit(self.X, self.y)

        # score
        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
