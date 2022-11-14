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

import unittest
import warnings

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import data, ddt
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import QuantumInstance, algorithm_globals, optionals

from qiskit_machine_learning.algorithms import VQR


@ddt
class TestVQR(QiskitMachineLearningTestCase):
    """VQR Tests."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # specify quantum instances
        algorithm_globals.random_seed = 12345
        from qiskit_aer import Aer

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

    def tearDown(self) -> None:
        super().tearDown()
        warnings.filterwarnings("always", category=PendingDeprecationWarning)
        warnings.filterwarnings("always", category=DeprecationWarning)

    @data(
        # optimizer, loss, quantum instance
        ("cobyla", "statevector", True),
        ("cobyla", "qasm", True),
        ("cobyla", "statevector", False),
        ("cobyla", "qasm", False),
        ("bfgs", "statevector", True),
        ("bfgs", "qasm", True),
        ("bfgs", "statevector", False),
        ("bfgs", "qasm", False),
        (None, "statevector", True),
        (None, "qasm", True),
        (None, "statevector", False),
        (None, "qasm", False),
    )
    def test_vqr(self, config):
        """Test VQR."""

        opt, q_i, has_ansatz = config

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
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )

        # fit to data
        regressor.fit(self.X, self.y)

        # score
        score = regressor.score(self.X, self.y)
        self.assertGreater(score, 0.5)

    def test_properties(self):
        """Test properties of VQR."""
        vqr = VQR(num_qubits=2, quantum_instance=self.qasm_quantum_instance)
        self.assertIsNotNone(vqr.feature_map)
        self.assertIsInstance(vqr.feature_map, ZZFeatureMap)
        self.assertEqual(vqr.feature_map.num_qubits, 2)

        self.assertIsNotNone(vqr.ansatz)
        self.assertIsInstance(vqr.ansatz, RealAmplitudes)
        self.assertEqual(vqr.ansatz.num_qubits, 2)

        self.assertEqual(vqr.num_qubits, 2)

    def test_incorrect_observable(self):
        """Test VQR with a wrong observable."""
        with self.assertRaises(ValueError):
            _ = VQR(
                num_qubits=2,
                quantum_instance=self.qasm_quantum_instance,
                observable=SparsePauliOp.from_list([("Z" * 2, 1)]),
            )


if __name__ == "__main__":
    unittest.main()
