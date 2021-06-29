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


if __name__ == "__main__":
    unittest.main()
