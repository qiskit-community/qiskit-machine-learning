# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test VQC """

from test.ml.common import QiskitMLTestCase
from qiskit.ml.datasets import ad_hoc_data
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ


class TestVQC(QiskitMLTestCase):
    """VQC tests."""

    def test_vqc(self):
        """VQC test."""

        feature_dim = 2  # dimension of each data point
        aqua_globals.random_seed = 10598
        shots = 1024

        _, training_input, test_input, _ = ad_hoc_data(training_size=20,
                                                       test_size=10,
                                                       n=feature_dim,
                                                       gap=0.3,
                                                       plot_data=False)

        optimizer = SPSA(max_trials=100, c0=4.0, skip_calibration=True)
        optimizer.set_options(save_steps=1)
        vqc = VQC(optimizer,
                  SecondOrderExpansion(feature_dimension=feature_dim, depth=2),
                  RYRZ(num_qubits=feature_dim, depth=3),
                  training_input,
                  test_input)
        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=shots,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)

        result = vqc.run(quantum_instance)
        self.assertAlmostEqual(result['testing_accuracy'], 1, delta=0.3)
