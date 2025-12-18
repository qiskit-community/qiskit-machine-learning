# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ``QNNCircuit`` circuit."""

import unittest
from test import QiskitMachineLearningTestCase

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    pauli_feature_map,
    real_amplitudes,
    zz_feature_map,
)
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.circuit.library import qnn_circuit


class TestQNNCircuitFunction(QiskitMachineLearningTestCase):
    """Tests for the ``qnn_circuit`` circuit."""

    def test_construction(self):
        """Test construction of ``qnn_circuit``."""

        circuit, fm_params, anz_params = qnn_circuit(num_qubits=2)

        with self.subTest("check resultant circuit built"):
            self.assertEqual(circuit.num_qubits, 2)
            self.assertEqual(len(fm_params), 2)
            self.assertEqual(len(anz_params), 8)

    def test_construction_fails(self):
        """Test the faulty construction"""

        # If no argument is passed a QiskitMachineLearningError is raised
        with self.assertRaises(QiskitMachineLearningError):
            qnn_circuit(feature_map=zz_feature_map(2), ansatz=real_amplitudes(1))

        # If no argument is passed a QiskitMachineLearningError is raised
        with self.assertRaises(QiskitMachineLearningError):
            qnn_circuit()

    def test_num_qubit_construction(self):
        """Test building the ``qnn_circuit`` with number of qubits."""

        circuit, fm_params, anz_params = qnn_circuit(1)

        # If not otherwise specified, the defaults are a ZFeatureMap/ZZFeatureMap and a
        # RealAmplitudes ansatz.
        with self.subTest("check input configuration after the circuit is build"):
            self.assertEqual(circuit.num_qubits, 1)
            self.assertEqual(type(circuit), QuantumCircuit)
            self.assertEqual(len(fm_params), 1)
            self.assertEqual(len(anz_params), 4)

    def test_feature_map_construction(self):
        """Test building the ``qnn_circuit`` with a feature map"""

        feature_map = pauli_feature_map(3)
        circuit, fm_params, anz_params = qnn_circuit(feature_map=feature_map)

        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual(len(fm_params), 3)
            self.assertEqual(len(anz_params), 12)

    def test_construction_for_input_mismatch(self):
        """Test the construction of ``qnn_circuit`` for input that does not match fails."""

        with self.assertRaises(QiskitMachineLearningError):
            qnn_circuit(num_qubits=4, feature_map=zz_feature_map(3), ansatz=real_amplitudes(2))