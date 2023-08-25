# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
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
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit.circuit.library import PauliFeatureMap, EfficientSU2
from qiskit_machine_learning import QiskitMachineLearningError

from qiskit_machine_learning.circuit.library import QNNCircuit


class TestQNNCircuit(QiskitMachineLearningTestCase):
    """Tests for the ``QNNCircuit`` circuit."""

    def test_construction_before_build(self):
        """Test construction of ``QNNCircuit`` before the circuit is built."""

        circuit = QNNCircuit(num_qubits=2)

        # The properties of the QNNCircuit are set when the class is instantiated.
        with self.subTest("check input configuration before circuit is build"):
            self.assertEqual(circuit.num_qubits, 2)
            self.assertEqual(type(circuit.feature_map), ZZFeatureMap)
            self.assertEqual(circuit.feature_map.num_qubits, 2)
            self.assertEqual(type(circuit.ansatz), RealAmplitudes)
            self.assertEqual(circuit.ansatz.num_qubits, 2)
            self.assertEqual(circuit.num_input_parameters, 2)
            self.assertEqual(circuit.num_weight_parameters, 8)

    def test_construction_fails(self):
        """Test the faulty construction"""

        # If no argument is passed a QiskitMachineLearningError is raised
        # when the class is attempted to be instantiated (before the circuit is built).
        with self.assertRaises(QiskitMachineLearningError):
            QNNCircuit(feature_map=ZZFeatureMap(2), ansatz=RealAmplitudes(1))

        # If no argument is passed a QiskitMachineLearningError is raised
        # when the class is attempted to be instantiated (before the circuit is built).
        with self.assertRaises(QiskitMachineLearningError):
            QNNCircuit()

    def test_num_qubit_construction(self):
        """Test building the ``QNNCircuit`` with number of qubits."""

        circuit = QNNCircuit(1)
        circuit._build()

        # If not otherwise specified, the defaults are a ZFeatureMap/ZZFeatureMap and a
        # RealAmplitudes ansatz.
        with self.subTest("check input configuration after the circuit is build"):
            self.assertEqual(circuit.num_qubits, 1)
            self.assertEqual(type(circuit.feature_map), ZFeatureMap)
            self.assertEqual(circuit.feature_map.num_qubits, 1)
            self.assertEqual(type(circuit.ansatz), RealAmplitudes)
            self.assertEqual(circuit.ansatz.num_qubits, 1)
            self.assertEqual(circuit.num_input_parameters, 1)
            self.assertEqual(circuit.num_weight_parameters, 4)

    def test_feature_map_construction(self):
        """Test building the ``QNNCircuit`` with a feature map"""

        feature_map = PauliFeatureMap(3)
        circuit = QNNCircuit(feature_map=feature_map)
        circuit._build()

        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 3)

        with self.subTest("check feature map type"):
            self.assertEqual(type(circuit.feature_map), PauliFeatureMap)

        with self.subTest("check number of qubits for feature map"):
            self.assertEqual(circuit.feature_map.num_qubits, 3)

        with self.subTest("check number of qubits for ansatz"):
            self.assertEqual(circuit.ansatz.num_qubits, 3)

        with self.subTest("check ansatz type"):
            self.assertEqual(type(circuit.ansatz), RealAmplitudes)

    def test_construction_for_input_missmatch(self):
        """Test the construction of ``QNNCircuit`` for input that does not match."""

        circuit = QNNCircuit(num_qubits=4, feature_map=ZZFeatureMap(3), ansatz=RealAmplitudes(2))

        # If the number of qubits is provided, it overrules the feature map
        # and ansatz settings.
        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 4)

        with self.subTest("check number of qubits for feature map"):
            self.assertEqual(circuit.ansatz.num_qubits, 4)

        with self.subTest("check number of qubits for ansatz"):
            self.assertEqual(circuit.ansatz.num_qubits, 4)

    def test_num_qubit_setter(self):
        """Test the properties after the number of qubits are updated."""

        # Instantiate a QNNCircuit with 3 qubits.
        circuit = QNNCircuit(3)
        # Update the number of qubits to 4.
        circuit.num_qubits = 4

        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 4)
            self.assertEqual(circuit.feature_map.num_qubits, 4)
            self.assertEqual(circuit.ansatz.num_qubits, 4)
            self.assertEqual(circuit.num_input_parameters, 4)
            self.assertEqual(circuit.num_weight_parameters, 16)

    def test_ansatz_setter(self):
        """Test the properties after the ansatz is updated."""
        # Instantiate QNNCircuit 2 qubits a PauliFeatureMap the default ansatz RealAmplitudes
        circuit = QNNCircuit(2, feature_map=PauliFeatureMap(2))
        # Update the ansatz to a 3 qubit "EfficientSU2"
        circuit.ansatz = EfficientSU2(3)

        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual(circuit.feature_map.num_qubits, 3)
            self.assertEqual(circuit.ansatz.num_qubits, 3)
            self.assertEqual(circuit.num_input_parameters, 3)
            self.assertEqual(circuit.num_weight_parameters, 24)
        with self.subTest("check updated ansatz"):
            self.assertEqual(type(circuit.feature_map), PauliFeatureMap)
            self.assertEqual(type(circuit.ansatz), EfficientSU2)

    def test_feature_map_setter(self):
        """Test that the number of qubits cannot be updated by a new ansatz."""

        # Instantiate QNNCircuit 3 qubits and the default feature map ZZFeatureMap and ansatz
        # RealAmplitudes
        circuit = QNNCircuit(3)
        # Update the feature_map to a 1 qubit "EfficientSU2"
        circuit.feature_map = ZFeatureMap(1)

        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 1)
            self.assertEqual(circuit.feature_map.num_qubits, 1)
            self.assertEqual(circuit.ansatz.num_qubits, 1)
            self.assertEqual(circuit.num_input_parameters, 1)
            self.assertEqual(circuit.num_weight_parameters, 4)
        with self.subTest("check updated ansatz"):
            self.assertEqual(type(circuit.feature_map), ZFeatureMap)

    def test_copy(self):
        """Test copy operation for ``QNNCircuit``."""

        circuit = QNNCircuit(8)
        circuit_copy = circuit.copy()

        # make sure that the copied circuit has the same properties as the original one
        self.assertEqual(circuit.num_qubits, circuit_copy.num_qubits)
        self.assertEqual(circuit.feature_map, circuit_copy.feature_map)
        self.assertEqual(circuit.ansatz, circuit_copy.ansatz)


if __name__ == "__main__":
    unittest.main()
