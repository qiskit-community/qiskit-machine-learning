# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2024, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for derive_num_qubits_feature_map_ansatz."""

from test import QiskitMachineLearningTestCase
import itertools
from ddt import ddt, idata, unpack
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.utils import derive_num_qubits_feature_map_ansatz


@ddt
class TestDeriveNumQubits(QiskitMachineLearningTestCase):
    """Tests for the derive_num_qubits_feature_map_ansatz function."""

    def setUp(self) -> None:
        super().setUp()
        self.properties = {
            "z1": z_feature_map(1),
            "z2": z_feature_map(2),
            "ra1": real_amplitudes(1),
            "ra2": real_amplitudes(2),
        }

    def test_all_none(self):
        """All parameters None must raise."""
        self.assertRaises(
            QiskitMachineLearningError, derive_num_qubits_feature_map_ansatz, None, None, None
        )

    def test_mismatch_feature_map_ansatz(self):
        """feature_map and ansatz with different qubit counts must raise."""
        self.assertRaises(
            QiskitMachineLearningError,
            derive_num_qubits_feature_map_ansatz,
            None,
            self.properties["z1"],
            self.properties["ra2"],
        )

    def test_mismatch_num_qubits_feature_map(self):
        """num_qubits != feature_map.num_qubits must raise."""
        self.assertRaises(
            QiskitMachineLearningError,
            derive_num_qubits_feature_map_ansatz,
            2,
            self.properties["z1"],
            None,
        )

    def test_mismatch_num_qubits_ansatz(self):
        """num_qubits != ansatz.num_qubits must raise."""
        self.assertRaises(
            QiskitMachineLearningError,
            derive_num_qubits_feature_map_ansatz,
            2,
            None,
            self.properties["ra1"],
        )

    @idata(
        itertools.chain(
            itertools.product([1], [None, "z1"], [None, "ra1"]),
            itertools.product([2], [None, "z2"], [None, "ra2"]),
        )
    )
    @unpack
    def test_num_qubits_is_set(self, num_qubits, feature_map, ansatz):
        """When num_qubits is set and inputs agree."""
        feature_map = self.properties.get(feature_map)
        ansatz = self.properties.get(ansatz)

        nq, fm, anz = derive_num_qubits_feature_map_ansatz(num_qubits, feature_map, ansatz)
        self.assertEqual(nq, num_qubits)
        self.assertEqual(fm.num_qubits, num_qubits)
        self.assertEqual(anz.num_qubits, num_qubits)
        self.assertIsInstance(fm, QuantumCircuit)
        self.assertIsInstance(anz, QuantumCircuit)

    @idata([(None, "ra1"), (None, "ra2"), ("z1", None), ("z1", "ra1"), ("z2", None), ("z2", "ra2")])
    @unpack
    def test_num_qubits_isnot_set(self, feature_map, ansatz):
        """When num_qubits is None and inputs determine the count."""
        feature_map = self.properties.get(feature_map)
        ansatz = self.properties.get(ansatz)
        expected = feature_map.num_qubits if feature_map is not None else ansatz.num_qubits

        nq, fm, anz = derive_num_qubits_feature_map_ansatz(None, feature_map, ansatz)
        self.assertEqual(nq, expected)
        self.assertEqual(fm.num_qubits, expected)
        self.assertEqual(anz.num_qubits, expected)
        self.assertIsInstance(fm, QuantumCircuit)
        self.assertIsInstance(anz, QuantumCircuit)
