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
"""Tests for adjusting number of qubits in a feature map / ansatz."""
from test import QiskitMachineLearningTestCase
import itertools
from ddt import ddt, idata, unpack
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.utils import derive_num_qubits_feature_map_ansatz


@ddt
class TestAdjustNumQubits(QiskitMachineLearningTestCase):
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
        """Test when all parameters are ``None``."""
        self.assertRaises(
            QiskitMachineLearningError, derive_num_qubits_feature_map_ansatz, None, None, None
        )

    def test_incompatible_feature_map_ansatz(self):
        """Test when feature map and ansatz are incompatible."""
        self.assertRaises(
            QiskitMachineLearningError,
            derive_num_qubits_feature_map_ansatz,
            None,
            self.properties["z1"],
            self.properties["ra2"],
        )

    @idata(
        itertools.chain(
            itertools.product([1], [None, "z1"], [None, "ra1"]),
            itertools.product([2], [None, "z2"], [None, "ra2"]),
        )
    )
    @unpack
    def test_num_qubits_is_set(self, num_qubits, feature_map, ansatz):
        """Test when the number of qubits is set."""
        feature_map = self.properties.get(feature_map)
        ansatz = self.properties.get(ansatz)

        # derived objects
        num_qubits_der, feature_map_der, ansatz_der = derive_num_qubits_feature_map_ansatz(
            num_qubits, feature_map, ansatz
        )
        self.assertEqual(num_qubits_der, num_qubits)

        self._test_feature_map(feature_map_der, feature_map, num_qubits)
        self._test_ansatz(ansatz_der, num_qubits)

    @idata([(None, "ra1"), (None, "ra2"), ("z1", None), ("z1", "ra1"), ("z2", None), ("z2", "ra2")])
    @unpack
    def test_num_qubits_isnot_set(self, feature_map, ansatz):
        """Test when the number of qubits is not set."""
        ansatz = self.properties.get(ansatz)
        feature_map = self.properties.get(feature_map)

        num_qubits_der, feature_map_der, ansatz_der = derive_num_qubits_feature_map_ansatz(
            None, feature_map, ansatz
        )
        num_qubits = feature_map.num_qubits if feature_map is not None else ansatz.num_qubits

        self.assertEqual(num_qubits_der, num_qubits)
        self._test_feature_map(feature_map_der, feature_map, num_qubits)
        self._test_ansatz(ansatz_der, num_qubits)

    def _test_feature_map(self, feature_map_der, feature_map_org, num_qubits_expected):
        self.assertIsNotNone(feature_map_der)
        self.assertEqual(feature_map_der.num_qubits, num_qubits_expected)

        if feature_map_org is None and num_qubits_expected == 1:
            self.assertIsInstance(feature_map_der, QuantumCircuit)
        if feature_map_org is None and num_qubits_expected == 2:
            self.assertIsInstance(feature_map_der, QuantumCircuit)

    def _test_ansatz(self, ansatz_der, num_qubits_expected):
        self.assertIsNotNone(ansatz_der)
        self.assertEqual(ansatz_der.num_qubits, num_qubits_expected)
        self.assertIsInstance(ansatz_der, QuantumCircuit)
