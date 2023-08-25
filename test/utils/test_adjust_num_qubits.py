# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for adjusting number of qubits in a feature map / ansatz."""
import itertools

from test import QiskitMachineLearningTestCase

from ddt import idata, unpack, ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes, ZZFeatureMap

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.utils import derive_num_qubits_feature_map_ansatz


@ddt
class TestAdjustNumQubits(QiskitMachineLearningTestCase):
    """Tests for the derive_num_qubits_feature_map_ansatz function."""

    def setUp(self) -> None:
        super().setUp()
        self.properties = {
            "z1": ZFeatureMap(1),
            "z2": ZFeatureMap(2),
            "ra1": RealAmplitudes(1),
            "ra2": RealAmplitudes(2),
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

    def test_no_adjustment(self):
        """Test when no adjustment can be made."""
        self.assertRaises(
            QiskitMachineLearningError,
            derive_num_qubits_feature_map_ansatz,
            2,
            QuantumCircuit(1),
            None,
        )
        self.assertRaises(
            QiskitMachineLearningError,
            derive_num_qubits_feature_map_ansatz,
            2,
            None,
            QuantumCircuit(1),
        )

    @idata(itertools.product([1, 2], [None, "z1", "z2"], [None, "ra1", "ra2"]))
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
            self.assertIsInstance(feature_map_der, ZFeatureMap)
        if feature_map_org is None and num_qubits_expected == 2:
            self.assertIsInstance(feature_map_der, ZZFeatureMap)

    def _test_ansatz(self, ansatz_der, num_qubits_expected):
        self.assertIsNotNone(ansatz_der)
        self.assertEqual(ansatz_der.num_qubits, num_qubits_expected)
        self.assertIsInstance(ansatz_der, RealAmplitudes)
