# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ``circuit_cache_key`` utility."""

from test import QiskitAlgorithmsTestCase
import io
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import qpy

from qiskit_machine_learning.utils import circuit_cache_key


class TestCircuitCacheKey(QiskitAlgorithmsTestCase):
    """Test the ``circuit_cache_key`` utility function."""

    def setUp(self):
        super().setUp()
        # Simple Bell circuit baseline (no metadata to avoid incidental differences)
        qc = QuantumCircuit(2, name="bell")
        qc.h(0)
        qc.cx(0, 1)
        self.qc = qc

    def test_returns_hex_sha256(self):
        """The key should look like a 64-char lowercase hex digest."""
        key = circuit_cache_key(self.qc)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)
        self.assertRegex(key, r"^[0-9a-f]{64}$")

    def test_stable_for_same_circuit_multiple_calls(self):
        """Calling on the same object repeatedly returns the same digest."""
        k1 = circuit_cache_key(self.qc)
        k2 = circuit_cache_key(self.qc)
        self.assertEqual(k1, k2)

    def test_equal_for_structurally_identical_circuits(self):
        """Two freshly constructed but identical circuits have the same key."""
        qc2 = QuantumCircuit(2, name="bell")
        qc2.h(0)
        qc2.cx(0, 1)
        self.assertEqual(circuit_cache_key(self.qc), circuit_cache_key(qc2))

    def test_changes_when_structure_changes(self):
        """Modifying the circuit structure should change the key."""
        k_before = circuit_cache_key(self.qc)
        qc_mod = self.qc.copy()
        qc_mod.barrier()  # structural change
        k_after = circuit_cache_key(qc_mod)
        self.assertNotEqual(k_before, k_after)

    def test_changes_when_parameters_are_bound(self):
        """Binding parameters changes the serialized program and thus the key."""
        theta = Parameter("θ")
        qc_param = QuantumCircuit(1)
        qc_param.rx(theta, 0)

        k_unbound = circuit_cache_key(qc_param)
        qc_bound = qc_param.assign_parameters({theta: 0.123})
        k_bound = circuit_cache_key(qc_bound)
        self.assertNotEqual(k_unbound, k_bound)

    def test_equal_after_qpy_roundtrip_load(self):
        """QPY load of the same circuit yields an equivalent key."""
        buf = io.BytesIO()
        qpy.dump([self.qc], buf)
        buf.seek(0)
        loaded = qpy.load(buf)[0]
        self.assertEqual(circuit_cache_key(self.qc), circuit_cache_key(loaded))

    def test_metadata_affects_key(self):
        """Metadata is part of QPY; changing it should change the key."""
        qc1 = self.qc.copy()
        qc2 = self.qc.copy()
        # Add metadata only to one; QPY includes it, so keys should differ.
        qc2.metadata = {"tag": "A"}
        k1 = circuit_cache_key(qc1)
        k2 = circuit_cache_key(qc2)
        self.assertNotEqual(k1, k2)
