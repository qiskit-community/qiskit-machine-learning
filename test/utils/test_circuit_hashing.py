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

"""Tests for the ``circuit_cache_key`` utility (JSON-structural variant)."""
import unittest

from test import QiskitAlgorithmsTestCase
import io
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import qpy

from qiskit_machine_learning.utils.circuit_hash import circuit_cache_key


class TestCircuitCacheKey(QiskitAlgorithmsTestCase):
    """Test the ``circuit_cache_key`` utility function."""

    def setUp(self):
        super().setUp()
        # Simple Bell circuit baseline (no metadata)
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

    def test_stable_for_same_object_multiple_calls(self):
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
        qc_mod.barrier()  # structural op
        k_after = circuit_cache_key(qc_mod)
        self.assertNotEqual(k_before, k_after)

    def test_changes_when_parameters_are_bound(self):
        """Binding parameters changes the key (params are part of the structural encoding)."""
        theta = Parameter("θ")
        qc_param = QuantumCircuit(1)
        qc_param.rx(theta, 0)

        k_unbound = circuit_cache_key(qc_param)
        qc_bound = qc_param.assign_parameters({theta: 0.123})
        k_bound = circuit_cache_key(qc_bound)
        self.assertNotEqual(k_unbound, k_bound)

    def test_numeric_type_normalization(self):
        """NP scalar vs float should yield identical keys."""
        theta = Parameter("t")
        qc_a = QuantumCircuit(1)
        qc_b = QuantumCircuit(1)
        qc_a.rx(theta, 0)
        qc_b.rx(theta, 0)

        k_unbound_a = circuit_cache_key(qc_a)
        k_unbound_b = circuit_cache_key(qc_b)
        self.assertEqual(k_unbound_a, k_unbound_b)

        qc_a_bound = qc_a.assign_parameters({theta: 0.5})
        qc_b_bound = qc_b.assign_parameters({theta: np.float64(0.5)})
        self.assertEqual(circuit_cache_key(qc_a_bound), circuit_cache_key(qc_b_bound))

    @unittest.skip("Global phase is not cached, but keep this as an example of circuit attribute.")
    def test_global_phase_affects_key(self):
        """Global phase is included in the key and should change it."""
        qc0 = self.qc.copy()
        qc1 = self.qc.copy()
        qc1.global_phase = 0.5
        self.assertNotEqual(circuit_cache_key(qc0), circuit_cache_key(qc1))

    def test_name_does_not_affect_key(self):
        """Circuit name is not part of the structural key."""
        qc1 = self.qc.copy()
        qc2 = self.qc.copy()
        qc2.name = "a_different_name"
        self.assertEqual(circuit_cache_key(qc1), circuit_cache_key(qc2))

    def test_metadata_does_not_affect_key(self):
        """Metadata is intentionally excluded; changing it should not change the key."""
        qc1 = self.qc.copy()
        qc2 = self.qc.copy()
        qc2.metadata = {"tag": "A"}
        self.assertEqual(circuit_cache_key(qc1), circuit_cache_key(qc2))

    def test_equal_after_qpy_roundtrip_load(self):
        """QPY load of the same circuit yields an equivalent key (structure preserved)."""
        buf = io.BytesIO()
        qpy.dump([self.qc], buf)
        buf.seek(0)
        loaded = qpy.load(buf)[0]
        self.assertEqual(circuit_cache_key(self.qc), circuit_cache_key(loaded))

    def test_measurement_wiring_affects_key(self):
        """Changing the classical wiring of measurements should change the key."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure(0, 0)
        qc1.measure(1, 1)

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.cx(0, 1)
        # swap classical targets
        qc2.measure(0, 1)
        qc2.measure(1, 0)

        self.assertNotEqual(circuit_cache_key(qc1), circuit_cache_key(qc2))

    def test_operation_order_affects_key(self):
        """Reordering otherwise identical operations changes the key."""
        qc1 = QuantumCircuit(2)
        qc1.rx(0.1, 0)
        qc1.ry(0.2, 1)

        qc2 = QuantumCircuit(2)
        qc2.ry(0.2, 1)
        qc2.rx(0.1, 0)

        self.assertNotEqual(circuit_cache_key(qc1), circuit_cache_key(qc2))
