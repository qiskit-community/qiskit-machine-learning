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
"""Unit tests for the QMLEstimator primitive."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.primitives import QMLEstimator


class TestQMLEstimator(unittest.TestCase):
    """Tests for the QMLEstimator wrapper."""

    def test_exact_mode_is_deterministic_and_ignores_precision(self):
        est = QMLEstimator(default_precision=0.0)

        qc = QuantumCircuit(1)
        qc.h(0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        pub = (qc, [obs])

        r0 = est.run([pub], precision=0.0).result()[0]
        r1 = est.run([pub], precision=0.123).result()[0]

        assert_allclose(r0.data.evs, np.array([0.0]), atol=0.0, rtol=0.0)
        assert_allclose(r0.data.stds, np.array([0.0]), atol=0.0, rtol=0.0)
        assert_allclose(r1.data.evs, r0.data.evs, atol=0.0, rtol=0.0)
        assert_allclose(r1.data.stds, r0.data.stds, atol=0.0, rtol=0.0)

        self.assertTrue(r0.metadata.get("exact", False))
        self.assertEqual(r0.metadata.get("target_precision"), 0.0)

    def test_delegate_mode_matches_statevector_estimator(self):
        qml = QMLEstimator(default_precision=0.25, seed=123)
        ref = StatevectorEstimator(default_precision=0.25, seed=123)

        qc = QuantumCircuit(1)
        qc.x(0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        pub = (qc, [obs])

        rq = qml.run([pub]).result()[0]
        rr = ref.run([pub]).result()[0]

        assert_allclose(rq.data.evs, rr.data.evs)
        assert_allclose(rq.data.stds, rr.data.stds)

    def test_input_output_formats(self):
        est = QMLEstimator(default_precision=0.0)

        qc = QuantumCircuit(1)
        qc.x(0)  # |1>
        z_spo = SparsePauliOp.from_list([("Z", 1.0)])

        # Various observable encodings should be accepted and produce the same output.
        pubs = [
            (qc, [{"Z": 1.0}]),   # mapping encoding
            (qc, ["Z"]),          # label string
            (qc, [z_spo]),        # SparsePauliOp
        ]
        for pub in pubs:
            res = est.run([pub]).result()[0]
            assert_allclose(res.data.evs, np.array([-1.0]), atol=0.0, rtol=0.0)
            assert_allclose(res.data.stds, np.array([0.0]), atol=0.0, rtol=0.0)
            self.assertTrue(res.metadata.get("exact", False))

        # Multiple observables: output should align with the observable list order.
        res_multi = est.run([(qc, ["Z", "I"])]).result()[0]
        assert_allclose(res_multi.data.evs, np.array([-1.0, 1.0]), atol=0.0, rtol=0.0)
        assert_allclose(res_multi.data.stds, np.array([0.0, 0.0]), atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
