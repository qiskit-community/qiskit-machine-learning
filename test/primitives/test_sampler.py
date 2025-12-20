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

"""Unit tests for the QMLSampler primitive."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from qiskit_machine_learning.primitives import QMLSampler


class TestQMLSampler(unittest.TestCase):
    """Tests for the QMLSampler wrapper."""

    def _assert_prob_dict_close(
        self, got: dict[str, float], ref: dict[str, float], atol: float = 1e-12
    ):
        self.assertEqual(set(got.keys()), set(ref.keys()))
        for k in got:
            assert_allclose(got[k], ref[k], atol=atol, rtol=0.0)

    def test_exact_mode_probabilities_and_dyadic_counts(self):
        sampler = QMLSampler()  # shots=None => exact mode

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        res = sampler.run([qc]).result()[0]
        probs = res.join_data().get_probabilities()
        self._assert_prob_dict_close(probs, {"0": 0.5, "1": 0.5})

        counts = res.join_data().get_counts()
        self.assertEqual(counts, {"0": 1, "1": 1})

        self.assertTrue(res.metadata.get("exact", False))
        self.assertIsNone(res.metadata.get("shots", None))

    def test_exact_mode_ignores_shots_override(self):
        sampler = QMLSampler()  # exact mode

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        r0 = sampler.run([qc]).result()[0].join_data().get_probabilities()
        r1 = sampler.run([qc], shots=1000).result()[0].join_data().get_probabilities()
        self._assert_prob_dict_close(r1, r0)

    def test_exact_mode_join_data_matches_statevector_probabilities(self):
        sampler = QMLSampler()  # exact mode

        qr = QuantumRegister(2, "q")
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(1, "b")
        qc = QuantumCircuit(qr, a, b)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(qr[0], a[0])
        qc.measure(qr[1], b[0])

        res = sampler.run([qc]).result()[0]
        joined = res.join_data(["a", "b"]).get_probabilities()

        unitary = qc.remove_final_measurements(inplace=False)
        ref = Statevector.from_instruction(unitary).probabilities_dict(qargs=[0, 1])

        self._assert_prob_dict_close(joined, ref)

    def test_input_output_parameter_sweep_shape(self):
        sampler = QMLSampler()  # exact mode

        theta = Parameter("theta")
        qc = QuantumCircuit(1, 1)
        qc.ry(theta, 0)
        qc.measure(0, 0)

        params = np.array([[0.0], [np.pi]])  # shape (2, 1) for one parameter
        res = sampler.run([(qc, params)]).result()[0]

        self.assertEqual(res.data.shape, (2,))
        self.assertEqual(res.data.c.shape, (2,))

        p0 = res.data.c.get_probabilities(loc=0)
        p1 = res.data.c.get_probabilities(loc=1)

        ref0 = Statevector.from_instruction(
            qc.assign_parameters({theta: 0.0}, inplace=False).remove_final_measurements(
                inplace=False
            )
        ).probabilities_dict(qargs=[0])
        ref1 = Statevector.from_instruction(
            qc.assign_parameters({theta: np.pi}, inplace=False).remove_final_measurements(
                inplace=False
            )
        ).probabilities_dict(qargs=[0])

        self._assert_prob_dict_close(p0, ref0)
        self._assert_prob_dict_close(p1, ref1)

    def test_sampling_mode_delegates_to_statevector_sampler(self):
        qml = QMLSampler(shots=256, seed=123)
        ref = StatevectorSampler(default_shots=256, seed=123)

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        rq = qml.run([qc]).result()[0].join_data().get_counts()
        rr = ref.run([qc]).result()[0].join_data().get_counts()
        self.assertEqual(rq, rr)


if __name__ == "__main__":
    unittest.main()
