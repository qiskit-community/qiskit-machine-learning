# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Quantum Bayesian Inference """

import unittest
from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.primitives import Sampler

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QBayesian


class TestQBayesianInference(QiskitMachineLearningTestCase):
    """Test QBayesianInference Algorithm"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598
        # Quantum Bayesian inference
        qc = self._create_bayes_net()
        self.qbayesian = QBayesian(qc)

    def _create_bayes_net(self):
        # Probabilities
        theta_a = 2 * np.arcsin(np.sqrt(0.25))
        theta_b_na = 2 * np.arcsin(np.sqrt(0.6))
        theta_b_a = 2 * np.arcsin(np.sqrt(0.7))
        theta_c_nbna = 2 * np.arcsin(np.sqrt(0.1))
        theta_c_nba = 2 * np.arcsin(np.sqrt(0.55))
        theta_c_bna = 2 * np.arcsin(np.sqrt(0.7))
        theta_c_ba = 2 * np.arcsin(np.sqrt(0.9))
        # Random variables
        qr_a = QuantumRegister(1, name="A")
        qr_b = QuantumRegister(1, name="B")
        qr_c = QuantumRegister(1, name="C")
        # Define a 3-qubit quantum circuit
        qc = QuantumCircuit(qr_a, qr_b, qr_c, name="Bayes net")
        # P(A)
        qc.ry(theta_a, 0)
        # P(B|-A)
        qc.x(0)
        qc.cry(theta_b_na, qr_a, qr_b)
        qc.x(0)
        # P(B|A)
        qc.cry(theta_b_a, qr_a, qr_b)
        # P(C|-B,-A)
        qc.x(0)
        qc.x(1)
        qc.mcry(theta_c_nbna, [qr_a[0], qr_b[0]], qr_c[0])
        qc.x(0)
        qc.x(1)
        # P(C|-B,A)
        qc.x(1)
        qc.mcry(theta_c_nba, [qr_a[0], qr_b[0]], qr_c[0])
        qc.x(1)
        # P(C|B,-A)
        qc.x(0)
        qc.mcry(theta_c_bna, [qr_a[0], qr_b[0]], qr_c[0])
        qc.x(0)
        # P(C|B,A)
        qc.mcry(theta_c_ba, [qr_a[0], qr_b[0]], qr_c[0])
        return qc

    def test_rejection_sampling(self):
        """Test rejection sampling with different amount of evidence"""
        test_cases = [{"A": 0, "B": 0}, {"A": 0}, {}]
        true_res = [
            {"000": 0.9, "100": 0.1},
            {"000": 0.36, "100": 0.04, "010": 0.18, "110": 0.42},
            {
                "000": 0.27,
                "001": 0.03375,
                "010": 0.135,
                "011": 0.0175,
                "100": 0.03,
                "101": 0.04125,
                "110": 0.315,
                "111": 0.1575,
            },
        ]
        for evd, res in zip(test_cases, true_res):
            samples = self.qbayesian.rejection_sampling(evidence=evd)
            self.assertTrue(
                np.all(
                    [
                        np.isclose(res[sample_key], sample_val, atol=0.08)
                        for sample_key, sample_val in samples.items()
                    ]
                )
            )

    def test_rejection_sampling_format_res(self):
        """Test rejection sampling with different result format"""
        test_cases = [{"A": 0, "C": 1}, {"C": 1}, {}]
        true_res = [
            {"P(B=0|A=0,C=1)", "P(B=1|A=0,C=1)"},
            {"P(A=0,B=0|C=1)", "P(A=1,B=0|C=1)", "P(A=0,B=1|C=1)", "P(A=1,B=1|C=1)"},
            {
                "P(A=0,B=0,C=0)",
                "P(A=1,B=0,C=0)",
                "P(A=0,B=1,C=0)",
                "P(A=1,B=1,C=0)",
                "P(A=0,B=0,C=1)",
                "P(A=1,B=0,C=1)",
                "P(A=0,B=1,C=1)",
                "P(A=1,B=1,C=1)",
            },
        ]
        for evd, res in zip(test_cases, true_res):
            self.assertTrue(
                res == set(self.qbayesian.rejection_sampling(evidence=evd, format_res=True).keys())
            )

    def test_inference(self):
        """Test inference with different amount of evidence"""
        test_q_1, test_e_1 = ({"B": 1}, {"A": 1, "C": 1})
        test_q_2 = {"B": 0}
        test_q_3 = {}
        test_q_4, test_e_4 = ({"B": 1}, {"A": 0})
        true_res = [0.79, 0.21, 1, 0.6]
        res = []
        samples = []
        # 1. Query basic inference
        res.append(self.qbayesian.inference(query=test_q_1, evidence=test_e_1))
        samples.append(self.qbayesian.samples)
        # 2. Query basic inference
        res.append(self.qbayesian.inference(query=test_q_2))
        samples.append(self.qbayesian.samples)
        # 3. Query marginalized inference
        res.append(self.qbayesian.inference(query=test_q_3))
        samples.append(self.qbayesian.samples)
        # 4. Query marginalized inference
        res.append(self.qbayesian.inference(query=test_q_4, evidence=test_e_4))
        # Correct inference
        self.assertTrue(np.all(np.isclose(true_res, res, atol=0.04)))
        # No change in samples
        self.assertTrue(samples[0] == samples[1])

    def test_parameter(self):
        """Tests parameter of methods"""
        # Test set threshold
        self.qbayesian.threshold = 0.9
        self.qbayesian.rejection_sampling(evidence={"A": 1})
        self.assertTrue(self.qbayesian.threshold == 0.9)
        # Test set limit
        # Not converged
        self.qbayesian.limit = 0
        self.qbayesian.rejection_sampling(evidence={"B": 1})
        self.assertFalse(self.qbayesian.converged)
        self.assertTrue(self.qbayesian.limit == 0)
        # Converged
        self.qbayesian.limit = 1
        self.qbayesian.rejection_sampling(evidence={"B": 1})
        self.assertTrue(self.qbayesian.converged)
        self.assertTrue(self.qbayesian.limit == 1)
        # Test sampler
        sampler = Sampler()
        self.qbayesian.sampler = sampler
        self.qbayesian.inference(query={"B": 1}, evidence={"A": 0, "C": 0})
        self.assertTrue(self.qbayesian.sampler == sampler)
        # Create a quantum circuit with a register that has more than one qubit
        with self.assertRaises(ValueError, msg="No ValueError in constructor with invalid input."):
            QBayesian(QuantumCircuit(QuantumRegister(2, "qr")))
        # Test invalid inference without evidence or generated samples
        with self.assertRaises(ValueError, msg="No ValueError in inference with invalid input."):
            QBayesian(QuantumCircuit(QuantumRegister(1, "qr"))).inference({"A": 0})

    def test_trivial_circuit(self):
        """Tests trivial quantum circuit"""
        # Define rotation angles
        theta_a = 2 * np.arcsin(np.sqrt(0.2))
        theta_b_a = 2 * np.arcsin(np.sqrt(0.9))
        theta_b_na = 2 * np.arcsin(np.sqrt(0.3))
        # Define quantum registers
        qr_a = QuantumRegister(1, name="A")
        qr_b = QuantumRegister(1, name="B")
        # Define a 2-qubit quantum circuit
        qc = QuantumCircuit(qr_a, qr_b, name="Bayes net small")
        qc.ry(theta_a, 0)
        qc.cry(theta_b_a, control_qubit=qr_a, target_qubit=qr_b)
        qc.x(0)
        qc.cry(theta_b_na, control_qubit=qr_a, target_qubit=qr_b)
        qc.x(0)
        # Inference
        self.assertTrue(
            np.all(
                np.isclose(
                    0.1,
                    QBayesian(circuit=qc).inference(query={"B": 0}, evidence={"A": 1}),
                    atol=0.04,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
