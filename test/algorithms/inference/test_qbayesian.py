# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import math
import unittest
from test import QiskitMachineLearningTestCase
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.inference.qbayesian import QBayesian
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister

class TestQBayesianInference(QiskitMachineLearningTestCase):
    """Test QBayesianInference Algorithm"""
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598
        # Probabilities
        theta_A = 2 * np.arcsin(np.sqrt(0.25))
        theta_B_nA = 2 * np.arcsin(np.sqrt(0.6))
        theta_B_A = 2 * np.arcsin(np.sqrt(0.7))
        theta_C_nBnA = 2 * np.arcsin(np.sqrt(0.1))
        theta_C_nBA = 2 * np.arcsin(np.sqrt(0.55))
        theta_C_BnA = 2 * np.arcsin(np.sqrt(0.7))
        theta_C_BA = 2 * np.arcsin(np.sqrt(0.9))
        # Random variables
        qrA = QuantumRegister(1, name='A')
        qrB = QuantumRegister(1, name='B')
        qrC = QuantumRegister(1, name='C')
        # Define a 3-qubit quantum circuit
        qcA = QuantumCircuit(qrA, qrB, qrC, name="Bayes net")
        # P(A)
        qcA.ry(theta_A, 0)
        # P(B|-A)
        qcA.x(0)
        qcA.cry(theta_B_nA, qrA, qrB)
        qcA.x(0)
        # P(B|A)
        qcA.cry(theta_B_A, qrA, qrB)
        # P(C|-B,-A)
        qcA.x(0)
        qcA.x(1)
        qcA.mcry(theta_C_nBnA, [qrA[0], qrB[0]], qrC[0])
        qcA.x(0)
        qcA.x(1)
        # P(C|-B,A)
        qcA.x(1)
        qcA.mcry(theta_C_nBA, [qrA[0], qrB[0]], qrC[0])
        qcA.x(1)
        # P(C|B,-A)
        qcA.x(0)
        qcA.mcry(theta_C_BnA, [qrA[0], qrB[0]], qrC[0])
        qcA.x(0)
        # P(C|B,A)
        qcA.mcry(theta_C_BA, [qrA[0], qrB[0]], qrC[0])
        # Quantum Bayesian inference
        self.qbayesian = QBayesian(qcA)


    def test_rejection_sampling(self):
        """Test rejection sampling with different amount of evidence"""
        test_cases = [{'A': 0, 'B': 0}, {'A': 0}, {}]
        true_res = [
            {'000': 2700, '001': 300},
            {'011': 1763, '001': 3504, '010': 13483, '000': 26948},
            {'100': 3042, '110': 31606, '001': 3341, '011': 1731, '111': 15653, '010': 13511, '000': 27109, '101': 4007}
                    ]
        for e in test_cases:
            samples = self.qbayesian.rejectionSampling(evidence=e)
            print(samples)
        #self.assertTrue(np.all(samples>0))
    def test_inference(self):
        test_q_1, test_e_1 = ({'B': 1}, {'A': 1, 'C': 1})
        test_q_2 = {'B': 0}
        true_res = [0.79, 0.21]
        res = []
        samples = []
        # 1. Query
        res.append(self.qbayesian.inference(query=test_q_1, evidence=test_e_1))
        samples.append(self.qbayesian.samples)
        # 2. Query
        res.append(self.qbayesian.inference(query=test_q_2))
        samples.append(self.qbayesian.samples)
        # Correct inference
        self.assertTrue(np.all(np.isclose(true_res, res, rtol=0.05)))
        # No change in samples
        self.assertTrue(samples[0] == samples[1])

    def test_parameter(self):
        ...

if __name__ == "__main__":
    unittest.main()
