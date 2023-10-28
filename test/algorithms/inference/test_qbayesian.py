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
from qiskit_machine_learning.algorithms.inference.qbayesian import QBayesian
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister

def test_ciruit():
    theta_A = 2 * np.arcsin(np.sqrt(0.25))
    theta_B_nA = 2 * np.arcsin(np.sqrt(0.6))
    theta_B_A = 2 * np.arcsin(np.sqrt(0.7))
    theta_C_nBnA = 2 * np.arcsin(np.sqrt(0.1))
    theta_C_nBA = 2 * np.arcsin(np.sqrt(0.55))
    theta_C_BnA = 2 * np.arcsin(np.sqrt(0.7))
    theta_C_BA = 2 * np.arcsin(np.sqrt(0.9))

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

    qcA.draw('mpl', style='bw', plot_barriers=False, justify='none', fold=-1).show()
    # In the order the qubits were added
    qubits = qcA.qubits
    #print(qubits)
    #print(qcA.num_qubits)
    #print(qcA.qregs)
    qbayesian = QBayesian(qcA)
    evidence = {'A': 0, 'C': 0}
    #a = qcA.qregs
    #print('a: ',a)
    #b = [qcA.qregs[0],qcA.qregs[2]]
    #print('b: ', b)
    #c = list(set(a)-set(b))
    #print(c)
    #ctrls = [qrg for qrg in qcA.qregs if qrg.name in evidence]
    #print(ctrls)
    #for q in c:
    #    print(q)
    #    qcA.mcx(ctrls, q)
    #print(qcA.qregs[0])
    #qcA.h(qcA.qregs[0])
    #qcA.draw('mpl', style='bw', plot_barriers=False, justify='none', fold=-1).show()
    #qc = QuantumCircuit(*qcA.qregs)
    #test_cases = [format(i, '03b') for i in range(8)]
    samples = qbayesian.rejectionSampling(evidence=evidence)

    query = {'B': 1}
    print(qbayesian.inference(query))
    print(samples)

test_ciruit()
