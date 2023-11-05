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
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator

"""Quantum Bayesian Inference"""

class QBayesian:
    r"""
    Implements Quantum Bayesian Inference algorithm. The algorithm has been developed in [1].

    **References**
        [1]: Low, Guang Hao, Theodore J. Yoder, and Isaac L. Chuang.
        "Quantum inference on Bayesian networks." Physical Review A 89.6 (2014): 062315.


    Usage:
    ------
    To use the `QBayesian` class, instantiate it with a quantum circuit that represents the Bayesian network.
    You can then use the `inference` method to estimate probabilities given evidence, optionally using
    rejection sampling and Grover's algorithm for amplification.

    Example:
    --------

    # Define a quantum circuit
    qc = QuantumCircuit(...)

    # Initialize the QBayesian class with the circuit
    qb = QBayesian(qc)

    # Perform inference
    result = qb.inference(query={...}, evidence={...})

    print("Probability of query given evidence:", result)
    """

    # Discrete quantum Bayesian network
    def __init__(self, circuit: QuantumCircuit):
        """
        Run the provided quantum circuit on the Aer simulator backend.

        Parameters:
            - circuit: The quantum circuit to be executed.
            Every r.v. should be assigned exactly one register of one distinct qubit.

        """
        # Test valid input
        for qrg in circuit.qregs:
            if qrg.size>1:
                raise ValueError("Every register needs to be mapped to exactly one unique qubit")
        # Initialize QBayesian
        self.circ = circuit
        # Label of register mapped to its qubit
        self.label2qubit = {qrg.name: qrg[0] for qrg in self.circ.qregs}
        # Label of register mapped to its qubit index
        self.label2qidx = {qrg.name: idx for idx, qrg in enumerate(self.circ.qregs)}
        self.samples = {}

    def getGroverOp(self, evidence: dict) -> GroverOperator:
        """
        Constructs a Grover operator based on the provided evidence. The evidence is used to determine
        the "good states" that the returned Grover operator can amplify.
        """
        # Evidence to reversed qubit index sorted by index
        num_qubits = self.circ.num_qubits
        e2idx = sorted(
            [(num_qubits-self.label2qidx[e_key]-1, e_val) for e_key, e_val in evidence.items()], key=lambda x: x[0]
        )
        # Binary format of good states
        num_evd = len(e2idx)
        bin_str = [format(i, f'0{(num_qubits-num_evd)}b') for i in range(2**(num_qubits-num_evd))]
        # Get good states
        good_states = []
        for b in bin_str:
            for e_idx, e_val in e2idx:
                b = b[:e_idx]+str(e_val)+b[e_idx:]
            good_states.append(b)
        # Get statevector by transform good states like 010 regarding its idx (2+1=3) of statevector to 1 and o/w to 0
        oracle = Statevector([int(format(i, f'0{num_qubits}b') in good_states) for i in range(2**num_qubits)])
        return GroverOperator(oracle, state_preparation=self.circ)

    def run_circuit(self, circuit: QuantumCircuit, shots=100_000) -> dict:
        """ Run the provided quantum circuit for the number of shots on the Aer simulator backend. """
        # Get the Aer simulator backend
        simulator_backend = AerSimulator()
        # Transpile the circuit for the given backend
        transpiled_circuit = transpile(circuit, simulator_backend)
        # Run the transpiled circuit on the simulator
        job = simulator_backend.run(transpiled_circuit, shots=shots)
        result = job.result()
        # Get the counts of quantum state results
        counts = result.get_counts(transpiled_circuit)
        # Convert counts to relative counts (probabilities)
        relative_counts = {state: count / shots for state, count in counts.items()}
        return relative_counts

    def powerGrover(self, groverOp: GroverOperator, evidence: dict, k: int) -> (GroverOperator, set):
        """
        Applies the Grover operator to the quantum circuit 2^k times. It measures the evidence qubits and returns a
        tuple containing the updated quantum circuit and a set of the measured evidence qubits.
        """
        # Create circuit
        qc = QuantumCircuit(*self.circ.qregs)
        qc.append(self.circ, self.circ.qregs)
        # Apply grover operator 2^k times
        qcGrover = QuantumCircuit(*self.circ.qregs)
        qcGrover.append(groverOp, self.circ.qregs)
        qcGrover = qcGrover.power(2 ** k)
        qc.append(qcGrover, self.circ.qregs)
        # Add quantum circuit for measuring
        qc_measure = QuantumCircuit(*self.circ.qregs)
        qc_measure.append(qc, self.circ.qregs)
        # Create a classical register with the size of the evidence
        measurement_ecr = ClassicalRegister(len(evidence))
        qc_measure.add_register(measurement_ecr)
        # Map the evidence qubits to the classical bits and measure them
        evidence_qubits = [self.label2qubit[e_key] for e_key in evidence]
        qc_measure.measure([q for q in evidence_qubits], measurement_ecr)
        # Run the circuit with the Grover operator and measurements
        e_samples = self.run_circuit(qc_measure, shots = 1024)
        E_count = {self.label2qubit[e]: 0 for e in evidence}
        for e_sample_key, e_sample_val in e_samples.items():
            # Go through reverse binary that matches order of qubits
            for i, char in enumerate(e_sample_key[::-1]):
                if int(char) == 1:
                    E_count[evidence_qubits[i]] += e_sample_val
                else:
                    E_count[evidence_qubits[i]] += -e_sample_val
        # Assign to every qubit if it is more often measured 1 -> 1 o/w 0
        E = {(e_count_key, int(e_count_val >= 0)) for e_count_key, e_count_val in E_count.items()}
        return qc, E

    def rejectionSampling(self, evidence: dict, shots: int = 100000) -> dict:

        # If evidence is empty
        if len(evidence) == 0:
            # Create circuit
            qc = QuantumCircuit(*self.circ.qregs)
            qc.append(self.circ, self.circ.qregs)
            # Measure
            qc.measure_all()
            # Run circuit
            samples = self.run_circuit(qc, shots=shots)
            return samples
        else:
            # Get grover operator if evidence not empty
            groverOp = self.getGroverOp(evidence)
            # Amplitude amplification
            e = {(self.label2qubit[e_key], e_val) for e_key, e_val in evidence.items()}
            E = {}
            k = -1
            # If the measurement of the evidence qubits matches the evidence stop
            while (e != E) and (k < 10):
                # Increment power
                k += 1
                # Create circuit with 2^k times grover operator
                qc, E = self.powerGrover(groverOp=groverOp, evidence=evidence, k=k)

        # Create a classical register with the size of the evidence
        measurement_qcr = ClassicalRegister(self.circ.num_qubits-len(evidence))
        qc.add_register(measurement_qcr)
        # Map the query qubits to the classical bits and measure them
        query_qubits = [(label, self.label2qidx[label], qubit) for label, qubit in self.label2qubit.items() if label not in evidence]
        query_qubits_sorted = sorted(query_qubits, key=lambda x: x[1])
        # Measure query variables and return their count
        qc.measure([q[2] for q in query_qubits_sorted], measurement_qcr)
        # Run circuit
        counts = self.run_circuit(qc, shots=shots)
        # Build default string with evidence
        query_string = ''
        varIdxSorted = [label for label, _ in sorted(self.label2qidx.items(), key=lambda x: x[1], reverse=True)]
        for var in varIdxSorted:
            if var in evidence:
                query_string += str(evidence[var])
            else:
                query_string += 'q'
        # Retrieve valid samples
        self.samples = {}
        # Replace placeholder q with query variables from samples
        for key, val in counts.items():
            query = query_string
            for char in key:
                query = query.replace('q', char, 1)
            self.samples[query] = val
        return self.samples


    def inference(self, query: dict, evidence: dict=None, shots: int=100000) -> float:
        """
        - query: The query variables. If Q is a real subset of X\E, it will be marginalized.
        - evidence: Provide evidence if rejection sampling should be executed. If you want to indicate no evidence
        insert an empty list. If you want to indicate no new evidence keep this variable empty.
        """
        if evidence is not None:
            self.rejectionSampling(evidence, shots)
        else:
            if not self.samples:
                raise ValueError("Provide evidence for rejection sampling or indicate no evidence with empty list")
        # Get sorted indices of query qubits
        query_indices = [(self.label2qidx[q_key], q_val) for q_key, q_val in query.items()]
        query_indices_sorted = sorted(query_indices, key=lambda x: x[0], reverse=True)
        # Get probability of query
        res = 0
        for sample_key, sample_val in self.samples.items():
            add = True
            for q_idx, q_val in query_indices_sorted:
                if int(sample_key[q_idx]) != q_val:
                    add = False
                    break
            if add:
                res += sample_val
        return res






