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
from qiskit import Aer, QuantumCircuit, transpile, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import GroverOperator

class QBayesian:

    # Discrete quantum Bayesian network
    def __init__(self, circuit: QuantumCircuit):
        """
        Run the provided quantum circuit on the Aer simulator backend.

        Parameters:
            - circuit: The quantum circuit to be executed.
            Every r.v. should be assigned exactly one register of one distinct qubit.

        """
        self.circ = circuit
        # Label of register mapped to its qubit
        self.label2qubit = {qrg.name: qrg[0] for qrg in self.circ.qregs}
        # Label of register mapped to its qubit index
        self.label2qidx = {qrg.name: idx for idx, qrg in enumerate(self.circ.qregs)}
        self.samples = {}

    def getGroverOp(self, evidence):
        """
        Creates Amplification Problem

        evidence: ...
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
        print(bin_str)
        for b in bin_str:
            for e_idx, e_val in e2idx:
                b = b[:e_idx]+str(e_val)+b[e_idx:]
            good_states.append(b)
        print(evidence)
        print("GOOD states")
        print(good_states)
        # Get statevector by transform good states like 010 regarding its idx (2+1=3) of statevector to 1 and o/w to 0
        print([int(format(i, f'0{num_qubits}b') in good_states) for i in range(2**num_qubits)])
        oracle = Statevector([int(format(i, f'0{num_qubits}b') in good_states) for i in range(2**num_qubits)])
        return GroverOperator(oracle, state_preparation=self.circ)


    def rejectionSampling(self, evidence, shots: int=None, grover_iter=None, backend=None):
        def run_circuit(circuit, shots=100_000):
            # TODO: needed??
            """
            Run the provided quantum circuit on the Aer simulator backend.

            Parameters:
            - circuit: The quantum circuit to be executed.
            - shots (default=10,000): The number of times the circuit is executed.

            Returns:
            - counts: A dictionary with the counts of each quantum state result.
            """
            # Get the Aer simulator backend
            simulator_backend = Aer.get_backend('aer_simulator')

            # Transpile the circuit for the given backend
            transpiled_circuit = transpile(circuit, simulator_backend)

            # Run the transpiled circuit on the simulator
            job = simulator_backend.run(transpiled_circuit, shots=shots)
            result = job.result()

            # Get the counts of quantum state results
            counts = result.get_counts(transpiled_circuit)

            return counts

        # Create circuit
        qc = QuantumCircuit(*self.circ.qregs)
        qc.append(self.circ, self.circ.qregs)
        # If evidence is empty
        if len(evidence) == 0:
            # Measure
            qc.measure_all()
            # Run circuit
            samples = run_circuit(qc, shots)
            return samples
        else:
            # Get grover operator if evidence not empty
            groverOp = self.getGroverOp(evidence)
            # Amplitude amplification
            e = {(self.label2qubit[e_key], e_val) for e_key, e_val in evidence.items()}
            E = {}
            k=-1
            # If the measurement of the evidence qubits matches the evidence stop
            while (e != E) or (k > 10):
                # Increment power
                k += 1
                # Create circuit
                qc = QuantumCircuit(*self.circ.qregs)
                qc.append(self.circ, self.circ.qregs)
                # Apply grover operator 2^k times
                qcGrover = QuantumCircuit(*self.circ.qregs)
                qcGrover.append(groverOp, self.circ.qregs)
                qcGrover = qcGrover.power(2**k)
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
                e_samples = run_circuit(qc_measure, shots=1024)
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

        print(k)

        # Create a classical register with the size of the evidence
        measurement_qcr = ClassicalRegister(self.circ.num_qubits-len(evidence))
        qc.add_register(measurement_qcr)
        # Map the query qubits to the classical bits and measure them
        query_qubits = [(label, self.label2qidx[label], qubit) for label, qubit in self.label2qubit.items() if label not in evidence]
        query_qubits_sorted = sorted(query_qubits, key=lambda x: x[1])
        # Measure query variables and return their count
        qc.measure([q[2] for q in query_qubits_sorted], measurement_qcr)
        # Run circuit
        counts = run_circuit(qc, shots=100000)
        print("Counts")
        print(counts)
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
                query=query.replace('q', char, 1)
            self.samples[query] = val
        print('samples')
        print(self.samples)
        return self.samples


    def inference(self, query, evidence: dict=None, shots: int=None):
        """
        - query: The query variables. If Q is a real subset of X\E the rest will be filled
        - evidence: Provide evidence if rejection sampling should be executed. If you want to indicate no evidence
        insert an empty list. If you want to indicate no new evidence keep this variable empty.
        """
        if evidence is not None:
            self.rejectionSampling(evidence, shots)
        else:
            if not self.samples:
                raise ValueError("Provide evidence for rejection sampling or indicate no evidence with empty list")

        q_count = 0
        tValidS = 0
        for sample_key, sample_val in self.samples.items():
            add = True
            for q_key, q_val in query.items():
                if int(sample_key[self.label2qidx[q_key]]) != q_val:
                    add = False
                    break
            if add:
                q_count += sample_val
            tValidS += sample_val
        return q_count/tValidS


    def visualize(self):
        """Visualizes valid samples"""
        return plot_histogram(self.samples)




