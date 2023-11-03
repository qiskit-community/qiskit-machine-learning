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
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.algorithms import AmplificationProblem
from qiskit_algorithms import Grover
from qiskit.primitives import Sampler

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


    def getAmplifyPrb(self, evidence):
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
        return AmplificationProblem(oracle, state_preparation=self.circ, is_good_state=good_states)


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

        # If evidence is empty
        if len(evidence) == 0:
            # Create circuit
            qc = QuantumCircuit(*self.circ.qregs)
            qc.append(self.circ, self.circ.qregs)
            # Measure
            qc.measure_all()
            # Run circuit
            samples = run_circuit(qc, shots)
            return samples

        # Amplitude amplification circuit if evidence not empty
        ampPrb = self.getAmplifyPrb(evidence)
        # Grover with default number of iterations given by good states from amplitude amplification problem
        grover = Grover(sampler=Sampler())
        # Run circuit
        a=grover.amplify(ampPrb)
        print(a.circuit_results)
        print(a.circuit_results[0])
        #TODO why multiple results here
        #print(grover.amplify(ampPrb).iterations)
        #print(grover.amplify(ampPrb).max_probability)
        counts = grover.amplify(ampPrb).circuit_results[-1]
        print("Result is ")
        print(counts)
        # Retrieve valid samples
        self.samples = {}
        for key, val in counts.items():
            accept = True
            for e_key, e_val in evidence.items():
                # Check if evidence fits with sample
                if int(key[self.circ.num_qubits-self.label2qidx[e_key]-1]) != e_val:
                    accept = False
                    break
            if accept:
                self.samples[key] = val
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




