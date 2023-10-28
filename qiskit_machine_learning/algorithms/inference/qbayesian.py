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


from qiskit import Aer, QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
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


    def getSe(self, ctrls):
        """
        Creates Se for Grover

        ctrls: control qubits represent the evidence var
        """
        # Create circuit with registers from given quantum circuit
        opSe = QuantumCircuit(*self.circ.qregs)
        # Q=X\E
        query_var = {self.label2qubit[reg.name] for reg in self.circ.qregs} - set(ctrls)
        # Generate Se
        for q in query_var:
            # multi control z gate
            opSe.h(q)
            opSe.mcx(ctrls, q)
            opSe.h(q)
            # x gate
            opSe.x(q)
            # multi control z gate
            opSe.h(q)
            opSe.mcx(ctrls, q)
            opSe.h(q)
            # x gate
            opSe.x(q)
        return opSe

    def rejectionSampling(self, evidence, shots: int=None):
        def run_circuit(circuit, shots=100_000):
            """
            Run the provided quantum circuit on the Aer simulator backend.

            Parameters:
            - circuit: The quantum circuit to be executed.
            - shots (default=10,000): The number of times the circuit is executed.

            Returns:
            - counts: A dictionary with the counts of each quantum state result.
            """
            print(shots)
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
        # Amplitude amplification circuit if evidence not empty
        e_reg = [self.label2qubit[qrg.name] for qrg in self.circ.qregs if qrg.name in evidence]
        if len(e_reg) != 0:
            # Get Se
            opSe = self.getSe(e_reg)
            # Grover
            opG = GroverOperator(opSe, self.circ)
            qc.append(opG, self.circ.qregs)
        # Measure
        qc.measure_all()
        # Run circuit
        if shots is None:
            counts = run_circuit(qc)
        else:
            print("here")
            counts = run_circuit(qc, shots)
        # Retrieve valid samples
        self.samples = {}
        re=0
        # Assume key is bin and e_key is the qubits number
        for key, val in counts.items():
            accept = True
            for e_key, e_val in evidence.items():
                if int(key[self.label2qidx[e_key]]) != e_val:
                    accept = False
                    break
            if accept:
                self.samples[key] = val
            else:
                re+=val
        print(counts)
        print(re)
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




