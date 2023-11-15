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
"""Quantum Bayesian Inference"""

from typing import Tuple, Dict, Set
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator


class QBayesian:
    r"""
    Implements Quantum Bayesian Inference algorithm. The algorithm has been developed in [1].

    **References**
        [1]: Low, Guang Hao, Theodore J. Yoder, and Isaac L. Chuang.
        "Quantum inference on Bayesian networks", Physical Review A 89.6 (2014): 062315.

    Usage:
    ------
    To use the `QBayesian` class, instantiate it with a quantum circuit that represents the
    Bayesian network. You can then use the `inference` method to estimate probabilities given
    evidence, optionally using rejection sampling and Grover's algorithm for amplification.

    Example:
    --------

    # Define a quantum circuit
    qc = QuantumCircuit(...)

    # Initialize the framework
    qb = QBayesian(qc)

    # Perform inference
    result = qb.inference(query={...}, evidence={...})

    print("Probability of query given evidence:", result)
    """

    # Discrete quantum Bayesian network
    def __init__(self, circuit: QuantumCircuit):
        """
        Run the provided quantum circuit on the Aer simulator backend. For other simulator
        overwrite the method run_circuit().

        Args:
            circuit (QuantumCircuit): The quantum circuit representing the Bayesian network.
            Each random variable should be assigned to exactly one register of one qubit. The
            first qubit in the circuit will be the one of highest order.

        Raises:
            ValueError: If any register in the circuit is not mapped to exactly one qubit.

        """
        # Test valid input
        for qrg in circuit.qregs:
            if qrg.size > 1:
                raise ValueError("Every register needs to be mapped to exactly one unique qubit")
        # Initialize parameter
        self.circ = circuit
        # Label of register mapped to its qubit
        self.label2qubit = {qrg.name: qrg[0] for qrg in self.circ.qregs}
        # Label of register mapped to its qubit index bottom up in significance
        self.label2qidx = {
            qrg.name: self.circ.num_qubits - idx - 1 for idx, qrg in enumerate(self.circ.qregs)
        }
        # Samples from rejection sampling
        self.samples: Dict[str, int] = {}
        # True if rejection sampling converged after limit
        self.converged = bool()

    def get_grover_op(self, evidence: dict) -> GroverOperator:
        """
        Constructs a Grover operator based on the provided evidence. The evidence is used to
        determine the "good states" that the Grover operator will amplify.
        Args:
            evidence (dict): A dictionary representing the evidence with keys as variable labels
            and values as states.
        Returns:
            GroverOperator: The constructed Grover operator.
        """
        # Evidence to reversed qubit index sorted by index
        num_qubits = self.circ.num_qubits
        e2idx = sorted(
            [(self.label2qidx[e_key], e_val) for e_key, e_val in evidence.items()],
            key=lambda x: x[0],
        )
        # Binary format of good states
        num_evd = len(e2idx)
        bin_str = [
            format(i, f"0{(num_qubits - num_evd)}b") for i in range(2 ** (num_qubits - num_evd))
        ]
        # Get good states
        good_states = []
        for b in bin_str:
            for e_idx, e_val in e2idx:
                b = b[:e_idx] + str(e_val) + b[e_idx:]
            good_states.append(b)
        # Get statevector by transform good states w.r.t its index to 1 and o/w to 0
        oracle = Statevector(
            [int(format(i, f"0{num_qubits}b") in good_states) for i in range(2**num_qubits)]
        )
        return GroverOperator(oracle, state_preparation=self.circ)

    def run_circuit(self, circuit: QuantumCircuit, shots=100_000) -> dict:
        """Run the quantum circuit for the number of shots on the Aer simulator backend."""
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

    def power_grover(
        self, grover_op: GroverOperator, evidence: dict, k: int, threshold: float
    ) -> Tuple[GroverOperator, set]:
        """
        Applies the Grover operator to the quantum circuit 2^k times, measures the evidence qubits,
        and returns a tuple containing the updated quantum circuit and a set of the measured
        evidence qubits.
        Args:
            grover_op (GroverOperator): The Grover operator to be applied.
            evidence (dict): A dictionary representing the evidence.
            k (int): The power to which the Grover operator is raised.
            threshold (float): The threshold for accepted evidence
        Returns:
            tuple: A tuple containing the updated quantum circuit and a set of the
            measured evidence qubits.
        """
        # Create circuit
        qc = QuantumCircuit(*self.circ.qregs)
        qc.append(self.circ, self.circ.qregs)
        # Apply Grover operator 2^k times
        qc_grover = QuantumCircuit(*self.circ.qregs)
        qc_grover.append(grover_op, self.circ.qregs)
        qc_grover = qc_grover.power(2**k)
        qc.append(qc_grover, self.circ.qregs)
        # Add quantum circuit for measuring
        qc_measure = QuantumCircuit(*self.circ.qregs)
        qc_measure.append(qc, self.circ.qregs)
        # Create a classical register with the size of the evidence
        measurement_ecr = ClassicalRegister(len(evidence))
        qc_measure.add_register(measurement_ecr)
        # Map the evidence qubits to the classical bits and measure them
        evidence_qubits = [self.label2qubit[e_key] for e_key in evidence]
        qc_measure.measure(evidence_qubits, measurement_ecr)
        # Run the circuit with the Grover operator and measurements
        e_samples = self.run_circuit(qc_measure, shots=1024 * self.circ.num_qubits)
        e_count = {self.label2qubit[e]: 0 for e in evidence}
        for e_sample_key, e_sample_val in e_samples.items():
            # Go through reverse binary that matches order of qubits
            for i, char in enumerate(e_sample_key[::-1]):
                if int(char) == 1:
                    e_count[evidence_qubits[i]] += e_sample_val
                else:
                    e_count[evidence_qubits[i]] += -e_sample_val
        # Assign to every evidence qubit if it is measured with high probability (th) 1 o/w 0
        e_meas = {
            (e_count_key, int(e_count_val >= threshold))
            for e_count_key, e_count_val in e_count.items()
        }
        return qc, e_meas

    def rejection_sampling(
        self, evidence: dict, shots: int = 100000, limit: int = 10, threshold: float = 0.8
    ) -> dict:
        """
        Performs rejection sampling given the evidence. If evidence is empty, it runs the circuit
        and measures all qubits. If evidence is provided, it uses the Grover operator for amplitude
        amplification and iterates until the evidence matches or a limit is reached.
        Args:
            evidence (dict): A dictionary representing the evidence.
            shots (int): The number of times the circuit will be executed.
            limit (int): The maximum number of iterations for the Grover operator.
            threshold (float): The threshold for accepted evidence
        Returns:
            dict: A dictionary containing the samples as a dictionary
        """
        # If evidence is empty
        if len(evidence) == 0:
            # Create circuit
            qc = QuantumCircuit(*self.circ.qregs)
            qc.append(self.circ, self.circ.qregs)
            # Measure
            qc.measure_all()
            # Run circuit
            self.samples = self.run_circuit(qc, shots=shots)
            return self.samples
        # Get Grover operator if evidence not empty
        grover_op = self.get_grover_op(evidence)
        # Amplitude amplification
        true_e = {(self.label2qubit[e_key], e_val) for e_key, e_val in evidence.items()}
        meas_e: Set[Tuple[str, int]] = set()
        best_qc, best_inter = QuantumCircuit(), 0
        self.converged = False
        k = -1
        # If the measurement of the evidence qubits matches the evidence stop
        while (true_e != meas_e) and (k < limit):
            # Increment power
            k += 1
            # Create circuit with 2^k times Grover operator
            qc, meas_e = self.power_grover(
                grover_op=grover_op, evidence=evidence, k=k, threshold=threshold
            )
            # Test number of
            if len(true_e.intersection(meas_e)) > best_inter:
                best_qc = qc
        if true_e == meas_e:
            self.converged = True

        # Create a classical register with the size of the evidence
        best_qc_meas = QuantumCircuit(*self.circ.qregs)
        best_qc_meas.append(best_qc, self.circ.qregs)
        measurement_qcr = ClassicalRegister(self.circ.num_qubits - len(evidence))
        best_qc_meas.add_register(measurement_qcr)
        # Map the query qubits to the classical bits and measure them
        query_qubits = [
            (label, self.label2qidx[label], qubit)
            for label, qubit in self.label2qubit.items()
            if label not in evidence
        ]
        query_qubits_sorted = sorted(query_qubits, key=lambda x: x[1], reverse=True)
        # Measure query variables and return their count
        best_qc_meas.measure([q[2] for q in query_qubits_sorted], measurement_qcr)
        # Run circuit
        counts = self.run_circuit(best_qc_meas, shots=shots)
        # Build default string with evidence
        query_string = ""
        var_idx_sorted = [label for label, _ in sorted(self.label2qidx.items(), key=lambda x: x[1])]
        for var in var_idx_sorted:
            if var in evidence:
                query_string += str(evidence[var])
            else:
                query_string += "q"
        # Retrieve valid samples
        self.samples = {}
        # Replace placeholder q with query variables from samples
        for key, val in counts.items():
            query = query_string
            for char in key:
                query = query.replace("q", char, 1)
            self.samples[query] = val
        return self.samples

    def inference(
        self,
        query: dict,
        evidence: dict = None,
        shots: int = 100000,
        limit: int = 10,
        threshold: float = 0.8,
    ) -> float:
        """
        Performs inference on the query variables given the evidence. It uses rejection sampling if
        evidence is provided and calculates the probability of the query.
        Args:
            query (dict): The query variables with keys as variable labels and values as states.
            If Q is a real subset of X without E, it will be marginalized.
            evidence (dict, optional): The evidence variables. If provided, rejection sampling is
            executed.  If you want to indicate no evidence insert an empty list.
            shots (int): The number of times the circuit will be executed.
            limit (int): The maximum number of 2^k times the Grover operator is integrated
            threshold (float): The threshold for accepted evidence
        Returns:
            float: The probability of the query given the evidence.
        Raises:
            ValueError: If evidence is required for rejection sampling and none is provided.
        """
        if evidence is not None:
            self.rejection_sampling(evidence, shots, limit, threshold)
        else:
            if not self.samples:
                raise ValueError("Provide evidence or indicate no evidence with empty list")
        # Get sorted indices of query qubits
        query_indices_rev = [(self.label2qidx[q_key], q_val) for q_key, q_val in query.items()]
        # Get probability of query
        res = 0
        for sample_key, sample_val in self.samples.items():
            add = True
            for q_idx, q_val in query_indices_rev:
                if int(sample_key[q_idx]) != q_val:
                    add = False
                    break
            if add:
                res += sample_val
        return res
