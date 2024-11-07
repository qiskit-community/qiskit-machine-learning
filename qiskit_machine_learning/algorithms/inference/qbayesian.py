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
"""Quantum Bayesian Inference"""

from __future__ import annotations

import copy
from typing import Tuple, Dict, Set, List

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit import Qubit
from qiskit.circuit.library import GroverOperator
from qiskit.primitives import BaseSampler, Sampler, BaseSamplerV2
from qiskit.transpiler.passmanager import BasePassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2


class QBayesian:
    r"""
    Implements a quantum Bayesian inference (QBI) algorithm that has been developed in [1]. The
    Bayesian network must be based on binary random variables (0/1) and represented by a quantum
    circuit. The quantum circuit can be passed in various forms as long as it represents the joint
    probability distribution of the network.

    For Bayesian networks with random variables that have more than two states, see for example [2].

    Note that ``QBayesian`` defines an order for the qubits in the circuit. The last qubit in the
    circuit will correspond to the most significant bit in the joint probability distribution. For
    example, if the random variables A, B, and C are entered into the circuit in this order with
    (A=1, B=0 and C=0), the probability is represented by the probability amplitude of quantum
    state 001.

    **Example**

    .. code-block:: python

        qc = QuantumCircuit(...)

        qb = QBayesian(qc)
        result = qb.inference(query={...}, evidence={...})
        print("Probability of query given evidence: ", result)

    **References**
        [1]: Low, Guang Hao, Theodore J. Yoder, and Isaac L. Chuang. "Quantum inference on Bayesian
        networks", Physical Review A 89.6 (2014): 062315.

        [2]: Borujeni, Sima E., et al. "Quantum circuit representation of Bayesian networks."
        Expert Systems with Applications 176 (2021): 114768.
    """

    # Discrete quantum Bayesian network
    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        limit: int = 10,
        threshold: float = 0.9,
        sampler: BaseSampler | BaseSamplerV2 | None = None,
        pass_manager: BasePassManager | None = None,
    ):
        """
        Args:
            circuit: The quantum circuit that represents the Bayesian network. Each random variable
                should be assigned to exactly one register of one qubit. The last qubit in the
                circuit corresponds to the most significant bit in the binary string, which
                represents the measured quantum state.
            limit: The maximum number of times the Grover operator is integrated (2^limit).
            threshold (float): The threshold to accept the evidence. For example, if set to 0.9,
                this means that each evidence qubit must be equal to the value of the evidence
                variable at least 90% of the measurements.
            sampler: The sampler primitive used to compute the Bayesian inference.
                If ``None`` is given, a default instance of the reference sampler defined
                by :class:`~qiskit.primitives.Sampler` will be used.
        Raises:
            ValueError: If any register in the circuit is not mapped to exactly one qubit.
        """
        # Test valid input
        for qrg in circuit.qregs:
            if qrg.size > 1:
                raise ValueError("Every register needs to be mapped to exactly one unique qubit.")

        # Initialize parameter
        self._circ = circuit
        self._limit = limit
        self._threshold = threshold
        if sampler is None:
            sampler = Sampler()
        self._sampler = sampler

        if pass_manager is None:
            _backend = GenericBackendV2(num_qubits=max(circuit.num_qubits, 2))
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=_backend)
        self._pass_manager = pass_manager

        # Label of register mapped to its qubit
        self._label2qubit = {qrg.name: qrg[0] for qrg in self._circ.qregs}
        # Label of register mapped to its qubit index bottom up in significance
        self._label2qidx = {
            qrg.name: self._circ.num_qubits - idx - 1 for idx, qrg in enumerate(self._circ.qregs)
        }
        # Distribution of samples from rejection sampling
        self._samples: Dict[str, float] = {}
        # True if rejection sampling converged after limit
        self._converged = bool()

    def _get_grover_op(self, evidence: Dict[str, int]) -> GroverOperator:
        """
        Constructs a Grover operator based on the provided evidence. The evidence is used to
        determine the "good states" that the Grover operator will amplify.

        Args:
            evidence: A dictionary representing the evidence with keys as variable labels
                and values as states.
        Returns:
            GroverOperator: The constructed Grover operator.
        """
        # Evidence to reversed qubit index sorted by index
        num_qubits = self._circ.num_qubits
        e2idx = sorted(
            [(self._label2qidx[e_key], e_val) for e_key, e_val in evidence.items()],
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
        return GroverOperator(oracle, state_preparation=self._circ)

    def _run_circuit(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Run the quantum circuit with the sampler."""
        counts = {}

        if isinstance(self._sampler, BaseSampler):
            # Sample from circuit
            job = self._sampler.run(circuit)
            result = job.result()

            # Get the counts of quantum state results
            counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities()

        elif isinstance(self._sampler, BaseSamplerV2):

            # Sample from circuit
            circuit_isa = self._pass_manager.run(circuit)
            job = self._sampler.run([circuit_isa])
            result = job.result()

            bit_array = list(result[0].data.values())[0]
            bitstring_counts = bit_array.get_counts()

            # Normalize the counts to probabilities
            total_shots = result[0].metadata["shots"]
            counts = {k: v / total_shots for k, v in bitstring_counts.items()}

            # Convert to quasi-probabilities
            # counts = QuasiDistribution(probabilities)
            # counts = {k: v for k, v in counts.items()}

        return counts

    def __power_grover(
        self, grover_op: GroverOperator, evidence: Dict[str, int], k: int
    ) -> Tuple[QuantumCircuit, Set[Tuple[Qubit, int]]]:
        """
        Applies the Grover operator to the quantum circuit 2^k times, measures the evidence qubits,
        and returns a tuple containing the updated quantum circuit and a set of the measured
        evidence qubits.

        Args:
            grover_op: The Grover operator to be applied.
            evidence: A dictionary representing the evidence.
            k: The power to which the Grover operator is raised.
        Returns:
            tuple: A tuple containing the updated quantum circuit and a set of the measured evidence
                qubits.
        """
        # Create circuit
        qc = QuantumCircuit(*self._circ.qregs)
        qc.append(self._circ, self._circ.qregs)
        # Apply Grover operator 2^k times
        qc_grover = QuantumCircuit(*self._circ.qregs)
        qc_grover.append(grover_op, self._circ.qregs)
        qc_grover = qc_grover.power(2**k)
        qc.append(qc_grover, self._circ.qregs)
        # Add quantum circuit for measuring
        qc_measure = QuantumCircuit(*self._circ.qregs)
        qc_measure.append(qc, self._circ.qregs)
        # Create a classical register with the size of the evidence
        measurement_ecr = ClassicalRegister(len(evidence))
        qc_measure.add_register(measurement_ecr)
        # Map the evidence qubits to the classical bits and measure them
        evidence_qubits = [self._label2qubit[e_key] for e_key in evidence]
        qc_measure.measure(evidence_qubits, measurement_ecr)
        # Run the circuit with the Grover operator and measurements
        e_samples = self._run_circuit(qc_measure)
        e_count = {self._label2qubit[e]: 0.0 for e in evidence}
        for e_sample_key, e_sample_val in e_samples.items():
            # Go through reverse binary that matches order of qubits
            for i, char in enumerate(e_sample_key[::-1]):
                if int(char) == 1:
                    e_count[evidence_qubits[i]] += e_sample_val
        # Assign to every evidence qubit if it is measured with high probability (th) 1 o/w 0
        e_meas = {
            (e_count_key, int(e_count_val >= self._threshold))
            for e_count_key, e_count_val in e_count.items()
        }
        return qc, e_meas

    def _format_samples(self, samples: Dict[str, float], evidence: List[str]) -> Dict[str, float]:
        """Transforms samples keys back to their variables names."""
        f_samples: Dict[str, float] = {}
        for smpl_key, smpl_val in samples.items():
            q_str, e_str = "", ""
            for var_name, var_idx in sorted(self._label2qidx.items(), key=lambda x: -x[1]):
                if var_name in evidence:
                    e_str += f"{var_name}={smpl_key[var_idx]},"
                else:
                    q_str += f"{var_name}={smpl_key[var_idx]},"
            if evidence:
                f_samples[f"P({q_str[:-1]}|{e_str[:-1]})"] = smpl_val
            else:
                f_samples[f"P({q_str[:-1]})"] = smpl_val
        return f_samples

    def rejection_sampling(
        self, evidence: Dict[str, int], format_res: bool = False
    ) -> Dict[str, float]:
        """
        Performs quantum rejection sampling given the evidence.

        Args:
            evidence: The keys of the dictionary are the evidence variables that are linked to the
                corresponding quantum register with their names and values (0/1). If evidence is
                empty, it measures all qubits. If evidence is given, it uses the Grover operator for
                amplitude amplification and repeats until the evidence matches or limit is reached.
            format_res: If true, maps the output back to variable names. For example, the output
                {'100': 0.23} with evidence A=0, B=0 will be mapped to {'P(C=1|A=0,B=0)': 0.23}.
        Returns:
            A dictionary with the probability distribution of the samples given the evidence, where
            the keys are the sequential values of the variables. Note that the last variable value
            appears as the first character for the key. If format_res is true, the output will be
            mapped back to the variable names, for example {'P(C=1|A=0,B=0)': 0.23}.
        """
        # If evidence is empty
        if len(evidence) == 0:
            # Create circuit
            qc = QuantumCircuit(*self._circ.qregs)
            qc.append(self._circ, self._circ.qregs)
            # Measure
            qc.measure_all()
            # Run circuit
            self._samples = self._run_circuit(qc)
        else:
            # Get Grover operator if evidence not empty
            grover_op = self._get_grover_op(evidence)
            # Amplitude amplification
            true_e = {(self._label2qubit[e_key], e_val) for e_key, e_val in evidence.items()}
            meas_e: Set[Tuple[str, int]] = set()
            best_qc, best_inter = QuantumCircuit(), -1
            self._converged = False
            k = -1
            # If the measurement of the evidence qubits matches the evidence stop
            while (true_e != meas_e) and (k < self._limit):
                # Increment power
                k += 1
                # Create circuit with 2^k times Grover operator
                qc, meas_e = self.__power_grover(grover_op=grover_op, evidence=evidence, k=k)
                # Test number of
                if len(true_e.intersection(meas_e)) > best_inter:
                    best_qc = qc
            if true_e == meas_e:
                self._converged = True
            # Create a classical register with the size of the evidence
            best_qc_meas = QuantumCircuit(*self._circ.qregs)
            best_qc_meas.append(best_qc, self._circ.qregs)
            measurement_qcr = ClassicalRegister(self._circ.num_qubits - len(evidence))
            best_qc_meas.add_register(measurement_qcr)
            # Map the query qubits to the classical bits and measure them
            query_qubits = [
                (label, self._label2qidx[label], qubit)
                for label, qubit in self._label2qubit.items()
                if label not in evidence
            ]
            query_qubits_sorted = sorted(query_qubits, key=lambda x: x[1], reverse=True)
            # Measure query variables and return their count
            best_qc_meas.measure([q[2] for q in query_qubits_sorted], measurement_qcr)
            # Run circuit
            counts = self._run_circuit(best_qc_meas)
            # Build default string with evidence
            query_string = ""
            var_idx_sorted = [
                label for label, _ in sorted(self._label2qidx.items(), key=lambda x: x[1])
            ]
            for var in var_idx_sorted:
                if var in evidence:
                    query_string += str(evidence[var])
                else:
                    query_string += "q"
            # Retrieve valid samples
            self._samples = {}
            # Replace placeholder q with query variables from samples
            for key, val in counts.items():
                query = query_string
                for char in key:
                    query = query.replace("q", char, 1)
                self._samples[query] = val
        if not format_res:
            return copy.deepcopy(self._samples)
        else:
            return self._format_samples(self._samples, list(evidence.keys()))

    def inference(
        self,
        query: Dict[str, int],
        evidence: Dict[str, int] = None,
    ) -> float:
        """
        Performs quantum inference for the query variables given the evidence. It uses quantum
        rejection sampling if evidence is given and calculates the probability of the query.

        Args:
            query: The keys of the dictionary are the query variables that are linked to the
                corresponding quantum registers with their names and values (0/1). If the query
                variables are a real subset of all variables without the evidence, the query will be
                marginalized.
            evidence: The evidence variables. If evidence is a dictionary, the rejection sampling is
                executed with the keys representing the variables linked to the corresponding
                quantum register by their names and values (0/1). If evidence is ``None``, the
                default, then samples from the previous rejection sampling are used.
        Returns:
            The probability of the query given the evidence.
        Raises:
            ValueError: If evidence is required for rejection sampling and ``None`` is given.
        """
        if evidence is not None:
            self.rejection_sampling(evidence)
        elif not self._samples:
            raise ValueError("Provide evidence or indicate no evidence with an empty dictionary")
        # Get sorted indices of query qubits
        query_indices_rev = [(self._label2qidx[q_key], q_val) for q_key, q_val in query.items()]
        # Get probability of query
        res = 0.0
        for sample_key, sample_val in self._samples.items():
            add = True
            for q_idx, q_val in query_indices_rev:
                if int(sample_key[q_idx]) != q_val:
                    add = False
                    break
            if add:
                res += sample_val
        return res

    @property
    def converged(self) -> bool:
        """Returns ``True`` if a solution for the evidence with the given threshold was found
        without reaching the maximum number of times the Grover operator was applied (2^limit)."""
        return self._converged

    @property
    def samples(self) -> Dict[str, float]:
        """Returns the samples generated from the rejection sampling."""
        return self._samples

    @property
    def limit(self) -> int:
        """Returns the maximum number of times the Grover operator can be applied (2^limit)."""
        return self._limit

    @limit.setter
    def limit(self, limit: int):
        """Set the maximum number of times the Grover operator can be applied (2^limit)."""
        self._limit = limit

    @property
    def sampler(self) -> BaseSampler | BaseSamplerV2:
        """Returns the sampler primitive used to compute the samples."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler | BaseSamplerV2):
        """Set the sampler primitive used to compute the samples."""
        self._sampler = sampler

    @property
    def threshold(self) -> float:
        """Returns the threshold to accept the evidence."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        """Set the threshold to accept the evidence."""
        self._threshold = threshold
