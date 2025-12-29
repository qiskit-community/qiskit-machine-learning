"""
Heisenberg XXX Model Implementation

This module implements the Heisenberg XXX model with bond-alternating couplings.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.hamiltonian_base import HamiltonianModel

class HeisenbergXXX(HamiltonianModel):
    """Heisenberg XXX model with bond-alternating couplings.

    This model implements the Heisenberg XXX interaction between neighboring spins
    with coupling strengths that can vary between bonds.

    Attributes:
        num_qubits (int): Number of qubits in the system.
        J_list (list): List of coupling constants for each bond.
    """

    def __init__(self, num_qubits, J_list):
        """Initialize the Heisenberg XXX model.

        Args:
            num_qubits (int): Number of qubits in the system.
            J_list (list): List of coupling constants for each bond.
        """
        super().__init__(num_qubits)
        # Ensure J_list has correct length (num_qubits - 1)
        if len(J_list) != num_qubits - 1:
            raise ValueError(f"J_list must have length {num_qubits-1}, got {len(J_list)}")
        self.J_list = J_list

    def get_hamiltonian(self):
        """Generate the Heisenberg XXX Hamiltonian.

        Returns:
            SparsePauliOp: Hamiltonian with XX, YY, ZZ interactions.
        """
        pauli_list = []
        coeffs = []

        for i in range(self.num_qubits - 1):
            # Add XX, YY, ZZ terms for each bond
            for pauli in ['XX', 'YY', 'ZZ']:
                label = ['I'] * self.num_qubits
                label[i] = pauli[0]
                label[i+1] = pauli[1]
                pauli_list.append(''.join(label))
                coeffs.append(self.J_list[i])

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    @classmethod
    def sample_parameters(cls, num_qubits, J_range=(-1, 1), num_samples=10):
        """Sample different J coupling configurations.

        Args:
            num_qubits (int): Number of qubits in the system.
            J_range (tuple): Range (min, max) for coupling constants.
            num_samples (int): Number of parameter sets to sample.

        Returns:
            list: List of HeisenbergXXX model instances with sampled parameters.
        """
        samples = []
        for _ in range(num_samples):
            # Generate random coupling constants for each bond
            J_list = np.random.uniform(J_range[0], J_range[1], num_qubits-1)
            samples.append(cls(num_qubits, J_list))
        return samples

    def get_phase(self):
        """Determine the phase of the Heisenberg model based on parameters.

        Returns:
            str: The phase label ("topological" or "trivial").
        """
        avg_J = np.mean(self.J_list)
        return "topological" if avg_J > 0 else "trivial"