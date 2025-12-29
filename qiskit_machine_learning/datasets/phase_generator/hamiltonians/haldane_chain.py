"""
Haldane Chain Model Implementation

This module implements the Haldane chain model with three-site interactions and
transverse fields.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.hamiltonian_base import HamiltonianModel

class HaldaneChain(HamiltonianModel):
    """Haldane Chain model.

    This model implements a Haldane chain with three-site interactions and
    transverse fields.

    Attributes:
        num_qubits (int): Number of qubits in the system.
        J (float): Strength of the three-site (ZXZ) interaction.
        h1 (float): Strength of the single-site X field.
        h2 (float): Strength of the two-site XX interaction.
    """

    def __init__(self, num_qubits, J, h1, h2):
        """Initialize the Haldane Chain model.

        Args:
            num_qubits (int): Number of qubits in the system.
            J (float): Strength of the three-site (ZXZ) interaction.
            h1 (float): Strength of the single-site X field.
            h2 (float): Strength of the two-site XX interaction.
        """
        super().__init__(num_qubits)
        self.J = J
        self.h1 = h1
        self.h2 = h2

    def get_hamiltonian(self):
        """Generate the Haldane Chain Hamiltonian.

        Returns:
            SparsePauliOp: Hamiltonian with ZXZ, X, and XX interactions.
        """
        pauli_list = []
        coeffs = []

        # Three-site ZXZ interactions
        for i in range(self.num_qubits - 2):
            label = ['I'] * self.num_qubits
            label[i] = 'Z'
            label[i+1] = 'X'
            label[i+2] = 'Z'
            pauli_list.append(''.join(label))
            coeffs.append(-self.J)  # Negative sign for antiferromagnetic coupling

        # Single-site X field terms
        for i in range(self.num_qubits):
            label = ['I'] * self.num_qubits
            label[i] = 'X'
            pauli_list.append(''.join(label))
            coeffs.append(-self.h1)

        # Two-site XX interaction terms
        for i in range(self.num_qubits - 1):
            label = ['I'] * self.num_qubits
            label[i] = 'X'
            label[i+1] = 'X'
            pauli_list.append(''.join(label))
            coeffs.append(-self.h2)

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    @classmethod
    def sample_parameters(cls, num_qubits, J_range=(-1, 1), h1_range=(-1, 1), h2_range=(-1, 1), num_samples=10):
        """Sample different parameter configurations.

        Args:
            num_qubits (int): Number of qubits in the system.
            J_range (tuple): Range (min, max) for J parameter.
            h1_range (tuple): Range (min, max) for h1 parameter.
            h2_range (tuple): Range (min, max) for h2 parameter.
            num_samples (int): Number of parameter sets to sample.

        Returns:
            list: List of HaldaneChain model instances with sampled parameters.
        """
        samples = []
        for _ in range(num_samples):
            J = np.random.uniform(*J_range)
            h1 = np.random.uniform(*h1_range)
            h2 = np.random.uniform(*h2_range)
            samples.append(cls(num_qubits, J, h1, h2))
        return samples

    def get_phase(self):
        """Determine the phase of the Haldane chain based on parameters.

        Returns:
            str: The phase label.
        """
        if abs(self.h1) > abs(self.J):
            return "SPT Paramagnetic"
        else:
            return "SPT Antiferromagnetic"