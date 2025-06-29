"""
Cluster Model Implementation

This module implements the Cluster model with Z, XX, and XZX interactions.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.hamiltonian_base import HamiltonianModel

class ClusterModel(HamiltonianModel):
    """Cluster Model with Z, XX, and XZX interactions.

    This model implements a cluster state Hamiltonian with various interaction terms.

    Attributes:
        num_qubits (int): Number of qubits in the system.
        J1 (float): Strength of the XX interaction.
        J2 (float): Strength of the XZX interaction.
    """

    def __init__(self, num_qubits, J1, J2):
        """Initialize the Cluster model.

        Args:
            num_qubits (int): Number of qubits in the system.
            J1 (float): Strength of the XX interaction.
            J2 (float): Strength of the XZX interaction.
        """
        super().__init__(num_qubits)
        self.J1 = J1
        self.J2 = J2

    def get_hamiltonian(self):
        """Generate the Cluster model Hamiltonian.

        Returns:
            SparsePauliOp: Hamiltonian with Z, XX, and XZX interactions.
        """
        pauli_list = []
        coeffs = []

        # Z terms (on all sites)
        for i in range(self.num_qubits):
            label = ['I'] * self.num_qubits
            label[i] = 'Z'
            pauli_list.append(''.join(label))
            coeffs.append(1.0)  # Standard coefficient of 1 for Z terms

            # XX terms (for adjacent sites)
            if i < self.num_qubits - 1:
                label = ['I'] * self.num_qubits
                label[i] = 'X'
                label[i+1] = 'X'
                pauli_list.append(''.join(label))
                coeffs.append(-self.J1)

            # XZX terms (for triplets of sites)
            if 0 < i < self.num_qubits - 1:
                label = ['I'] * self.num_qubits
                label[i-1] = 'X'
                label[i] = 'Z'
                label[i+1] = 'X'
                pauli_list.append(''.join(label))
                coeffs.append(-self.J2)

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    @classmethod
    def sample_parameters(cls, num_qubits, J1_range=(-1, 1), J2_range=(-1, 1), num_samples=10):
        """Sample different parameter configurations.

        Args:
            num_qubits (int): Number of qubits in the system.
            J1_range (tuple): Range (min, max) for J1 parameter.
            J2_range (tuple): Range (min, max) for J2 parameter.
            num_samples (int): Number of parameter sets to sample.

        Returns:
            list: List of ClusterModel instances with sampled parameters.
        """
        samples = []
        for _ in range(num_samples):
            J1 = np.random.uniform(*J1_range)
            J2 = np.random.uniform(*J2_range)
            samples.append(cls(num_qubits, J1, J2))
        return samples

    def get_phase(self):
        """Determine the phase of the Cluster model based on parameters.

        Returns:
            str: The phase label.
        """
        if abs(self.J1) > abs(self.J2):
            return "Haldane phase"
        elif self.J1 > 0 and self.J2 > 0:
            return "ferromagnetic"
        elif self.J1 < 0 and self.J2 < 0:
            return "anti-ferromagnetic"
        else:
            return "trivial"