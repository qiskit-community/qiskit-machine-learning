"""
ANNNI Model Implementation

This module implements the Axial Next-Nearest-Neighbor Ising (ANNNI) model.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.hamiltonian_base import HamiltonianModel

class ANNNIModel(HamiltonianModel):
    """Axial Next-Nearest-Neighbor Ising (ANNNI) Model.

    This model includes nearest-neighbor and next-nearest-neighbor Ising interactions
    with a transverse field.

    Attributes:
        num_qubits (int): Number of qubits in the system.
        J1 (float): Nearest-neighbor coupling constant.
        J2 (float): Next-nearest-neighbor coupling constant.
        B (float): Transverse field strength.
    """

    def __init__(self, num_qubits, J1, J2, B):
        """Initialize the ANNNI model.

        Args:
            num_qubits (int): Number of qubits in the system.
            J1 (float): Nearest-neighbor coupling constant.
            J2 (float): Next-nearest-neighbor coupling constant.
            B (float): Transverse field strength.
        """
        super().__init__(num_qubits)
        self.J1 = J1
        self.J2 = J2
        self.B = B

    def get_hamiltonian(self):
        """Generate the ANNNI model Hamiltonian.

        Returns:
            SparsePauliOp: Hamiltonian with XX, XX (next-nearest), and Z interactions.
        """
        pauli_list = []
        coeffs = []

        # Nearest-neighbor XX interactions
        for i in range(self.num_qubits - 1):
            label = ['I'] * self.num_qubits
            label[i] = 'X'
            label[i+1] = 'X'
            pauli_list.append(''.join(label))
            coeffs.append(-self.J1)

        # Next-nearest-neighbor XX interactions
        for i in range(self.num_qubits - 2):
            label = ['I'] * self.num_qubits
            label[i] = 'X'
            label[i+2] = 'X'
            pauli_list.append(''.join(label))
            coeffs.append(-self.J2)

        # Transverse field Z terms
        for i in range(self.num_qubits):
            label = ['I'] * self.num_qubits
            label[i] = 'Z'
            pauli_list.append(''.join(label))
            coeffs.append(-self.B)

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    @classmethod
    def sample_parameters(cls, num_qubits, J1_range=(-1, 1), J2_range=(-1, 1), B_range=(-1, 1), num_samples=10):
        """Sample different parameter configurations.

        Args:
            num_qubits (int): Number of qubits in the system.
            J1_range (tuple): Range (min, max) for J1 parameter.
            J2_range (tuple): Range (min, max) for J2 parameter.
            B_range (tuple): Range (min, max) for B parameter.
            num_samples (int): Number of parameter sets to sample.

        Returns:
            list: List of ANNNIModel instances with sampled parameters.
        """
        samples = []
        for _ in range(num_samples):
            J1 = np.random.uniform(*J1_range)
            J2 = np.random.uniform(*J2_range)
            B = np.random.uniform(*B_range)
            samples.append(cls(num_qubits, J1, J2, B))
        return samples

    def get_phase(self):
        """Determine the phase of the ANNNI model based on parameters.

        Returns:
            str: The phase label.
        """
        if abs(self.J1) > abs(self.B):
            return "ferromagnetic"
        elif abs(self.J1) < abs(self.B):
            return "paramagnetic"
        elif abs(self.J2) > abs(self.J1):
            return "antiphase"
        else:
            return "floating"