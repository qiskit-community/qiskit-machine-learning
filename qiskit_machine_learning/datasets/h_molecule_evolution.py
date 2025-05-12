# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
H Molecule Evolution
"""

from __future__ import annotations

import warnings
import os

import numpy as np
import pickle as pkl

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.providers import QiskitBackendNotFoundError

from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from scipy.linalg import expm


from ..utils import algorithm_globals


# pylint: disable=too-many-positional-arguments
def h_molecule_evolution_data(
    delta_t: float,
    train_end: int,
    test_start: int,
    test_end: int,
    molecule: str = "H2",
    noise_mode: str = "ibm_brisbane",
    formatting: str = "ndarray"
) -> (
    tuple[Statevector, np.ndarray, list[Statevector], np.ndarray, list[Statevector]]
):
    r""" """

    occupancy = {"H2": 2, "H3": 2, "H6": 6}
    num_occupancy = occupancy[molecule]

    # Noise Models for Training Data
    simulator = _noise_simulator(noise_mode)

    # Import Hamiltonian and Unitary Evolution Circuit
    qc, t, hamiltonian = _evolution_circuit(molecule)
    qc_evo = qc.assign_parameters({t: delta_t})

    # Get Hartree Fock State
    psi_hf = _initial_state(hamiltonian, num_occupancy)

    # Time stamps for Train & Test
    idx_train, idx_test = np.arange(0, train_end + 1), np.arange(test_start, test_end + 1)
    x_train, x_test = delta_t * idx_train, delta_t * idx_test

    # Noisy Shortterm Evolutions
    y_train = _simulate_shortterm(psi_hf, qc_evo, simulator, train_end)

    # Ideal Longterm Evolutions
    y_test = _ideal_longterm(psi_hf, hamiltonian, x_test)

    if formatting == "ndarray":
        y_train = _to_np(y_train)
        y_test = _to_np(y_test)

    return (psi_hf, x_train, y_train, x_test, y_test)


def _evolution_circuit(molecule):
    """Get the parametrized circuit for evolution after Trotterization.
    Returns:
    - QuantumCircuit (for training set)
    - Parameter Object "t" (for training set)
    - Original Hamiltonian (for testing set)"""

    spo = _hamiltonian_import(molecule)

    t = Parameter("t")
    trotterizer = SuzukiTrotter(order=2, reps=1)
    u_evolution = PauliEvolutionGate(spo, time=t, synthesis=trotterizer)

    n_qubits = spo.num_qubits
    qc = QuantumCircuit(n_qubits)
    qc.append(u_evolution, range(n_qubits))

    qc_flat = qc.decompose()

    return qc_flat, t, spo


def _hamiltonian_import(molecule):
    """Import Hamiltonian from Hamiltonians folder"""

    dir_path = os.path.dirname(__file__)
    filename = os.path.join(dir_path, f"hamiltonians\\{molecule}.bin")

    with open(filename, "rb") as f:
        spo = pkl.load(f)

    return spo


def _initial_state(hamiltonian, num_occupancy):
    """Sets a realistic initial state

    JW map automatically keeps orbitals in ascending order of energy"""

    n_qubits = hamiltonian.num_qubits

    bitstring = ["1"] * num_occupancy + ["0"] * (n_qubits - num_occupancy)

    occupation_label = "".join(bitstring)

    return Statevector.from_label(occupation_label)


def _noise_simulator(noise_mode):
    """Returns a Noisy/Noiseless AerSimulator object"""

    if noise_mode == "noiseless":
        noise_model = None

    elif noise_mode == "reduced":
        single_qubit_error = depolarizing_error(0.001, 1)
        two_qubit_error = depolarizing_error(0.01, 2)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ["u1", "u2", "u3"])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])

    # If the given Model is an IBM location
    else:

        service = QiskitRuntimeService()

        try:
            backend = service.backend(noise_mode)
        except QiskitBackendNotFoundError:
            backends = service.backends(min_num_qubits=4, operational=True, simulator=False)
            raise QiskitBackendNotFoundError(
                f"""The specified backend '{noise_mode}' was not found / was busy. Please select one from {backends}"""
            )

        noise_model = NoiseModel.from_backend(backend)

    simulator = AerSimulator(noise_model=noise_model)
    return simulator

def _simulate_shortterm(psi_hf, qc_evo, simulator, train_end):
    """Simulates short-term dynamics using a noisy simulator."""
    
    y_train = []
    psi = psi_hf.copy()
    
    for _ in range(train_end):
        
        # Create a new quantum circuit for each step
        qc = QuantumCircuit(psi.num_qubits)
        
        # psi persists after each step
        qc.initialize(psi.data, qc.qubits)
        qc.append(qc_evo, qc.qubits)

        qc.save_statevector()
        qc_resolved = transpile(qc, simulator)
        
        # Execute the circuit on the noisy simulator
        job = simulator.run(qc_resolved)
        result = job.result()
        
        # Update the statevector with the result
        psi = Statevector(result.get_statevector(qc))
        y_train.append(psi.copy())
    
    return y_train

def _ideal_longterm(psi_hf, H, t):
    """
    Return the list of statevectors  exp(-i H t_k) @ psi_hf
    for every t_k in `times`, using an exact matrix exponential.
    """
    h_dense = H.to_matrix()
    y_test = []

    for t_k in t:
        u_t = expm(-1j * h_dense * t_k) 
        psi_t = Statevector(u_t @ psi_hf.data)
        y_test.append(psi_t)

    return y_test

def _to_np(states):
    """Convert list[Statevector] to ndarray"""
    dim = len(states[0])
    return np.stack([sv.data for sv in states], axis=0).reshape(len(states), dim, 1)


print(h_molecule_evolution_data(0.1, 3, 6, 8))
