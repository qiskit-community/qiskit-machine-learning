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
import pickle as pkl

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from scipy.linalg import expm


# pylint: disable=too-many-positional-arguments
def h_molecule_evolution_data(
    delta_t: float,
    train_end: int,
    test_start: int,
    test_end: int,
    molecule: str = "H2",
    noise_mode: str = "reduced",
    formatting: str = "ndarray",
) -> (
    tuple[Statevector, np.ndarray, list[Statevector], np.ndarray, list[Statevector]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    r"""

    Generates a dataset based on the time-evolution of Hydrogen molecules that can be used
    to benchmark Variational Fast Forward (VFF) pipelines such as those discussed in [1].
    The dataset generator gives the user the Hartree-Fock (HF) state of a Hydrogen molecule's
    spin orbital occupancy, a few noisy short-term evolutions of this state over time for
    training the VFF and exact long-term evolutions of this state for comparing with the
    long-term inferences made by the pipeline.


    The Fermionic Hamiltonian for the Hydrogen Molecule is first obtained using Quantum
    Chemistry calculations. This is then mapped to Qubits in Pauli form using the
    Jordan-Wigner Mapping. Thus each qubit state represents the occupancy state of
    one spin orbital each. The HF state :math:`\ket{\psi_{HF}}` is obtained by setting a
    :math:`\ket{1}` state for the lowest energy orbitals and :math:`\ket{0}` for higher.


    For generating the short-term evolutions with realistic noise models that will be
    incurred by a Quantum Computer if it were to simulate short-term evolution terms,
    the unitary operator for the evolution is first transpiled into a circuit with
    :class:`~qiskit.synthesis.SuzukiTrotter` and :class:`~qiskit.circuit.library.PauliEvolutionGate`.
    That is, suppose :math:`U` represents the noisy circuit's effect on a given state to simulate
    the evolution through a time step of :math:`\Delta T` (``delta_t`` given by the user), then


    .. math::
        U \approx e^{- j H \Delta T}


    Where the approximate sign signifies that there is noise added by the noisy simulation and
    also the approximate nature of transpiling with a trotterized hamiltonian. Now, the
    short-term evolution terms are generate until ``train_end`` such evolutions. Suppose we
    denote ``train_end`` as N. Then


    .. math::
        \text{y_train} =
            \left[\ket{\psi_{HF}}, U \ket{\psi_{HF}}, ...U^N \ket{\psi_{HF}}\right]


    Long-term evolution for testing as numerically generated from the exact Hamiltonian without
    the uncertainities introduced by noise and trotterization. Suppose ``test_start`` is denoted
    as P and ``test_end`` as Q. Then


    .. math::
        \text{y_test} =
            \left[e^{-jHP\Delta T} \ket{\psi_{HF}}...e^{-jHQ\Delta T} \ket{\psi_{HF}}\right]


    The choice of noise added in simulation is determined by ``noise_mode``, which can also
    fetch calibration data from IBM runtimes. ``formatting`` parameter can be used to get
    the data as numpy arrays or as list of statevectors as per the usecase.


    **References:**

    [1] Filip M-A, Muñoz Ramo D, Fitzpatrick N. *Variational Phase Estimation
    with Variational Fast Forwarding*. Quantum. 2024 Mar;8:1278.
    `arXiv:2211.16097 <https://arxiv.org/abs/2211.16097>`_

    [2] Cîrstoiu C, Holmes Z, Iosue J, Cincio L, Coles PJ, Sornborger A.
    *Variational fast forwarding for quantum simulation beyond the coherence time*.
    npj Quantum Information. 2020 Sep;6(1):82.
    `arXiv:1910.04292 <https://arxiv.org/abs/1910.04292>`_

    Parameters:
        delta_t : Time step per evolution term (in atomic units). 1 a.u. = 2.42e-17 s
        train_end :  Generate short term evolutions up until :math:`U ^ \text{train_end}`
        test_start : Generate long term evolution terms from :math:`U ^ \text{test_start}`
        test_end : Generate long term evolution terms until :math:`U ^ \text{test_end}`
        molecule : Decides which molecule is being simulation. The options are:

                * ``"H2"``: A linear H2 molecule at 0.735 A bond-length
                * ``"H3"``: H3 molecule at an equilateral triangle of side 0.9 A

            Default is ``"H2"``.
        noise_mode: The noise model used in the simulation of noisy short term evolutions
            Choices are:

                * ``"noiseless"``: Which will generate no noise
                * ``"reduced"``: Uses a low noise profile
                * One of the IBM runtimes such as "ibm_brisbane".

            Default is ``"reduced"``. The available runtime backends can be found using
            :class:`qiskit_ibm_runtime.QiskitRuntimeService.backends`
        formatting: The format in which datapoints are given.
            Choices are:

                * ``"ndarray"``: gives a numpy array of shape (n_points, 2**n_qubits, 1)
                * ``"statevector"``: gives a python list of Statevector objects

            Default is ``"ndarray"``.

    Returns:
        Tuple
        containing the following:

        * **Hartree-Fock State** : ``np.ndarray`` | ``qiskit.quantum_info.Statevector``
        * **training_timestamps** : ``np.ndarray``
        * **training_states** : ``np.ndarray`` | ``qiskit.quantum_info.Statevector``
        * **testing_timestamps** : ``np.ndarray``
        * **testing_states** : ``np.ndarray`` | ``qiskit.quantum_info.Statevector``

    """

    # Errors and Warnings
    if delta_t <= 0:
        raise ValueError("delta_t must be positive (atomic-units of time).")

    if not isinstance(train_end, int) or train_end < 1:
        raise ValueError("train_end must be a positive integer.")

    if not isinstance(test_start, int) or test_start < 1:
        raise ValueError("test_start must be a positive integer.")

    if not isinstance(test_end, int) or test_end <= test_start:
        raise ValueError("test_end must be an integer greater than test_start.")

    if molecule not in {"H2", "H3"}:  # H6 disabled for now
        raise ValueError("molecule must be 'H2' or 'H3'; 'H6' is temporarily unsupported.")

    if formatting not in {"ndarray", "statevector"}:
        raise ValueError("formatting must be 'ndarray' or 'statevector'.")

    if test_start <= train_end:
        warnings.warn("Training and testing ranges overlap; this can cause data leakage.")

    backend = None
    if noise_mode not in {"reduced", "noiseless"}:
        try:
            service = QiskitRuntimeService()
            # real, operational, ≥4-qubit devices
            backends = service.backends(min_num_qubits=4, operational=True, simulator=False)
            backend_names = [b.name for b in backends]  # list-comprehension
            allowed_modes = backend_names + ["reduced", "noiseless"]

            if noise_mode not in allowed_modes:
                raise ValueError(
                    f"'{noise_mode}' is not available. " f"Choose from {allowed_modes}"
                )

            backend = service.backend(noise_mode)
        except Exception as exc:
            raise RuntimeError(
                "Unable to fetch IBM backends; check your internet connection "
                "and IBM Quantum account configuration."
            ) from exc

    # Electron Occupancy
    occupancy = {"H2": 2, "H3": 2, "H6": 6}
    num_occupancy = occupancy[molecule]

    # Noise Models for Training Data
    simulator = _noise_simulator(noise_mode, backend)

    # Import Hamiltonian and Unitary Evolution Circuit
    qc, time, hamiltonian = _evolution_circuit(molecule)
    qc_evo = qc.assign_parameters({time: delta_t})

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
        psi_hf = psi_hf.probabilities()

    return (psi_hf, x_train, y_train, x_test, y_test)


def _evolution_circuit(molecule):
    """Get the parametrized circuit for evolution after Trotterization.
    Returns:
    - QuantumCircuit (for training set)
    - Parameter Object "t" (for training set)
    - Original Hamiltonian (for testing set)"""

    spo = _hamiltonian_import(molecule)

    time = Parameter("time")
    trotterizer = SuzukiTrotter(order=2, reps=1)
    u_evolution = PauliEvolutionGate(spo, time=time, synthesis=trotterizer)

    n_qubits = spo.num_qubits
    qc = QuantumCircuit(n_qubits)
    qc.append(u_evolution, range(n_qubits))

    qc_flat = qc.decompose()

    return qc_flat, time, spo


def _hamiltonian_import(molecule):
    """Import Hamiltonian from Hamiltonians folder"""

    dir_path = os.path.dirname(__file__)
    filename = os.path.join(dir_path, f"hamiltonians/h_molecule_hamiltonians/{molecule}.bin")

    with open(filename, "rb") as ham_file:
        spo = pkl.load(ham_file)

    return spo


def _initial_state(hamiltonian, num_occupancy):
    """Sets a realistic initial state

    JW map automatically keeps orbitals in ascending order of energy"""

    n_qubits = hamiltonian.num_qubits

    bitstring = ["1"] * num_occupancy + ["0"] * (n_qubits - num_occupancy)

    occupation_label = "".join(bitstring)

    return Statevector.from_label(occupation_label)


def _noise_simulator(noise_mode, backend):
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
        noise_model = NoiseModel.from_backend(backend)

    simulator = AerSimulator(noise_model=noise_model)
    return simulator


def _simulate_shortterm(psi_hf, qc_evo, simulator, train_end):
    """Simulates short-term dynamics using a noisy simulator."""

    y_train = [
        psi_hf,
    ]
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


def _ideal_longterm(psi_hf, hamiltonian, timestamps):
    """
    Return the list of statevectors  exp(-i H t_k) @ psi_hf
    for every t_k in `times`, using an exact matrix exponential.
    """
    h_dense = hamiltonian.to_matrix()
    y_test = []

    for t_k in timestamps:
        u_t = expm(-1j * h_dense * t_k)
        psi_t = Statevector(u_t @ psi_hf.data)
        y_test.append(psi_t)

    return y_test


def _to_np(states):
    """Convert list[Statevector] to ndarray"""
    dim = len(states[0])
    return np.stack([sv.data for sv in states], axis=0).reshape(len(states), dim, 1)
