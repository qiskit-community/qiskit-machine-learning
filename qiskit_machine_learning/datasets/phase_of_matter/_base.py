# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shared utilities for Phase of Matter dataset generators."""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from qiskit.quantum_info import SparsePauliOp, Statevector


def pauli_term(op_list: list[tuple[str, int]], n: int) -> SparsePauliOp:
    """Build a single n-qubit Pauli term from a list of (pauli_char, site) pairs.

    Sites not listed are identity. Uses Qiskit's little-endian convention:
    site 0 is the rightmost character in the Pauli string.

    Args:
        op_list: List of (Pauli character, qubit site index) pairs.
        n: Total number of qubits.

    Returns:
        SparsePauliOp representing the term.
    """
    chars = ["I"] * n
    for pauli_char, site in op_list:
        chars[site] = pauli_char
    return SparsePauliOp("".join(reversed(chars)))


def _canonicalize_phase(vec: np.ndarray) -> np.ndarray:
    """Fix the global phase so that the leading large-magnitude element is real positive.

    Eigenvectors are defined only up to a global complex phase; this
    canonicalization makes repeated calls to ``eigsh`` return numerically
    identical arrays for the same Hamiltonian.
    """
    threshold = 1e-10 * np.max(np.abs(vec))
    for val in vec:
        if abs(val) > threshold:
            return vec * (np.conj(val) / abs(val))
    return vec


def get_ground_state_exact(hamiltonian: SparsePauliOp) -> np.ndarray:
    """Return the ground-state vector via sparse exact diagonalization.

    Uses ``scipy.sparse.linalg.eigsh`` with ``which='SA'`` (smallest algebraic
    eigenvalue).  Practical limit: n ≤ 16 qubits (2^16 × 2^16 matrix).

    The returned vector is phase-canonicalized so that repeated calls for the
    same Hamiltonian yield identical arrays.

    Args:
        hamiltonian: Hamiltonian as a SparsePauliOp.

    Returns:
        Complex numpy array of shape ``(2**n,)`` — the normalised ground state.
    """
    mat = hamiltonian.to_matrix(sparse=True).astype(complex)
    _, vecs = scipy.sparse.linalg.eigsh(mat, k=1, which="SA")
    return _canonicalize_phase(vecs[:, 0])


def get_ground_state_vqe(
    hamiltonian: SparsePauliOp,
    backend,  # pylint: disable=unused-argument
) -> Statevector:
    """Approximate the ground state via VQE using qiskit primitives.

    .. warning::

        VQE is provided for hardware-experiment workflows only.  For reliable
        phase labels, use the default exact diagonalization (``backend=None``).
        VQE approximations near phase boundaries may produce incorrect labels.

    Uses an ``EfficientSU2`` ansatz (1 repetition) with COBYLA optimisation via
    ``StatevectorEstimator`` from ``qiskit.primitives``.  The ``backend``
    argument is accepted for API consistency and future hardware integration;
    the current implementation uses ``StatevectorEstimator`` unconditionally.

    Args:
        hamiltonian: Hamiltonian as a SparsePauliOp.
        backend: Reserved for future hardware integration.  Currently unused;
            pass any non-``None`` value to activate this pathway.

    Returns:
        Qiskit ``Statevector`` of the approximate ground state.
    """
    # Deferred imports so qiskit-aer is only required when VQE is used.
    from qiskit.circuit.library import EfficientSU2  # pylint: disable=import-outside-toplevel
    from qiskit.primitives import StatevectorEstimator  # pylint: disable=import-outside-toplevel
    from scipy.optimize import minimize  # pylint: disable=import-outside-toplevel

    n = hamiltonian.num_qubits
    ansatz = EfficientSU2(n, reps=1, entanglement="linear")
    num_params = ansatz.num_parameters
    estimator = StatevectorEstimator()

    def cost(params: np.ndarray) -> float:
        pub = (ansatz, [hamiltonian], [params])
        return float(estimator.run([pub]).result()[0].data.evs[0])

    rng = np.random.default_rng(0)
    x0 = rng.uniform(-np.pi, np.pi, num_params)
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.5})
    return Statevector(ansatz.assign_parameters(result.x))
