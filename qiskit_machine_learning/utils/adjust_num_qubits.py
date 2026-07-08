# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2024, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Helper functions to derive the number of qubits, feature map, and ansatz."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map

from ..exceptions import QiskitMachineLearningError


# pylint: disable=invalid-name
def derive_num_qubits_feature_map_ansatz(
    num_qubits: int | None = None,
    feature_map: QuantumCircuit | None = None,
    ansatz: QuantumCircuit | None = None,
) -> tuple[int, QuantumCircuit, QuantumCircuit]:
    """
    Derive the number of qubits, feature map, and ansatz from the parameters.

    All provided arguments must agree on the number of qubits. If only some are
    provided, the missing ones are constructed at the agreed qubit count using
    :func:`~qiskit.circuit.library.zz_feature_map` (or
    :func:`~qiskit.circuit.library.z_feature_map` for a single qubit) and
    :func:`~qiskit.circuit.library.real_amplitudes`.

    Args:
        num_qubits: Number of qubits.
        feature_map: A feature map.
        ansatz: An ansatz.

    Returns:
        A tuple of number of qubits, feature map, and ansatz.

    Raises:
        QiskitMachineLearningError: If no arguments are provided, or if the
            provided arguments disagree on the number of qubits.
    """
    counts: dict[str, int] = {}
    if num_qubits is not None:
        counts["num_qubits"] = num_qubits
    if feature_map is not None:
        counts["feature_map"] = feature_map.num_qubits
    if ansatz is not None:
        counts["ansatz"] = ansatz.num_qubits

    if not counts:
        raise QiskitMachineLearningError(
            "Unable to determine number of qubits: provide `num_qubits` (int), "
            "`feature_map` (QuantumCircuit), or `ansatz` (QuantumCircuit)."
        )

    unique_counts = set(counts.values())
    if len(unique_counts) > 1:
        details = ", ".join(f"{k}={v}" for k, v in counts.items())
        raise QiskitMachineLearningError(
            f"Inconsistent qubit counts: {details}. "
            "Adjust the inputs to match before passing them as arguments."
        )

    resolved = next(iter(unique_counts))

    if feature_map is None:
        feature_map = z_feature_map(resolved) if resolved == 1 else zz_feature_map(resolved)
    if ansatz is None:
        ansatz = real_amplitudes(resolved)

    return resolved, feature_map, ansatz
