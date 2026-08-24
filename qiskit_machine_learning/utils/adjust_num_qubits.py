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
"""Helper functions to derive feature map and ansatz circuits."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map

from ..exceptions import QiskitMachineLearningError


def derive_num_qubits_feature_map_ansatz(
    feature_map: QuantumCircuit | None = None,
    ansatz: QuantumCircuit | None = None,
) -> tuple[int, QuantumCircuit, QuantumCircuit]:
    """
    Derives a correct number of qubits, feature map, and ansatz from the parameters.

    If the feature map or ansatz or both are ``None``, then
    :func:`~qiskit.circuit.library.zz_feature_map` and
    :func:`~qiskit.circuit.library.real_amplitudes` are created respectively. If there's just
    one qubit, :func:`~qiskit.circuit.library.z_feature_map` is created instead.

    If both the feature map and ansatz are provided, they must have the same number of qubits.

    If all the parameters are ``None`` an error is raised.

    Args:
        feature_map: A feature map.
        ansatz: An ansatz.

    Returns:
        A tuple of number of qubits, feature map, and ansatz.

    Raises:
        QiskitMachineLearningError: If correct values can not be derived from the parameters.
    """
    if feature_map is None and ansatz is None:
        raise QiskitMachineLearningError(
            "Unable to determine number of qubits: "
            "provide `feature_map` (QuantumCircuit) and/or `ansatz` (QuantumCircuit)."
        )

    if feature_map is not None and ansatz is not None:
        if feature_map.num_qubits != ansatz.num_qubits:
            raise QiskitMachineLearningError(
                f"Inconsistent qubit numbers detected between the feature map "
                f"({feature_map.num_qubits}) and the ansatz ({ansatz.num_qubits}). "
                "Construct both circuits with the same number of qubits before passing them."
            )
        resolved_num_qubits = feature_map.num_qubits
    elif feature_map is not None:
        resolved_num_qubits = feature_map.num_qubits
    else:
        resolved_num_qubits = ansatz.num_qubits

    def default_feature_map(num_qubits: int) -> QuantumCircuit:
        return z_feature_map(num_qubits) if num_qubits == 1 else zz_feature_map(num_qubits)

    def default_ansatz(num_qubits: int) -> QuantumCircuit:
        return real_amplitudes(num_qubits)

    if feature_map is None:
        feature_map = default_feature_map(resolved_num_qubits)
    if ansatz is None:
        ansatz = default_ansatz(resolved_num_qubits)

    return resolved_num_qubits, feature_map, ansatz
