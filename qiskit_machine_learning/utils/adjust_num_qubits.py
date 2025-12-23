# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Helper functions to adjust number of qubits."""
from __future__ import annotations

import warnings
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map

from ..exceptions import QiskitMachineLearningError
from ..utils.deprecation import issue_deprecation_msg


# pylint: disable=invalid-name
def derive_num_qubits_feature_map_ansatz(
    num_qubits: int | None = None,
    feature_map: QuantumCircuit | None = None,
    ansatz: QuantumCircuit | None = None,
    use_methods: bool = True,
) -> tuple[int, QuantumCircuit, QuantumCircuit]:
    """
    Derives a correct number of qubits, feature map, and ansatz from the parameters.

    If the number of qubits is not ``None``, then the feature map and ansatz are adjusted to this
    number of qubits if required. If such an adjustment fails, an error is raised. Also, if the
    feature map or ansatz or both are ``None``, then :func:`~qiskit.circuit.library.zz_feature_map`
    and :func:`~qiskit.circuit.library.real_amplitudes` are created respectively. If there's just
    one qubit, :func:`~qiskit.circuit.library.z_feature_map` is created instead.

    If the number of qubits is ``None``, then the number of qubits is derived from the feature map
    or ansatz. Both the feature map and ansatz in this case must have the same number of qubits.
    If the number of qubits of the feature map is not the same as the number of qubits of
    the ansatz, an error is raised. If only one of the feature map and ansatz are ``None``, then

    :func:`~qiskit.circuit.library.zz_feature_map` or :func:`~qiskit.circuit.library.real_amplitudes`
    are created respectively.

    With `use_methods` set True:

    If the number of qubits is not ``None``, then the feature map and ansatz are adjusted to this
    number of qubits if required. If such an adjustment fails, an error is raised. Also, if the
    feature map or ansatz or both are ``None``, then :meth:`~qiskit.circuit.library.zz_feature_map`
    and :meth:`~qiskit.circuit.library.real_amplitudes` are created respectively. If there's just
    one qubit, :meth:`~qiskit.circuit.library.z_feature_map` is created instead.

    If the number of qubits is ``None``, then the number of qubits is derived from the feature map
    or ansatz. Both the feature map and ansatz in this case must have the same number of qubits.
    If the number of qubits of the feature map is not the same as the number of qubits of
    the ansatz, an error is raised. If only one of the feature map and ansatz are ``None``, then
    :meth:`~qiskit.circuit.library.zz_feature_map` or :class:`~qiskit.circuit.library.real_amplitudes`
    are created respectively.

    If all the parameters are none an error is raised.

    Args:
        num_qubits: Number of qubits.
        feature_map: A feature map.
        ansatz: An ansatz.
        use_methods: weather to use the method implementation of circuits (Qiskit >=2) or the class
            implementation (deprecated in Qiskit 2 and will be removed in Qiskit 3).

    Returns:
        A tuple of number of qubits, feature map, and ansatz. All are not none.

    Raises:
        QiskitMachineLearningError: If correct values can not be derived from the parameters.
    """

    if not use_methods:
        issue_deprecation_msg(
            msg="Using BlueprintCircuit based classes is deprecated",
            version="0.9.0",
            remedy="Use QnnCircuit (instead) of QNNCircuit or if you "
            "are using this method directly set use_methods to True. "
            "When using methods later adjustment of the number of qubits is not "
            "possible and if not as circuits based on BlueprintCircuit, "
            "like ZZFeatureMap to which this defaults, which could do this, "
            "have been deprecated.",
            period="4 months",
        )
    candidates = {}

    if feature_map is not None:
        candidates["feature_map"] = feature_map.num_qubits
    if ansatz is not None:
        candidates["ansatz"] = ansatz.num_qubits
    if num_qubits is not None:
        candidates["num_qubits"] = num_qubits

    if not candidates:
        raise QiskitMachineLearningError(
            "Unable to determine number of qubits: "
            "provide `num_qubits` (int), `feature_map` (QuantumCircuit), "
            "or `ansatz` (QuantumCircuit)."
        )

    # Check consensus on num_qubits
    unique_vals = set(candidates.values())
    if len(unique_vals) > 1:
        conflicts = ", ".join(f"{k}={v}" for k, v in candidates.items())
        warnings.warn(
            (
                f"Inconsistent qubit numbers detected: {conflicts}. "
                "Ensure all inputs agree on the number of qubits."
            ),
            UserWarning,
        )

    # Final resolved number of qubits
    resolved_num_qubits = max(unique_vals)

    def default_feature_map(n: int) -> QuantumCircuit:
        return z_feature_map(n) if n == 1 else zz_feature_map(n)

    def default_ansatz(n: int) -> QuantumCircuit:
        return real_amplitudes(n)

    if feature_map is None:
        feature_map = default_feature_map(resolved_num_qubits)
        candidates["feature_map"] = feature_map.num_qubits
    else:
        feature_map = _pad_if_needed(feature_map, resolved_num_qubits)

    if ansatz is None:
        ansatz = default_ansatz(resolved_num_qubits)
        candidates["ansatz"] = ansatz.num_qubits
    else:
        ansatz = _pad_if_needed(ansatz, resolved_num_qubits)

    # Mismatch in the circuits' num_qubits is unacceptable
    if candidates["feature_map"] != candidates["ansatz"]:
        raise QiskitMachineLearningError(
            f"Inconsistent qubit numbers detected between the feature map ({candidates['feature_map']}) "
            f"and the ansatz ({candidates['ansatz']}). These must match at all times."
        )

    return resolved_num_qubits, feature_map, ansatz


def _pad_if_needed(circ: QuantumCircuit, requested_num_qubits: int) -> QuantumCircuit | None:
    circ_nq = circ.num_qubits

    if requested_num_qubits == circ_nq:
        return circ

    if requested_num_qubits < circ_nq:
        raise QiskitMachineLearningError(
            f"Requesting num_qubits={requested_num_qubits} to a circuit with {circ_nq} qubits. "
            f"Circuit cutting is not supported by default. Please, remove qubit registers manually."
        )

    warnings.warn(
        (
            f"Requesting num_qubits={requested_num_qubits} to a circuit with {circ_nq} qubits. "
            f"Padding with {requested_num_qubits - circ_nq} idle qubits."
        ),
        UserWarning,
    )
    padded = QuantumCircuit(requested_num_qubits, circ.num_clbits, name=circ.name)
    padded.compose(circ, inplace=True)
    return padded


# pylint: disable=unused-argument
def _adjust_num_qubits(circuit: QuantumCircuit, circuit_name: str, num_qubits: int) -> None:
    """
    Tries to adjust the number of qubits of the circuit by trying to set ``num_qubits`` properties.

    Args:
        circuit: A circuit to adjust.
        circuit_name: A circuit name, used in the error description.
        num_qubits: A number of qubits to set.

    Raises:
        QiskitMachineLearningError: if number of qubits can't be adjusted.

    """
    issue_deprecation_msg(
        msg="No longer in use",
        version="0.9.0",
        remedy="Check ",
        period="0 months",
    )
