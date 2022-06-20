# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Helper functions to adjust number of qubits."""

from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from qiskit_machine_learning import QiskitMachineLearningError


# pylint: disable=invalid-name
def derive_num_qubits_feature_map_ansatz(
    num_qubits: int = None, feature_map: QuantumCircuit = None, ansatz: QuantumCircuit = None
) -> Tuple[int, QuantumCircuit, QuantumCircuit]:
    """
    Derives a correct number of qubits, feature map, and ansatz from the parameters.

    If the number of qubits is not ``None``, then the feature map and ansatz are adjusted to this
    number of qubits if required. If such an adjustment fails, an error is raised.
    Also, if the feature map or ansatz or both are ``None``, then ``ZZFeatureMap`` and
    ``RealAmplitudes`` are created respectively.

    If the number of qubits is ``None``, then the number of qubits is derived from the feature map
    or ansatz. Both the feature map and ansatz in this case must have the same number of qubits.
    If the number of qubits of the feature map is not the same as the number of qubits of
    the ansatz, an error is raised. If only one of the feature map and ansatz are ``None``, then
    ``ZZFeatureMap`` or ``RealAmplitudes`` are created respectively.

    If all the parameters are none an error is raised.

    Args:
        num_qubits: Number of qubits.
        feature_map: A feature map.
        ansatz: An ansatz.

    Returns:
        A tuple of number of qubits, feature map, and ansatz. All are not none.

    Raises:
        QiskitMachineLearningError: If correct values can not be derived from the parameters.
    """
    # check num_qubits, feature_map, and ansatz
    if num_qubits is None and feature_map is None and ansatz is None:
        raise QiskitMachineLearningError(
            "Need at least one of number of qubits, feature map, or ansatz!"
        )

    if num_qubits is not None:
        if feature_map is not None:
            if feature_map.num_qubits != num_qubits:
                _adjust_num_qubits(feature_map, "feature map", num_qubits)
        else:
            feature_map = ZZFeatureMap(num_qubits)
        if ansatz is not None:
            if ansatz.num_qubits != num_qubits:
                _adjust_num_qubits(ansatz, "ansatz", num_qubits)
        else:
            ansatz = RealAmplitudes(num_qubits)
    else:
        if feature_map is not None and ansatz is not None:
            if feature_map.num_qubits != ansatz.num_qubits:
                raise QiskitMachineLearningError("Incompatible feature_map and ansatz!")
            num_qubits = feature_map.num_qubits
        elif feature_map:
            num_qubits = feature_map.num_qubits
            ansatz = RealAmplitudes(num_qubits)
        elif ansatz:
            num_qubits = ansatz.num_qubits
            feature_map = ZZFeatureMap(num_qubits)

    return num_qubits, feature_map, ansatz


def _adjust_num_qubits(circuit: QuantumCircuit, circuit_name: str, num_qubits: int):
    """
    Tries to adjust the number of qubits of the circuit by trying to set ``num_qubits`` properties.

    Args:
        circuit: A circuit to adjust.
        circuit_name: A circuit name, used in the error description.
        num_qubits: A number of qubits to set.

    Raises:
        QiskitMachineLearningError: if number of qubits can't be adjusted.

    """
    try:
        circuit.num_qubits = num_qubits
    except AttributeError as ex:
        raise QiskitMachineLearningError(
            f"The number of qubits {circuit.num_qubits} of the {circuit_name} does not match "
            f"the number of qubits {num_qubits}, and the {circuit_name} does not allow setting "
            "the number of qubits using `num_qubits`."
        ) from ex
