# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QNN circuit."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit

from ...utils import derive_num_qubits_feature_map_ansatz


def qnn_circuit(
    num_qubits: int | None = None,
    feature_map: QuantumCircuit | None = None,
    ansatz: QuantumCircuit | None = None,
):
    """
    The qnn_circuit creates a QuantumCircuit that is a composition of a feature map
    and an ansatz circuit. Also returned are the feature map and ansatz parameters for
    use for inputs and weights as needed for a neural network, such as
    :class:`~qiskit-machine-learning.neural_networks.SamplerQNN`.

    If only the number of qubits is provided the :meth:`~qiskit.circuit.library.real_amplitudes`
    ansatz and the :meth:`~qiskit.circuit.library.zz_feature_map` feature map are used. If the
    number of qubits is 1 the :meth:`~qiskit.circuit.library.z_feature_map` is used. If only a
    feature map is provided, the :meth:`~qiskit.circuit.library.real_amplitudes` ansatz with the
    corresponding number of qubits is used. If only an ansatz is provided the
    :meth:`~qiskit.circuit.library.zz_feature_map` with the corresponding number of qubits is used.

    At least one parameter has to be provided. If a feature map and an ansatz is provided, the
    number of qubits must be the same.

    Example:

    .. code-block:: python

        from qiskit_machine_learning.circuit.library import qnn_circuit
        qnn_qc, fm_params, anz_params = qnn_circuit(2)
        qnn_qc.draw(fold=60)
        #      ┌───┐┌─────────────┐     »
        # q_0: ┤ H ├┤ P(2.0*x[0]) ├──■──»
        #      ├───┤├─────────────┤┌─┴─┐»
        # q_1: ┤ H ├┤ P(2.0*x[1]) ├┤ X ├»
        #      └───┘└─────────────┘└───┘»
        # «                                          ┌───┐»
        # «q_0: ──────────────────────────────────■──┤ H ├»
        # «     ┌──────────────────────────────┐┌─┴─┐├───┤»
        # «q_1: ┤ P(2.0*(x[0] - π)*(x[1] - π)) ├┤ X ├┤ H ├»
        # «     └──────────────────────────────┘└───┘└───┘»
        # «     ┌─────────────┐                                     »
        # «q_0: ┤ P(2.0*x[0]) ├──■──────────────────────────────────»
        # «     ├─────────────┤┌─┴─┐┌──────────────────────────────┐»
        # «q_1: ┤ P(2.0*x[1]) ├┤ X ├┤ P(2.0*(x[0] - π)*(x[1] - π)) ├»
        # «     └─────────────┘└───┘└──────────────────────────────┘»
        # «          ┌──────────┐     ┌──────────┐     ┌──────────┐»
        # «q_0: ──■──┤ Ry(θ[0]) ├──■──┤ Ry(θ[2]) ├──■──┤ Ry(θ[4]) ├»
        # «     ┌─┴─┐├──────────┤┌─┴─┐├──────────┤┌─┴─┐├──────────┤»
        # «q_1: ┤ X ├┤ Ry(θ[1]) ├┤ X ├┤ Ry(θ[3]) ├┤ X ├┤ Ry(θ[5]) ├»
        # «     └───┘└──────────┘└───┘└──────────┘└───┘└──────────┘»
        # «          ┌──────────┐
        # «q_0: ──■──┤ Ry(θ[6]) ├
        # «     ┌─┴─┐├──────────┤
        # «q_1: ┤ X ├┤ Ry(θ[7]) ├
        # «     └───┘└──────────┘
        print(fm_params)
        # ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])
        print(anz_params)
        # ParameterView([ParameterVectorElement(θ[0]), ParameterVectorElement(θ[1]),
        #                ParameterVectorElement(θ[2]), ParameterVectorElement(θ[3]),
        #                ParameterVectorElement(θ[4]), ParameterVectorElement(θ[5]),
        #                ParameterVectorElement(θ[6]), ParameterVectorElement(θ[7])])

    Although all arguments to qnn_circuit default to None at least one must be provided,
    to determine the number of qubits from.

    If more than one is passed:

    1) If num_qubits is provided the feature map and/or ansatz circuits supplied must be the
    same number of qubits.

    2) If num_qubits is not provided the feature_map and ansatz must be set to the same number
    of qubits.

    Args:
        num_qubits:  Number of qubits, a positive integer. Optional if feature_map or ansatz is
                     provided, otherwise required. If not provided num_qubits defaults from the
                     sizes of feature_map and/or ansatz.
        feature_map: A feature map. Optional if num_qubits or ansatz is provided, otherwise
                     required. If not provided defaults to
                     :meth:`~qiskit.circuit.library.zz_feature_map` or
                     :meth:`~qiskit.circuit.library.z_feature_map` if num_qubits is determined
                     to be 1.
        ansatz:      An ansatz. Optional if num_qubits or feature_map is provided, otherwise
                     required. If not provided defaults to
                     :meth:`~qiskit.circuit.library.real_amplitudes`.

    Returns:
        The composed feature map and ansatz circuit, the feature map parameters and the
        ansatz parameters.

    Raises:
        QiskitMachineLearningError: If a valid number of qubits cannot be derived from the \
        provided input arguments.
    """
    # Check if circuit is constructed with valid configuration and set properties accordingly.
    num_qubits, feature_map, ansatz = derive_num_qubits_feature_map_ansatz(
        num_qubits, feature_map, ansatz
    )
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc, feature_map.parameters, ansatz.parameters
