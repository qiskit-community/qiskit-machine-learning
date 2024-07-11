# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
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
from typing import List

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit.library import BlueprintCircuit

from qiskit_machine_learning import QiskitMachineLearningError

from ...utils import derive_num_qubits_feature_map_ansatz


class QNNCircuit(BlueprintCircuit):
    """
    The QNN circuit is a blueprint circuit that wraps feature map and ansatz circuits.
    It can be used to simplify the composition of these two.

    If only the number of qubits is provided the :class:`~qiskit.circuit.library.RealAmplitudes`
    ansatz and the :class:`~qiskit.circuit.library.ZZFeatureMap` feature map are used. If the
    number of qubits is 1 the :class:`~qiskit.circuit.library.ZFeatureMap` is used. If only a
    feature map is provided, the :class:`~qiskit.circuit.library.RealAmplitudes` ansatz with the
    corresponding number of qubits is used. If only an ansatz is provided the
    :class:`~qiskit.circuit.library.ZZFeatureMap` with the corresponding number of qubits is used.

    At least one parameter has to be provided. If a feature map and an ansatz is provided, the
    number of qubits must be the same.

    In case number of qubits is provided along with either a feature map, an ansatz or both, a
    potential mismatch between the three inputs with respect to the number of qubits is resolved by
    constructing the :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` with the given
    number of qubits. If one of the :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`
    properties is set after the class construction, the circuit is adjusted  to incorporate the
    changes. This means, a new valid configuration that considers the latest property update will be
    derived. This ensures that the classes properties are consistent at all times.

    Example:

    .. code-block:: python

        from qiskit_machine_learning.circuit.library import QNNCircuit
        qnn_qc = QNNCircuit(2)
        print(qnn_qc)
        # prints:
        #      ┌──────────────────────────┐»
        # q_0: ┤0                         ├»
        #      │  ZZFeatureMap(x[0],x[1]) │»
        # q_1: ┤1                         ├»
        #      └──────────────────────────┘»
        # «     ┌──────────────────────────────────────────────────────────┐
        # «q_0: ┤0                                                         ├
        # «     │  RealAmplitudes(θ[0],θ[1],θ[2],θ[3],θ[4],θ[5],θ[6],θ[7]) │
        # «q_1: ┤1                                                         ├
        # «     └──────────────────────────────────────────────────────────┘

        print(qnn_qc.num_qubits)
        # prints: 2

        print(qnn_qc.input_parameters)
        # prints: ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])

        print(qnn_qc.weight_parameters)
        # prints: ParameterView([ParameterVectorElement(θ[0]), ParameterVectorElement(θ[1]),
        #         ParameterVectorElement(θ[2]), ParameterVectorElement(θ[3]),
        #         ParameterVectorElement(θ[4]), ParameterVectorElement(θ[5]),
        #         ParameterVectorElement(θ[6]), ParameterVectorElement(θ[7])])
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | None = None,
        ansatz: QuantumCircuit | None = None,
    ) -> None:
        """
        Although all parameters default to None at least one parameter must be provided, to determine
        the number of qubits from it, when the instance is created.

        If more than one parameter is passed:

        1) If num_qubits is provided the feature map and/or ansatz supplied will be overridden to
        circuits with num_qubits, as long as the respective circuit supports updating its number of
        qubits.

        2) If num_qubits is not provided the feature_map and ansatz must be set to the same number
        of qubits.

        Args:
            num_qubits:  Number of qubits, a positive integer. Optional if feature_map or ansatz is
                         provided, otherwise required. If not provided num_qubits defaults from the
                         sizes of feature_map and ansatz.
            feature_map: A feature map. Optional if num_qubits or ansatz is provided, otherwise
                         required. If not provided defaults to
                         :class:`~qiskit.circuit.library.ZZFeatureMap` or
                         :class:`~qiskit.circuit.library.ZFeatureMap` if num_qubits is determined
                         to be 1.
            ansatz:      An ansatz. Optional if num_qubits or feature_map is provided, otherwise
                         required. If not provided defaults to
                         :class:`~qiskit.circuit.library.RealAmplitudes`.

        Returns:
            The composed feature map and ansatz circuit.

        Raises:
            QiskitMachineLearningError: If a valid number of qubits cannot be derived from the \
            provided input arguments.
        """

        super().__init__()
        self._feature_map = feature_map
        self._ansatz = ansatz
        # Check if circuit is constructed with valid configuration and set properties accordingly.
        self.num_qubits, self._feature_map, self._ansatz = derive_num_qubits_feature_map_ansatz(
            num_qubits, feature_map, ansatz
        )

    def _build(self):
        super()._build()
        self.compose(self.feature_map, inplace=True)
        self.compose(self.ansatz, inplace=True)

    def _check_configuration(self, raise_on_failure=True):
        try:
            self.num_qubits, self.feature_map, self.ansatz = derive_num_qubits_feature_map_ansatz(
                self.num_qubits, self.feature_map, self.ansatz
            )
        except QiskitMachineLearningError as qml_ex:
            if raise_on_failure:
                raise qml_ex

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return super().num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits. If num_qubits is set
        the feature map and ansatz are adjusted to circuits with num_qubits qubits.

        Args:
            num_qubits:  The number of qubits, a positive integer.
        """
        if self.num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self.qregs: List[QuantumRegister] = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, name="q")]
                (
                    self.num_qubits,
                    self._feature_map,
                    self._ansatz,
                ) = derive_num_qubits_feature_map_ansatz(
                    num_qubits, self._feature_map, self._ansatz
                )

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns feature_map.

        Returns:
            The feature map.
        """
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: QuantumCircuit) -> None:
        """Set the feature map. If the feature map is updated the ``QNNCircuit`` is adjusted
        according to the feature map being passed. This includes:
        1) The num_qubits is adjusted to the feature map number of qubits.
        2) The ansatz is adjusted to a circuit with the feature_map number of qubits.

        Args:
            feature_map: The feature map.
        """
        if self.feature_map != feature_map:
            # invalidate the circuit
            self._invalidate()
            self.num_qubits = feature_map.num_qubits
            self.num_qubits, self._feature_map, self._ansatz = derive_num_qubits_feature_map_ansatz(
                self.num_qubits, feature_map, self.ansatz
            )

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns ansatz.

        Returns:
            The ansatz.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit) -> None:
        """Set the ansatz. If the ansatz is updated the ``QNNCircuit`` is adapted
        according to the ansatz being passed. This includes:
        1) The num_qubits is adjusted to the ansatz number of qubits.
        2) The feature_map is adjusted to a circuit with the ansatz number of qubits.

        Args:
            ansatz: The ansatz.
        """
        if self.ansatz != ansatz:
            # invalidate the circuit
            self._invalidate()
            self.num_qubits = ansatz.num_qubits
            self.num_qubits, self._feature_map, self._ansatz = derive_num_qubits_feature_map_ansatz(
                self.num_qubits, self.feature_map, ansatz
            )

    @property
    def input_parameters(self) -> ParameterView:
        """Returns the parameters of the feature map.

        Returns:
            The parameters of the feature map.
        """
        return self._feature_map.parameters

    @property
    def num_input_parameters(self) -> int:
        """Returns the number of input parameters in the circuit.

        Returns:
            The number of input parameters.
        """
        return len(self._feature_map.parameters)

    @property
    def weight_parameters(self) -> ParameterView:
        """Returns the parameters of the ansatz. These corresponding to the trainable weights.

        Returns:
            The parameters of the ansatz.
        """
        return self._ansatz.parameters

    @property
    def num_weight_parameters(self) -> int:
        """Returns the number of weights in the circuit.

        Returns:
            The number of weights.
        """
        return len(self._ansatz.parameters)
