# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Two Layer Neural Network consisting of a first parametrized circuit representing a feature map
to map the input data to a quantum states and a second one representing a ansatz that can
be trained to solve a particular tasks."""

from typing import Optional, Union

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp, StateFn, OperatorBase, ExpectationBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

from .opflow_qnn import OpflowQNN
from ..utils import derive_num_qubits_feature_map_ansatz


class TwoLayerQNN(OpflowQNN):
    """Two Layer Quantum Neural Network consisting of a feature map, a ansatz,
    and an observable.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        feature_map: Optional[QuantumCircuit] = None,
        ansatz: Optional[QuantumCircuit] = None,
        observable: Optional[OperatorBase] = None,
        exp_val: Optional[ExpectationBase] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        input_gradients: bool = False,
    ):
        r"""
        Args:
            num_qubits: The number of qubits to represent the network, if None and neither the
                feature_map not the ansatz are given, raise exception.
            feature_map: The (parametrized) circuit to be used as feature map. If None is given,
                the ``ZZFeatureMap`` is used if the number of qubits is larger than 1. For
                a single qubit two-layer QNN the ``ZFeatureMap`` is used per default.
            ansatz: The (parametrized) circuit to be used as ansatz. If None is given,
                the ``RealAmplitudes`` circuit is used.
            observable: observable to be measured to determine the output of the network. If None
                is given, the :math:`Z^{\otimes num_qubits}` observable is used.
            exp_val: The Expected Value converter to be used for the operator obtained from the
                feature map and ansatz.
            quantum_instance: The quantum instance to evaluate the network.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        Raises:
            QiskitMachineLearningError: In case of inconsistent num_qubits, feature_map, ansatz.
        """

        num_qubits, feature_map, ansatz = derive_num_qubits_feature_map_ansatz(
            num_qubits, feature_map, ansatz
        )

        self._feature_map = feature_map
        input_params = list(self._feature_map.parameters)

        self._ansatz = ansatz
        weight_params = list(self._ansatz.parameters)

        # construct circuit
        self._circuit = QuantumCircuit(num_qubits)
        self._circuit.append(self._feature_map, range(num_qubits))
        self._circuit.append(self._ansatz, range(num_qubits))

        # construct observable
        self.observable = (
            observable if observable is not None else PauliSumOp.from_list([("Z" * num_qubits_, 1)])
        )

        # combine all to operator
        operator = StateFn(self.observable, is_measurement=True) @ StateFn(self._circuit)

        super().__init__(
            operator=operator,
            input_params=input_params,
            weight_params=weight_params,
            exp_val=exp_val,
            quantum_instance=quantum_instance,
            input_gradients=input_gradients,
        )

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the used feature map."""
        return self._feature_map

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the used ansatz."""
        return self._ansatz

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return self._circuit.num_qubits
