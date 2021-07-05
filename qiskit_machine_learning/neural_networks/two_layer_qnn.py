# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
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
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import PauliSumOp, StateFn, OperatorBase, ExpectationBase
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

from .opflow_qnn import OpflowQNN
from ..exceptions import QiskitMachineLearningError


class TwoLayerQNN(OpflowQNN):
    """Two Layer Quantum Neural Network consisting of a feature map, a ansatz,
    and an observable.
    """

    def __init__(
        self,
        num_qubits: int = None,
        feature_map: QuantumCircuit = None,
        ansatz: QuantumCircuit = None,
        observable: Optional[OperatorBase] = None,
        exp_val: Optional[ExpectationBase] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
        input_gradients: bool = False,
    ):
        r"""
        Args:
            num_qubits: The number of qubits to represent the network, if None and neither the
                feature_map not the ansatz are given, raise exception.
            feature_map: The (parametrized) circuit to be used as feature map. If None is given,
                the `ZZFeatureMap` is used.
            ansatz: The (parametrized) circuit to be used as ansatz. If None is given,
                the `RealAmplitudes` circuit is used.
            observable: observable to be measured to determine the output of the network. If None
                is given, the `Z^{\otimes num_qubits}` observable is used.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        Raises:
            QiskitMachineLearningError: In case of inconsistent num_qubits, feature_map, ansatz.
        """

        # check num_qubits, feature_map, and ansatz
        if num_qubits is None and feature_map is None and ansatz is None:
            raise QiskitMachineLearningError(
                "Need at least one of num_qubits, feature_map, or ansatz!"
            )
        num_qubits_: int = None
        feature_map_: QuantumCircuit = None
        ansatz_: QuantumCircuit = None
        if num_qubits:
            num_qubits_ = num_qubits
            if feature_map:
                if feature_map.num_qubits != num_qubits:
                    raise QiskitMachineLearningError("Incompatible num_qubits and feature_map!")
                feature_map_ = feature_map
            else:
                feature_map_ = ZZFeatureMap(num_qubits)
            if ansatz:
                if ansatz.num_qubits != num_qubits:
                    raise QiskitMachineLearningError("Incompatible num_qubits and ansatz!")
                ansatz_ = ansatz
            else:
                ansatz_ = RealAmplitudes(num_qubits)
        else:
            if feature_map and ansatz:
                if feature_map.num_qubits != ansatz.num_qubits:
                    raise QiskitMachineLearningError("Incompatible feature_map and ansatz!")
                feature_map_ = feature_map
                ansatz_ = ansatz
                num_qubits_ = feature_map.num_qubits
            elif feature_map:
                num_qubits_ = feature_map.num_qubits
                feature_map_ = feature_map
                ansatz_ = RealAmplitudes(num_qubits_)
            elif ansatz:
                num_qubits_ = ansatz.num_qubits
                ansatz_ = ansatz
                feature_map_ = ZZFeatureMap(num_qubits_)

        self._feature_map = feature_map_
        input_params = list(self._feature_map.parameters)

        self._ansatz = ansatz_
        weight_params = list(self._ansatz.parameters)

        # construct circuit
        self._circuit = QuantumCircuit(num_qubits_)
        self._circuit.append(self._feature_map, range(num_qubits_))
        self._circuit.append(self._ansatz, range(num_qubits_))

        # construct observable
        self.observable = (
            observable if observable else PauliSumOp.from_list([("Z" * num_qubits_, 1)])
        )

        # combine all to operator
        operator = ~StateFn(self.observable) @ StateFn(self._circuit)

        super().__init__(
            operator,
            input_params,
            weight_params,
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
