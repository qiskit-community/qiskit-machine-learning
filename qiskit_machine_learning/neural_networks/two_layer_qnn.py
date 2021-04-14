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
from qiskit.opflow import PauliSumOp, StateFn, OperatorBase, ExpectationBase
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.utils.num_qubits_helper import _retrieve_arguments_if_none
from .opflow_qnn import OpflowQNN


class TwoLayerQNN(OpflowQNN):
    """Two Layer Quantum Neural Network consisting of a feature map, a ansatz,
    and an observable.
    """

    def __init__(self, num_qubits: int = None,
                 feature_map: QuantumCircuit = None,
                 ansatz: QuantumCircuit = None,
                 observable: Optional[OperatorBase] = None,
                 exp_val: Optional[ExpectationBase] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):
        r"""Initializes the Two Layer Quantum Neural Network.

        Args:
            num_qubits: The number of qubits to represent the network, if None and neither the
                feature_map not the ansatz are given, raise exception.
            feature_map: The (parametrized) circuit to be used as feature map. If None is given,
                the `ZZFeatureMap` is used.
            ansatz: The (parametrized) circuit to be used as ansatz. If None is given,
                the `RealAmplitudes` circuit is used.
            observable: observable to be measured to determine the output of the network. If None
                is given, the `Z^{\otimes num_qubits}` observable is used.
            quantum_instance: Quantum Instance or Backend or BaseBackend.

        Raises:
            QiskitMachineLearningError: In case of inconsistent num_qubits, feature_map, ansatz.
        """

        # check num_qubits, feature_map, and ansatz
        self._ansatz, self._feature_map, num_qubits_ = _retrieve_arguments_if_none(ansatz,
                                                                                   feature_map,
                                                                                   num_qubits)

        input_params = list(self._feature_map.parameters)
        weight_params = list(self._ansatz.parameters)

        # construct circuit
        self._circuit = QuantumCircuit(num_qubits_)
        self._circuit.append(self._feature_map, range(num_qubits_))
        self._circuit.append(self._ansatz, range(num_qubits_))

        # construct observable
        self.observable = observable if observable else PauliSumOp.from_list(
            [('Z' * num_qubits_, 1)])

        # combine all to operator
        operator = ~StateFn(self.observable) @ StateFn(self._circuit)

        super().__init__(operator, input_params, weight_params, quantum_instance=quantum_instance)

    @property
    def feature_map(self) -> QuantumCircuit:
        """ Returns the used feature map."""
        return self._feature_map

    @property
    def ansatz(self) -> QuantumCircuit:
        """ Returns the used ansatz."""
        return self._ansatz

    @property
    def circuit(self) -> QuantumCircuit:
        """ Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """ Returns the number of qubits used by ansatz and feature map."""
        return self._circuit.num_qubits
