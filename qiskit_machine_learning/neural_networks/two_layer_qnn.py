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
to map the input data to a quantum states and a second one representing a variational form that can
be trained to solve a particular tasks."""
from typing import Optional, Union

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import PauliSumOp, StateFn, OperatorBase, ExpectationBase
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

from .opflow_qnn import OpflowQNN


class TwoLayerQNN(OpflowQNN):
    """Two Layer Quantum Neural Network consisting of a feature map, a variational form,
    and an observable.
    """

    def __init__(self, num_qubits: int, feature_map: QuantumCircuit = None,
                 var_form: QuantumCircuit = None,
                 observable: Optional[OperatorBase] = None,
                 exp_val: Optional[ExpectationBase] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):
        r"""Initializes the Two Layer Quantum Neural Network.

        Args:
            num_qubits: The number of qubits to represent the network.
            feature_map: The (parametrized) circuit to be used as feature map. If None is given,
                the `ZZFeatureMap` is used.
            var_form: The (parametrized) circuit to be used as variational form. If None is given,
                the `RealAmplitudes` circuit is used.
            observable: observable to be measured to determine the output of the network. If None
                is given, the `Z^{\otimes num_qubits}` observable is used.
            exp_val: The Expected Value converter to be used for the operator.
            quantum_instance: The quantum instance to evaluate the networks.
        """

        # set number of qubits, feature map, and variational form
        self.num_qubits = num_qubits

        # TODO: circuits need to have well-defined parameter order!
        self._feature_map = feature_map if feature_map else ZZFeatureMap(num_qubits)
        idx = np.argsort([p.name for p in self._feature_map.parameters])
        input_params = list(self._feature_map.parameters)
        input_params = [input_params[i] for i in idx]

        # TODO: circuits need to have well-defined parameter order!
        self._var_form = var_form if var_form else RealAmplitudes(num_qubits)
        idx = np.argsort([p.name for p in self._var_form.parameters])
        weight_params = list(self._var_form.parameters)
        weight_params = [weight_params[i] for i in idx]

        # construct circuit
        self._qc = QuantumCircuit(num_qubits)
        self._qc.append(self._feature_map, range(num_qubits))
        self._qc.append(self._var_form, range(num_qubits))

        # set observable
        self.observable = observable if observable else PauliSumOp.from_list([('Z'*num_qubits, 1)])

        # combine all to operator
        operator = ~StateFn(self.observable) @ StateFn(self._qc)

        super().__init__(operator, self.feature_map.parameters, self.var_form.parameters,
                         exp_val=exp_val, quantum_instance=quantum_instance)
