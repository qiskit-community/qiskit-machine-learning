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
from ..exceptions import QiskitMachineLearningError


class TwoLayerQNN(OpflowQNN):
    """Two Layer Quantum Neural Network consisting of a feature map, a variational form,
    and an observable.
    """

    def __init__(self, num_qubits: int = None,
                 feature_map: QuantumCircuit = None,
                 var_form: QuantumCircuit = None,
                 observable: Optional[OperatorBase] = None,
                 exp_val: Optional[ExpectationBase] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):
        r"""Initializes the Two Layer Quantum Neural Network.

        Args:
            num_qubits: The number of qubits to represent the network, if None and neither the
                feature_map not the var_form are given, raise exception.
            feature_map: The (parametrized) circuit to be used as feature map. If None is given,
                the `ZZFeatureMap` is used.
            var_form: The (parametrized) circuit to be used as variational form. If None is given,
                the `RealAmplitudes` circuit is used.
            observable: observable to be measured to determine the output of the network. If None
                is given, the `Z^{\otimes num_qubits}` observable is used.
<<<<<<< HEAD
            exp_val: The Expected Value converter to be used for the operator.
            quantum_instance: The quantum instance to evaluate the networks.
        """

        # set number of qubits, feature map, and variational form
        self.num_qubits = num_qubits
        self._feature_map = feature_map if feature_map else ZZFeatureMap(num_qubits)
        self._var_form = var_form if var_form else RealAmplitudes(num_qubits)
=======

        Raises:
            QiskitMachineLearningError: In case of inconsistent num_qubits, feature_map, var_form.
        """

        # check num_qubits, feature_map, and var_form
        if num_qubits is None and feature_map is None and var_form is None:
            raise QiskitMachineLearningError(
                'Need at least one of num_qubits, feature_map, or var_form!')
        num_qubits_: int = None
        feature_map_: QuantumCircuit = None
        var_form_: QuantumCircuit = None
        if num_qubits:
            num_qubits_ = num_qubits
            if feature_map:
                if feature_map.num_qubits != num_qubits:
                    raise QiskitMachineLearningError('Incompatible num_qubits and feature_map!')
                feature_map_ = feature_map
            else:
                feature_map_ = ZZFeatureMap(num_qubits)
            if var_form:
                if var_form.num_qubits != num_qubits:
                    raise QiskitMachineLearningError('Incompatible num_qubits and var_form!')
                var_form_ = var_form
            else:
                var_form_ = RealAmplitudes(num_qubits)
        else:
            if feature_map and var_form:
                if feature_map.num_qubits != var_form.num_qubits:
                    raise QiskitMachineLearningError('Incompatible feature_map and var_form!')
                feature_map_ = feature_map
                var_form_ = var_form
                num_qubits_ = feature_map.num_qubits
            elif feature_map:
                num_qubits_ = feature_map.num_qubits
                feature_map_ = feature_map
                var_form_ = RealAmplitudes(num_qubits_)
            elif var_form:
                num_qubits_ = var_form.num_qubits
                var_form_ = var_form
                feature_map_ = ZZFeatureMap(num_qubits_)

        self._feature_map = feature_map_
        input_params = list(self._feature_map.parameters)

        self._var_form = var_form_
        weight_params = list(self._var_form.parameters)
>>>>>>> pr/13

        # construct circuit
        self._circuit = QuantumCircuit(num_qubits_)
        self._circuit.append(self._feature_map, range(num_qubits_))
        self._circuit.append(self._var_form, range(num_qubits_))

<<<<<<< HEAD
        # set observable
        self.observable = observable if observable else PauliSumOp.from_list([('Z'*num_qubits, 1)])
=======
        # construct observable
        self.observable = observable if observable else PauliSumOp.from_list([('Z'*num_qubits_, 1)])
>>>>>>> pr/13

        # combine all to operator
        operator = ~StateFn(self.observable) @ StateFn(self._circuit)

<<<<<<< HEAD
        super().__init__(operator, self._feature_map.parameters, self._var_form.parameters,
                         exp_val=exp_val, quantum_instance=quantum_instance)
=======
        super().__init__(operator, input_params, weight_params, quantum_instance=quantum_instance)

    @property
    def feature_map(self) -> QuantumCircuit:
        """ Returns the used feature map."""
        return self._feature_map

    @property
    def var_form(self) -> QuantumCircuit:
        """ Returns the used variational form."""
        return self._var_form

    @property
    def circuit(self) -> QuantumCircuit:
        """ Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """ Returns the number of qubits used by variational form and feature map."""
        return self._circuit.num_qubits
>>>>>>> pr/13
