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
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.opflow import ExpectationBase, OperatorBase, PauliSumOp, StateFn
from qiskit.primitives import BaseEstimator
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

import qiskit_machine_learning.optionals as _optionals

from ..utils import derive_num_qubits_feature_map_ansatz
from .estimator_qnn import EstimatorQNN
from .neural_network import NeuralNetwork
from .opflow_qnn import OpflowQNN

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:
    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass

class TwoLayerQNN(NeuralNetwork):
    """Two Layer Quantum Neural Network consisting of a feature map, a ansatz,
    and an observable.
    """
    #TODO: inherit from EstimatorQNN after deprecation of OpflowQNN

    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | None = None,
        ansatz: QuantumCircuit | None = None,
        observable: OperatorBase | QuantumCircuit | None = None,
        exp_val: ExpectationBase | None = None,
        quantum_instance: QuantumInstance | Backend | None = None,
        input_gradients: bool = False,
        estimator: BaseEstimator | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ):
        r"""
        Args:
            num_qubits: The number of qubits to represent the network. If ``None`` is given,
                the number of qubits is derived from the feature map or ansatz. If neither of those
                is given, raises an exception. The number of qubits in the feature map and ansatz
                are adjusted to this number if required.
            feature_map: The (parametrized) circuit to be used as a feature map. If ``None`` is given,
                the ``ZZFeatureMap`` is used if the number of qubits is larger than 1. For
                a single qubit two-layer QNN the ``ZFeatureMap`` circuit is used per default.
            ansatz: The (parametrized) circuit to be used as an ansatz. If ``None`` is given,
                the ``RealAmplitudes`` circuit is used.
            observable: observable to be measured to determine the output of the network. If
                ``None`` is given, the :math:`Z^{\otimes num\_qubits}` observable is used.
            exp_val: The Expected Value converter to be used for the operator obtained from the
                feature map and ansatz.
            quantum_instance: The quantum instance to evaluate the network.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using
                :class:`~qiskit_machine_learning.connectors.TorchConnector`.
        Raises:
            QiskitMachineLearningError: Needs at least one out of ``num_qubits``, ``feature_map`` or
                ``ansatz`` to be given. Or the number of qubits in the feature map and/or ansatz
                can't be adjusted to ``num_qubits``.
            ValueError: If both ``quantum_instance`` and ``estimator`` are given.
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
        self._observable = (
            observable if observable is not None else PauliSumOp.from_list([("Z" * num_qubits, 1)])
        )

        if quantum_instance is not None and estimator is not None:
            raise ValueError("Only one of quantum_instance or sampler can be passed, not both!")

        # # check positionally passing the sampler in the place of quantum_instance
        # # which will be removed in future
        # if isinstance(quantum_instance, BaseSampler):
        #     sampler = quantum_instance
        #     quantum_instance = None

        self._quantum_instance = None
        if quantum_instance is not None:
            warnings.warn(
                "The quantum_instance argument has been superseded by the sampler argument. "
                "This argument will be deprecated in a future release and subsequently "
                "removed after that.",
                category=PendingDeprecationWarning,
                stacklevel=2,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=PendingDeprecationWarning)
                self.quantum_instance = quantum_instance

        self._estimator = estimator
        if estimator is not None:
            # if estimator is passed, use ``EstimatorQNN``
            self._estimator_qnn = EstimatorQNN(
                estimator=estimator,
                circuit=self._circuit,
                observables=self._observable,
                input_params=input_params,
                weight_params=weight_params,
                gradient=gradient,
                input_gradients=input_gradients,
            )
            output_shape = self._estimator_qnn.output_shape
        else:
            # Otherwise, use ``OpflowQNN``
            # combine all to operator
            operator = StateFn(self._observable, is_measurement=True) @ StateFn(self._circuit)
            self._opflow_qnn = OpflowQNN(
                operator=operator,
                input_params=input_params,
                weight_params=weight_params,
                exp_val=exp_val,
                quantum_instance=quantum_instance,
                input_gradients=input_gradients,
            )
            output_shape = self._opflow_qnn.output_shape

        super().__init__(
            len(input_params),
            len(weight_params),
            sparse=False,
            output_shape=output_shape,
            input_gradients=input_gradients,
        )

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        if self._estimator is not None:
            return self._estimator_qnn.forward(input_data, weights)
        else:
            return self._opflow_qnn.forward(input_data, weights)

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:
        if self._estimator is not None:
            return self._estimator_qnn.backward(input_data, weights)
        else:
            return self._opflow_qnn.backward(input_data, weights)

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
