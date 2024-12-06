# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network regressor."""
from __future__ import annotations

from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler.passmanager import BasePassManager

from .neural_network_regressor import NeuralNetworkRegressor
from ...neural_networks import EstimatorQNN
from ...optimizers import Optimizer, Minimizer
from ...utils import derive_num_qubits_feature_map_ansatz
from ...utils.loss_functions import Loss


class VQR(NeuralNetworkRegressor):
    """A convenient Variational Quantum Regressor implementation."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | None = None,
        ansatz: QuantumCircuit | None = None,
        observable: BaseOperator | None = None,
        loss: str | Loss = "squared_error",
        optimizer: Optimizer | Minimizer | None = None,
        warm_start: bool = False,
        initial_point: np.ndarray | None = None,
        callback: Callable[[np.ndarray, float], None] | None = None,
        *,
        estimator: BaseEstimator | None = None,
        pass_manager: BasePassManager | None = None,
    ) -> None:
        r"""
        Args:
            num_qubits: The number of qubits for the underlying QNN.
                If ``None`` then the number of qubits is derived from the
                feature map or ansatz, but if neither of these are given an error is raised.
                The number of qubits in the feature map and ansatz are adjusted to this
                number if required.
            feature_map: The (parametrized) circuit to be used as a feature map for the underlying
                QNN. If ``None`` the :class:`~qiskit.circuit.library.ZZFeatureMap`
                is used if the number of qubits is larger than 1. For a single qubit regression
                problem the :class:`~qiskit.circuit.library.ZFeatureMap` is used by default.
            ansatz: The (parametrized) circuit to be used as an ansatz for the underlying
                QNN. If ``None`` then the :class:`~qiskit.circuit.library.RealAmplitudes`
                circuit is used.
            observable: The observable to be measured in the underlying QNN. If ``None``,
                use the default :math:`Z^{\otimes num\_qubits}` observable.
            loss: A target loss function to be used in training. Default is squared error.
            optimizer: An instance of an optimizer or a callable to be used in training.
                Refer to :class:`~qiskit_machine_learning.optimizers.Minimizer` for more information on
                the callable protocol. When `None` defaults to
                :class:`~qiskit_machine_learning.optimizers.SLSQP`.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.
            callback: A reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
            estimator: an optional Estimator primitive instance to be used by the underlying
                :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN` neural network. If
                ``None`` is passed then an instance of the reference Estimator will be used.
            pass_manager: The pass manager to transpile the circuits, if necessary.
                Defaults to ``None``, as some primitives do not need transpiled circuits.
        Raises:
            QiskitMachineLearningError: Needs at least one out of ``num_qubits``, ``feature_map`` or
                ``ansatz`` to be given. Or the number of qubits in the feature map and/or ansatz
                can't be adjusted to ``num_qubits``.
            ValueError: if the type of the observable is not compatible with ``estimator``.
        """
        if observable is not None and not isinstance(observable, BaseOperator):
            raise ValueError(
                f"Unsupported type of the observable, expected "
                f"'BaseOperator', got {type(observable)}"
            )

        self._estimator = estimator

        num_qubits, feature_map, ansatz = derive_num_qubits_feature_map_ansatz(
            num_qubits, feature_map, ansatz
        )

        # construct circuit
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._num_qubits = num_qubits
        circuit = QuantumCircuit(self._num_qubits)
        circuit.compose(self._feature_map, inplace=True)
        circuit.compose(self._ansatz, inplace=True)

        observables = [observable] if observable is not None else None

        if pass_manager:
            circuit.measure_all()
            circuit = pass_manager.run(circuit)
            observables = (
                [observable.apply_layout(circuit.layout)] if observable is not None else None
            )

        neural_network = EstimatorQNN(
            estimator=estimator,
            circuit=circuit,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            pass_manager=pass_manager,
        )

        super().__init__(
            neural_network=neural_network,
            loss=loss,
            optimizer=optimizer,
            warm_start=warm_start,
            initial_point=initial_point,
            callback=callback,
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
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return self._num_qubits
