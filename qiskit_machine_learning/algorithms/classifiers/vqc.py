# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of variational quantum classifier."""

from __future__ import annotations
from typing import Callable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult
from qiskit.primitives import BaseSampler

from ...deprecation import warn_deprecated, DeprecatedType
from ...neural_networks import CircuitQNN, SamplerQNN
from ...utils import derive_num_qubits_feature_map_ansatz
from ...utils.loss_functions import Loss

from .neural_network_classifier import NeuralNetworkClassifier


class VQC(NeuralNetworkClassifier):
    r"""A convenient Variational Quantum Classifier implementation.

    The variational quantum classifier (VQC) is a variational algorithm where the measured
    bitstrings are interpreted as the output of a classifier.

    Constructs a quantum circuit and corresponding neural network, then uses it to instantiate a
    neural network classifier.

    Labels can be passed in various formats, they can be plain labels, a one dimensional numpy
    array that contains integer labels like `[0, 1, 2, ...]`, or a numpy array with categorical
    string labels. One hot encoded labels are also supported. Internally, labels are transformed
    to one hot encoding and the classifier is always trained on one hot labels.

    Multi-label classification is not supported. E.g., :math:`[[1, 1, 0], [0, 1, 1], [1, 0, 1]]`.
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | None = None,
        ansatz: QuantumCircuit | None = None,
        loss: str | Loss = "cross_entropy",
        optimizer: Optimizer | None = None,
        warm_start: bool = False,
        quantum_instance: QuantumInstance | Backend | None = None,
        initial_point: np.ndarray | None = None,
        callback: Callable[[np.ndarray, float], None] | None = None,
        *,
        sampler: BaseSampler | None = None,
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits for the underlying QNN.
                If ``None`` is given, the number of qubits is derived from the
                feature map or ansatz. If neither of those is given, raises an exception.
                The number of qubits in the feature map and ansatz are adjusted to this
                number if required.
            feature_map: The (parametrized) circuit to be used as a feature map for the underlying
                QNN. If ``None`` is given, the ``ZZFeatureMap`` is used if the number of qubits
                is larger than 1. For a single qubit classification problem the ``ZFeatureMap``
                is used by default.
            ansatz: The (parametrized) circuit to be used as an ansatz for the underlying
                QNN. If ``None`` is given then the ``RealAmplitudes`` circuit is used.
            loss: A target loss function to be used in training. Default value is ``cross_entropy``.
            optimizer: An instance of an optimizer to be used in training. When ``None`` defaults
                to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            quantum_instance: Deprecated: If a quantum instance is sent and ``sampler`` is ``None``,
                the underlying QNN will be of type
                :class:`~qiskit_machine_learning.neural_networks.CircuitQNN`, and the quantum
                instance will be used to compute the neural network's results. If a sampler
                instance is also set, it will override the `quantum_instance` parameter and
                a :class:`~qiskit_machine_learning.neural_networks.SamplerQNN`
                will be used instead.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
            sampler: If a sampler instance is sent, the underlying QNN will be of type
                :class:`~qiskit_machine_learning.neural_networks.SamplerQNN`, and the sampler
                primitive will be used to compute the neural network's results.
        Raises:
            QiskitMachineLearningError: Needs at least one out of ``num_qubits``, ``feature_map`` or
                ``ansatz`` to be given. Or the number of qubits in the feature map and/or ansatz
                can't be adjusted to ``num_qubits``.
        """

        num_qubits, feature_map, ansatz = derive_num_qubits_feature_map_ansatz(
            num_qubits, feature_map, ansatz
        )

        # construct circuit
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._num_qubits = num_qubits
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.compose(self.feature_map, inplace=True)
        self._circuit.compose(self.ansatz, inplace=True)

        # needed for mypy
        neural_network: SamplerQNN | CircuitQNN = None
        if quantum_instance is not None and sampler is None:
            warn_deprecated(
                "0.5.0", DeprecatedType.ARGUMENT, old_name="quantum_instance", new_name="sampler"
            )
            neural_network = CircuitQNN(
                self._circuit,
                input_params=self.feature_map.parameters,
                weight_params=self.ansatz.parameters,
                interpret=self._get_interpret(2),
                output_shape=2,
                quantum_instance=quantum_instance,
                input_gradients=False,
            )
        else:
            # construct sampler QNN by default
            neural_network = SamplerQNN(
                sampler=sampler,
                circuit=self._circuit,
                input_params=self.feature_map.parameters,
                weight_params=self.ansatz.parameters,
                interpret=self._get_interpret(2),
                output_shape=2,
                input_gradients=False,
            )

        super().__init__(
            neural_network=neural_network,
            loss=loss,
            one_hot=True,
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
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return self.circuit.num_qubits

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> OptimizerResult:
        """
        Fit the model to data matrix X and targets y.

        Args:
            X: The input feature values.
            y: The input target values. Required to be one-hot encoded.

        Returns:
            Trained classifier.
        """
        X, y = self._validate_input(X, y)
        num_classes = self._num_classes

        # instance check required by mypy (alternative to cast)
        if isinstance(self._neural_network, (CircuitQNN, SamplerQNN)):
            self._neural_network.set_interpret(self._get_interpret(num_classes), num_classes)

        return super()._minimize(X, y)

    def _get_interpret(self, num_classes: int):
        def parity(x: int, num_classes: int = num_classes) -> int:
            return x % num_classes

        return parity
