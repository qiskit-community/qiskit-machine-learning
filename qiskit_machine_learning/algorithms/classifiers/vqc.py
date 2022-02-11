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
"""An implementation of quantum neural network classifier."""

from typing import Union, Optional, Callable, cast
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import Optimizer

from ...exceptions import QiskitMachineLearningError
from ...neural_networks import CircuitQNN
from ...utils.loss_functions import Loss

from .neural_network_classifier import NeuralNetworkClassifier


class VQC(NeuralNetworkClassifier):
    """Quantum neural network classifier."""

    def __init__(
        self,
        num_qubits: int = None,
        feature_map: QuantumCircuit = None,
        ansatz: QuantumCircuit = None,
        loss: Union[str, Loss] = "cross_entropy",
        optimizer: Optional[Optimizer] = None,
        warm_start: bool = False,
        quantum_instance: QuantumInstance = None,
        initial_point: np.ndarray = None,
        callback: Optional[Callable[[np.ndarray, float], None]] = None,
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits for the underlying CircuitQNN. If None, derive from
                feature_map or ansatz. If neither of those is given, raise exception.
            feature_map: The feature map for underlying CircuitQNN. If None, use ZZFeatureMap.
            ansatz: The ansatz for the underlying CircuitQNN. If None, use RealAmplitudes.
            loss: A target loss function to be used in training. Default is cross entropy.
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            quantum_instance: The quantum instance to execute circuits on.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
        Raises:
            QiskitMachineLearningError: Needs at least one out of num_qubits, feature_map or
                ansatz to be given.
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

        # construct circuit
        self._feature_map = feature_map_
        self._ansatz = ansatz_
        self._num_qubits = num_qubits_
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.compose(self.feature_map, inplace=True)
        self._circuit.compose(self.ansatz, inplace=True)

        # construct circuit QNN
        neural_network = CircuitQNN(
            self._circuit,
            self.feature_map.parameters,
            self.ansatz.parameters,
            interpret=self._get_interpret(2),
            output_shape=2,
            quantum_instance=quantum_instance,
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

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        """
        Fit the model to data matrix X and targets y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained classifier.
        """
        num_classes = len(np.unique(y, axis=0))
        cast(CircuitQNN, self._neural_network).set_interpret(
            self._get_interpret(num_classes), num_classes
        )
        return super().fit(X, y)

    def _get_interpret(self, num_classes: int):
        def parity(x: int, num_classes: int = num_classes) -> int:
            return x % num_classes

        return parity
