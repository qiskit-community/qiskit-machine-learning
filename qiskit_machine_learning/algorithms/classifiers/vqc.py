# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network classifier."""

from typing import Union, cast
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import Optimizer

from qiskit_machine_learning.utils.num_qubits_helper import retrieve_arguments_if_none
from ...neural_networks import CircuitQNN
from ...utils.loss_functions import Loss

from .neural_network_classifier import NeuralNetworkClassifier


class VQC(NeuralNetworkClassifier):
    """Quantum neural network classifier."""

    def __init__(self,
                 num_qubits: int = None,
                 feature_map: QuantumCircuit = None,
                 ansatz: QuantumCircuit = None,
                 loss: Union[str, Loss] = 'cross_entropy',
                 optimizer: Optimizer = None,
                 warm_start: bool = False,
                 quantum_instance: QuantumInstance = None) -> None:
        """
        Args:
            num_qubits: The number of qubits for the underlying CircuitQNN. If None, derive from
                feature_map or ansatz. If neither of those is given, raise exception.
            feature_map: The feature map for underlying CircuitQNN. If None, use ZZFeatureMap.
            ansatz: The ansatz for the underlying CircuitQNN. If None, use RealAmplitudes.
            loss: A target loss function to be used in training. Default is cross entropy.
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: Needs at least one out of num_qubits, feature_map or
                ansatz to be given.
        """

        # check num_qubits, feature_map, and ansatz
        ansatz_, feature_map_, num_qubits_ = retrieve_arguments_if_none(ansatz, feature_map,
                                                                        num_qubits)

        # construct circuit
        self._feature_map = feature_map_
        self._ansatz = ansatz_
        self._num_qubits = num_qubits_
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.compose(feature_map, inplace=True)
        self._circuit.compose(ansatz, inplace=True)

        # construct circuit QNN
        neural_network = CircuitQNN(self._circuit,
                                    feature_map.parameters,
                                    ansatz.parameters,
                                    interpret=self._get_interpret(2),
                                    output_shape=2,
                                    quantum_instance=quantum_instance)

        super().__init__(neural_network=neural_network,
                         loss=loss,
                         one_hot=True,
                         optimizer=optimizer,
                         warm_start=warm_start)

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
        cast(CircuitQNN, self._neural_network).set_interpret(self._get_interpret(num_classes),
                                                             num_classes)
        return super().fit(X, y)

    def _get_interpret(self, num_classes):
        def parity(x, num_classes=num_classes):
            return '{:b}'.format(x).count('1') % num_classes

        return parity
