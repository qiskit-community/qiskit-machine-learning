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

from typing import Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import Optimizer

from ...exceptions import QiskitMachineLearningError
from ...neural_networks import CircuitQNN
from ...utils.loss_functions.loss import Loss

from . import NeuralNetworkClassifier


class VQC(NeuralNetworkClassifier):
    """Quantum neural network classifier."""

    def __init__(self,
                 feature_map: QuantumCircuit,
                 var_form: QuantumCircuit,
                 loss: Union[str, Loss] = 'l1',  # TODO: different default?
                 optimizer: Optimizer = None,
                 warm_start: bool = False,
                 quantum_instance: QuantumInstance = None) -> None:
        """
        Args:
            feature_map: The QuantumCircuit instance to use.
            var_form: The variational form instance.
            loss: A target loss function to be used in training. Default is `l2`, L2 loss.
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        # TODO: add getters/setters

        # construct circuit
        if feature_map.num_qubits != var_form.num_qubits:
            raise QiskitMachineLearningError('Feature map and var form need same number of qubits!')
        self._feature_map = feature_map
        self._var_form = var_form
        self._num_qubits = feature_map.num_qubits
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.append(feature_map, range(self._num_qubits))
        self._circuit.append(var_form, range(self._num_qubits))

        # construct circuit QNN
        neural_network = CircuitQNN(self._circuit,
                                    feature_map.parameters,
                                    var_form.parameters,
                                    interpret=self._get_interpret(2),
                                    output_shape=2,
                                    quantum_instance=quantum_instance)

        super().__init__(neural_network, loss, optimizer, warm_start)

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        """
        Fit the model to data matrix X and target(s) y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained classifier.
        """
        num_classes = len(np.unique(y))
        self._neural_network.set(self._get_interpret(num_classes), num_classes)
        return super().fit(X, y)

    def _get_interpret(self, num_classes):
        def parity(x, num_classes=num_classes):
            return '{:b}'.format(x).count('1') % num_classes
        return parity
