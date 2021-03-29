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
"""An implementation of quantum neural network regressor."""

from typing import Union, cast
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import Optimizer

from ...exceptions import QiskitMachineLearningError
from ...neural_networks import CircuitQNN
from ...utils.loss_functions.loss import Loss

from .neural_network_regressor import NeuralNetworkRegressor


class VQR(NeuralNetworkRegressor):
    """Quantum neural network regressor."""

    def __init__(self,
                 feature_map: QuantumCircuit,
                 var_form: QuantumCircuit,
                 loss: Union[str, Loss] = 'l2',
                 optimizer: Optimizer = None,
                 warm_start: bool = False,
                 quantum_instance: QuantumInstance = None) -> None:
        """
        Args:
            feature_map: The QuantumCircuit instance to use.
            var_form: The variational form instance.
            loss: A target loss function to be used in training. Default is L2.
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        # construct circuit
        if feature_map.num_qubits != var_form.num_qubits:
            raise QiskitMachineLearningError('Feature map and var form need same number of qubits!')
        self._feature_map = feature_map
        self._var_form = var_form
        self._num_qubits = feature_map.num_qubits
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.compose(feature_map, inplace=True)
        self._circuit.compose(var_form, inplace=True)

        # construct circuit QNN
        neural_network = CircuitQNN(self._circuit,
                                    feature_map.parameters,
                                    var_form.parameters,
                                    interpret=None,
                                    output_shape=2,
                                    quantum_instance=quantum_instance)

        super().__init__(neural_network=neural_network,
                         loss=loss,
                         optimizer=optimizer,
                         warm_start=warm_start)

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
        return self.circuit.num_qubits

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        """
        Fit the model to data matrix X and target y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained regressor.
        """
        return super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict using the network specified to the regression.

        Args:
            X: The input data.
        Raises:
            QiskitMachineLearningError: Model needs to be fit to some training data first
        Returns:
            The predicted values.
        """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> int:  # pylint: disable=invalid-name
        """
        Return R-squared on the given test data and targeted values.

        Args:
            X: Test samples.
            y: True target values given `X`.
        Raises:
            QiskitMachineLearningError: Model needs to be fit to some training data first
        Returns:
            R-squared value.
        """

        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        return super().score(X, y)

