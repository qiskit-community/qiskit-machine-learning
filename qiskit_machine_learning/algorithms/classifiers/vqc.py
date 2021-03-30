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
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import Optimizer

from ...exceptions import QiskitMachineLearningError
from ...neural_networks import CircuitQNN
from ...utils.loss_functions.loss import Loss

from .neural_network_classifier import NeuralNetworkClassifier


class VQC(NeuralNetworkClassifier):
    """Quantum neural network classifier."""

    def __init__(self,
                 num_qubits: int = None,
                 feature_map: QuantumCircuit = None,
                 var_form: QuantumCircuit = None,
                 loss: Union[str, Loss] = 'cross_entropy',
                 optimizer: Optimizer = None,
                 warm_start: bool = False,
                 quantum_instance: QuantumInstance = None) -> None:
        """
        Args:
            num_qubits: The number of qubits for the underlying CircuitQNN. If None, derive from
                feature_map or var_form. If neither of those is given, raise exception.
            feature_map: The feature map for underlying CircuitQNN. If None, use ZZFeatureMap.
            var_form: The variational for the underlying CircuitQNN. If None, use RealAmplitudes.
            loss: A target loss function to be used in training. Default is cross entropy.
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: Needs at least one out of num_qubits, feature_map or
                var_form to be given.
        """
<<<<<<< HEAD
        # TODO: add getters/setters
=======

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
>>>>>>> pr/13

        # construct circuit
        self._feature_map = feature_map_
        self._var_form = var_form_
        self._num_qubits = num_qubits_
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.compose(feature_map, inplace=True)
        self._circuit.compose(var_form, inplace=True)

        # construct circuit QNN
        neural_network = CircuitQNN(self._circuit,
                                    feature_map.parameters,
                                    var_form.parameters,
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
