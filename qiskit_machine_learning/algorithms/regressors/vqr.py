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

from typing import Union

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase
from qiskit.utils import QuantumInstance

from .neural_network_regressor import NeuralNetworkRegressor
from ...exceptions import QiskitMachineLearningError
from ...neural_networks import TwoLayerQNN
from ...utils.loss_functions.loss import Loss


class VQR(NeuralNetworkRegressor):
    """Quantum neural network regressor using TwoLayerQNN"""

    def __init__(self,
                 feature_map: QuantumCircuit,
                 var_form: QuantumCircuit,
                 observable: Union[QuantumCircuit, OperatorBase] = None,
                 loss: Union[str, Loss] = 'l2',
                 optimizer: Optimizer = None,
                 warm_start: bool = False,
                 quantum_instance: QuantumInstance = None) -> None:
        r"""
        Args:
            feature_map: The QuantumCircuit instance to use.
            var_form: The variational form instance.
            observable: observable to be measured to determine the output of the network. If None
                is given, the `Z^{\otimes num_qubits}` observable is used.
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
        self._observable = observable

        # construct circuit QNN
        neural_network = TwoLayerQNN(num_qubits=self._num_qubits,
                                     feature_map=self.feature_map,
                                     var_form=self.var_form,
                                     observable=self._observable,
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
    def num_qubits(self) -> int:
        """ Returns the number of qubits used by variational form and feature map."""
        return self._num_qubits
