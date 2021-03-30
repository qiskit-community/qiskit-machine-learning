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

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase
from qiskit.utils import QuantumInstance

from .neural_network_regressor import NeuralNetworkRegressor
from ...neural_networks import TwoLayerQNN
from ...utils.loss_functions.loss import Loss


class VQR(NeuralNetworkRegressor):
    """Quantum neural network regressor using TwoLayerQNN"""

    def __init__(self,
                 num_qubits: int = None,
                 feature_map: QuantumCircuit = None,
                 var_form: QuantumCircuit = None,
                 observable: Union[QuantumCircuit, OperatorBase] = None,
                 loss: Union[str, Loss] = 'l2',
                 optimizer: Optimizer = None,
                 warm_start: bool = False,
                 quantum_instance: QuantumInstance = None) -> None:
        r"""
        Args:
            num_qubits: The number of qubits to be used. If None, and neither feature_map nor
                var_form are given, it is initially set to 2, i.e., the default of the TwoLayerQNN.
            feature_map: The feature map to be used to construct a TwoLayerQNN. If None, use the
                ZZFeatureMap, i.e., the default of the TwoLayerQNN.
            var_form: The variational to be used to construct a TwoLayerQNN. If None, use the
                RealAmplitudes, i.e., the default of the TwoLayerQNN.
            observable: The observable to be measured in the underlying TwoLayerQNN. If  None, use
                the default from the TwoLayerQNN, i.e., `Z^{\otimes num_qubits}`.
            loss: A target loss function to be used in training. Default is L2.
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: Neither num_qubits, nor feature_map, nor var_form given.
        """

        # construct QNN
        neural_network = TwoLayerQNN(num_qubits=num_qubits,
                                     feature_map=feature_map,
                                     var_form=var_form,
                                     observable=observable,
                                     quantum_instance=quantum_instance)

        super().__init__(neural_network=neural_network,
                         loss=loss,
                         optimizer=optimizer,
                         warm_start=warm_start)

    @property
    def feature_map(self) -> QuantumCircuit:
        """ Returns the used feature map."""
        return cast(self.neural_network, TwoLayerQNN).feature_map

    @property
    def var_form(self) -> QuantumCircuit:
        """ Returns the used variational form."""
        return cast(self.neural_network, TwoLayerQNN).var_form

    @property
    def num_qubits(self) -> int:
        """ Returns the number of qubits used by variational form and feature map."""
        return cast(self.neural_network, TwoLayerQNN).num_qubits
