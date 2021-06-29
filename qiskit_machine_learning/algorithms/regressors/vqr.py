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

from typing import Union, Optional, cast

import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase
from qiskit.utils import QuantumInstance

from .neural_network_regressor import NeuralNetworkRegressor
from ...neural_networks import TwoLayerQNN
from ...utils.loss_functions import Loss


class VQR(NeuralNetworkRegressor):
    """Quantum neural network regressor using TwoLayerQNN"""

    def __init__(
        self,
        num_qubits: int = None,
        feature_map: QuantumCircuit = None,
        ansatz: QuantumCircuit = None,
        observable: Union[QuantumCircuit, OperatorBase] = None,
        loss: Union[str, Loss] = "squared_error",
        optimizer: Optional[Optimizer] = None,
        warm_start: bool = False,
        quantum_instance: QuantumInstance = None,
        initial_point: np.ndarray = None,
    ) -> None:
        r"""
        Args:
            num_qubits: The number of qubits to be used. If None, and neither feature_map nor
                ansatz are given, it is initially set to 2, i.e., the default of the TwoLayerQNN.
            feature_map: The feature map to be used to construct a TwoLayerQNN. If None, use the
                ZZFeatureMap, i.e., the default of the TwoLayerQNN.
            ansatz: The ansatz to be used to construct a TwoLayerQNN. If None, use the
                RealAmplitudes, i.e., the default of the TwoLayerQNN.
            observable: The observable to be measured in the underlying TwoLayerQNN. If  None, use
                the default from the TwoLayerQNN, i.e., `Z^{\otimes num_qubits}`.
            loss: A target loss function to be used in training. Default is squared error.
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.

        Raises:
            QiskitMachineLearningError: Neither num_qubits, nor feature_map, nor ansatz given.
        """

        # construct QNN
        neural_network = TwoLayerQNN(
            num_qubits=num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            observable=observable,
            quantum_instance=quantum_instance,
            input_gradients=False,
        )

        super().__init__(
            neural_network=neural_network,
            loss=loss,
            optimizer=optimizer,
            warm_start=warm_start,
            initial_point=initial_point,
        )

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the used feature map."""
        return cast(TwoLayerQNN, super().neural_network).feature_map

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the used ansatz."""
        return cast(TwoLayerQNN, super().neural_network).ansatz

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return cast(TwoLayerQNN, super().neural_network).num_qubits
