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
"""A base ML model with a Scikit-Learn like interface."""

from abc import abstractmethod
from typing import Union, Optional

import numpy as np
from qiskit.algorithms.optimizers import Optimizer

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import (
    Loss,
    L1Loss,
    L2Loss,
    CrossEntropyLoss,
    CrossEntropySigmoidLoss,
)


class TrainableModel:
    """Base class for ML model. This class defines Scikit-Learn like interface to implement."""

    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: Union[str, Loss] = "l2",
        optimizer: Optimizer = None,
        warm_start: bool = False,
    ):
        """
        Args:
            neural_network: An instance of an quantum neural network. If the neural network has a
                one-dimensional output, i.e., `neural_network.output_shape=(1,)`, then it is
                expected to return values in [-1, +1] and it can only be used for binary
                classification. If the output is multi-dimensional, it is assumed that the result
                is a probability distribution, i.e., that the entries are non-negative and sum up
                to one. Then there are two options, either one-hot encoding or not. In case of
                one-hot encoding, each probability vector resulting a neural network is considered
                as one sample and the loss function is applied to the whole vector. Otherwise, each
                entry of the probability vector is considered as an individual sample and the loss
                function is applied to the index and weighted with the corresponding probability.
            loss: A target loss function to be used in training. Default is `l2`, i.e. L2 loss.
                Can be given either as a string for 'l1', 'l2', 'cross_entropy',
                'cross_entropy_sigmoid', or as a loss function implementing the Loss interface.
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        self._neural_network = neural_network
        if len(neural_network.output_shape) > 1:
            raise QiskitMachineLearningError("Invalid neural network output shape!")
        if isinstance(loss, Loss):
            self._loss = loss
        else:
            loss = loss.lower()
            if loss == "l1":
                self._loss = L1Loss()
            elif loss == "l2":
                self._loss = L2Loss()
            elif loss == "cross_entropy":
                self._loss = CrossEntropyLoss()
            elif loss == "cross_entropy_sigmoid":
                self._loss = CrossEntropySigmoidLoss()
            else:
                raise QiskitMachineLearningError(f"Unknown loss {loss}!")

        self._optimizer = optimizer
        self._warm_start = warm_start
        self._fit_result = None

    @property
    def neural_network(self):
        """Returns the underlying neural network."""
        return self._neural_network

    @property
    def loss(self):
        """Returns the underlying neural network."""
        return self._loss

    @property
    def optimizer(self) -> Optimizer:
        """Returns an optimizer to be used in training."""
        return self._optimizer

    @property
    def warm_start(self) -> bool:
        """Returns the warm start flag."""
        return self._warm_start

    @warm_start.setter
    def warm_start(self, warm_start: bool) -> None:
        """Sets the warm start flag."""
        self._warm_start = warm_start

    @abstractmethod
    # pylint: disable=invalid-name
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainableModel":
        """
        Fit the model to data matrix X and target(s) y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained model.

        Raises:
            QiskitMachineLearningError: In case of invalid data (e.g. incompatible with network)
        """
        raise NotImplementedError

    @abstractmethod
    # pylint: disable=invalid-name
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the network specified to the model.

        Args:
            X: The input data.
        Raises:
            QiskitMachineLearningError: Model needs to be fit to some training data first
        Returns:
            The predicted classes.
        """
        raise NotImplementedError

    @abstractmethod
    # pylint: disable=invalid-name
    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Returns a score of this model given samples and true values for the samples. In case of
        classification this should be mean accuracy, in case of regression the coefficient of
        determination :math:`R^2` of the prediction.

        Args:
            X: Test samples.
            y: True values for ``X``.
            sample_weight: Sample weights. Default is ``None``.

        Returns:
            a float score of the model.
        """
        raise NotImplementedError
