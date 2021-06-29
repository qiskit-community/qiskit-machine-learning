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
from qiskit.algorithms.optimizers import Optimizer, SLSQP

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import (
    Loss,
    L1Loss,
    L2Loss,
    CrossEntropyLoss,
    CrossEntropySigmoidLoss,
)
from qiskit_machine_learning.deprecation import deprecate_values


class TrainableModel:
    """Base class for ML model. This class defines Scikit-Learn like interface to implement."""

    @deprecate_values(
        "0.2.0", {"loss": {"l1": "absolute_error", "l2": "squared_error"}}, stack_level=4
    )
    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: Union[str, Loss] = "squared_error",
        optimizer: Optional[Optimizer] = None,
        warm_start: bool = False,
        initial_point: np.ndarray = None,
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
            loss: A target loss function to be used in training. Default is `squared_error`,
                i.e. L2 loss. Can be given either as a string for 'absolute_error' (i.e. L1 Loss),
                'squared_error', 'cross_entropy', 'cross_entropy_sigmoid', or as a loss function
                implementing the Loss interface.
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.

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
            if loss == "absolute_error":
                self._loss = L1Loss()
            elif loss == "squared_error":
                self._loss = L2Loss()
            elif loss == "cross_entropy":
                self._loss = CrossEntropyLoss()
            elif loss == "cross_entropy_sigmoid":
                self._loss = CrossEntropySigmoidLoss()
            elif loss == "l1":
                self._loss = L1Loss()
            elif loss == "l2":
                self._loss = L2Loss()
            else:
                raise QiskitMachineLearningError(f"Unknown loss {loss}!")

        if optimizer is None:
            optimizer = SLSQP()
        self._optimizer = optimizer
        self._warm_start = warm_start
        self._fit_result = None
        self._initial_point = initial_point

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

    @property
    def initial_point(self) -> np.ndarray:
        """Returns current initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray) -> None:
        """Sets the initial point"""
        self._initial_point = initial_point

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

    def _choose_initial_point(self) -> np.ndarray:
        """Choose an initial point for the optimizer. If warm start is set and the model is
        already trained then use a fit result as an initial point. If initial point is passed,
        then use this value, otherwise pick a random location.

        Returns:
            An array as an initial point
        """
        if self._warm_start and self._fit_result is not None:
            self._initial_point = self._fit_result[0]
        elif self._initial_point is None:
            self._initial_point = np.random.rand(self._neural_network.num_weights)
        return self._initial_point
