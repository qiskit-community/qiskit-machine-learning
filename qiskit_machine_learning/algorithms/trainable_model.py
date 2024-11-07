# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A base ML model with a Scikit-Learn like interface."""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable
import numpy as np

from qiskit_machine_learning import QiskitMachineLearningError

from .objective_functions import ObjectiveFunction
from .serializable_model import SerializableModelMixin
from ..optimizers import Optimizer, SLSQP, OptimizerResult, Minimizer
from ..utils import algorithm_globals
from ..neural_networks import NeuralNetwork
from ..utils.loss_functions import (
    Loss,
    L1Loss,
    L2Loss,
    CrossEntropyLoss,
)


class TrainableModel(SerializableModelMixin):
    """Base class for ML model that defines a scikit-learn like interface for Estimators."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: str | Loss = "squared_error",
        optimizer: Optimizer | Minimizer | None = None,
        warm_start: bool = False,
        initial_point: np.ndarray = None,
        callback: Callable[[np.ndarray, float], None] | None = None,
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
                'squared_error', 'cross_entropy', or as a loss function
                implementing the Loss interface.
            optimizer: An instance of an optimizer or a callable to be used in training.
                Refer to :class:`~qiskit_machine_learning.optimizers.Minimizer` for more information on
                the callable protocol. When `None` defaults to
                :class:`~qiskit_machine_learning.optimizers.SLSQP`.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.
            callback: A reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
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
            else:
                raise QiskitMachineLearningError(f"Unknown loss {loss}!")

        # call the setter that has some additional checks
        self.optimizer = optimizer

        self._warm_start = warm_start
        self._fit_result: OptimizerResult | None = None
        self._initial_point = initial_point
        self._callback = callback

    @property
    def neural_network(self):
        """Returns the underlying neural network."""
        return self._neural_network

    @property
    def loss(self):
        """Returns the underlying neural network."""
        return self._loss

    @property
    def optimizer(self) -> Optimizer | Minimizer:
        """Returns an optimizer to be used in training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer | Minimizer | None = None):
        """Sets the optimizer to use in training process."""
        if optimizer is None:
            optimizer = SLSQP()
        self._optimizer = optimizer

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

    @property
    def weights(self) -> np.ndarray:
        """Returns trained weights as a numpy array. The weights can be also queried by calling
        `model.fit_result.x`, but in this case their representation depends on the optimizer used.

        Raises:
            QiskitMachineLearningError: If the model has not been fit.
        """
        self._check_fitted()
        return np.asarray(self._fit_result.x)

    @property
    def fit_result(self) -> OptimizerResult:
        """Returns a resulting object from the optimization procedure. Please refer to the
        documentation of the `OptimizerResult
        <https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.OptimizerResult.html>`_
        class for more details.

        Raises:
            QiskitMachineLearningError: If the model has not been fit.
        """
        self._check_fitted()
        return self._fit_result

    @property
    def callback(self) -> Callable[[np.ndarray, float], None] | None:
        """Return the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable[[np.ndarray, float], None] | None) -> None:
        """Set the callback."""
        self._callback = callback

    def _check_fitted(self) -> None:
        if self._fit_result is None:
            raise QiskitMachineLearningError("The model has not been fitted yet")

    # pylint: disable=invalid-name
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainableModel:
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
        if not self._warm_start:
            self._fit_result = None

        self._fit_result = self._fit_internal(X, y)
        return self

    @abstractmethod
    # pylint: disable=invalid-name
    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> OptimizerResult:
        raise NotImplementedError

    @abstractmethod
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
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
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
            self._initial_point = self._fit_result.x  # type: ignore[assignment]
        elif self._initial_point is None:
            self._initial_point = algorithm_globals.random.random(self._neural_network.num_weights)
        return self._initial_point

    def _get_objective(
        self,
        function: ObjectiveFunction,
    ) -> Callable:
        """
        Wraps the given `ObjectiveFunction` to add callback calls, if `callback` is not None, along
        with evaluating the objective value. Returned objective function is passed to
        `Optimizer.minimize()`.
        Args:
            function: The objective function whose objective is to be evaluated.

        Returns:
            Objective function to evaluate objective value and optionally invoke callback calls.
        """
        if self._callback is None:
            return function.objective

        def objective(objective_weights):
            objective_value = function.objective(objective_weights)
            self._callback(objective_weights, objective_value)
            return objective_value

        return objective

    def _minimize(self, function: ObjectiveFunction) -> OptimizerResult:
        """
        Minimizes the objective function.

        Args:
            function: a function to minimize.

        Returns:
            An optimization result.
        """
        objective = self._get_objective(function)

        initial_point = self._choose_initial_point()
        if callable(self._optimizer):
            optimizer_result = self._optimizer(  # type: ignore[call-arg]
                fun=objective, x0=initial_point, jac=function.gradient
            )
        else:
            optimizer_result = self._optimizer.minimize(
                fun=objective,
                x0=initial_point,
                jac=function.gradient,  # type: ignore[arg-type]
            )
        return optimizer_result
