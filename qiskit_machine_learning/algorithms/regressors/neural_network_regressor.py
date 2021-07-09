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

from typing import Union, Optional, Callable

import numpy as np
from qiskit.algorithms.optimizers import Optimizer
from sklearn.base import RegressorMixin

from ..objective_functions import (
    BinaryObjectiveFunction,
    MultiClassObjectiveFunction,
    ObjectiveFunction,
)
from ..trainable_model import TrainableModel
from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions import Loss


class NeuralNetworkRegressor(TrainableModel, RegressorMixin):
    """Quantum neural network regressor. Implements Scikit-Learn compatible methods for
    regression and extends ``RegressorMixin``. See `Scikit-Learn <https://scikit-learn.org>`__
    for more details.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: Union[str, Loss] = "squared_error",
        optimizer: Optional[Optimizer] = None,
        warm_start: bool = False,
        initial_point: np.ndarray = None,
        callback: Optional[Callable[[np.ndarray, float], None]] = None,
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
            callback: a callback that can access the intermediate data during the optimization.
                Two parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the weights for the objective function, and the computed objective value.
        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        super().__init__(neural_network, loss, optimizer, warm_start, initial_point)
        self._callback = callback

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        # mypy definition
        function: ObjectiveFunction = None
        if self._neural_network.output_shape == (1,):
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)

        objective = self.get_objective(function, self._callback)

        self._fit_result = self._optimizer.optimize(
            self._neural_network.num_weights,
            objective,
            function.gradient,
            initial_point=self._choose_initial_point(),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        if self._fit_result is None:
            raise QiskitMachineLearningError("Model needs to be fit to some training data first!")

        return self._neural_network.forward(X, self._fit_result[0])

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        return RegressorMixin.score(self, X, y, sample_weight)
