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
"""A common model and objective functions for classifiers/regressors."""

from abc import abstractmethod
from typing import Optional

import numpy as np


class ObjectiveFunction:
    """An abstract objective function.
    Provides methods for computing objective value and gradients."""
    def __init__(self, X: np.ndarray, y: np.ndarray, neural_network, loss) -> None: # pylint: disable=invalid-name
        super().__init__()
        self._X = X
        self._y = y
        self._neural_network = neural_network
        self._loss = loss
        self._last_forward_id: Optional[str] = None
        self._last_forward: Optional[np.ndarray] = None

    @abstractmethod
    def objective(self, w) -> float:
        """

        Args:
            w:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, w) -> np.ndarray:
        """

        Args:
            w:

        Returns:

        """
        raise NotImplementedError

    def _forward(self, weights: np.ndarray):
        data_id = str(id(weights)) + str(id(self._X))
        if data_id != self._last_forward_id:
            self._last_forward = self._neural_network.forward(self._X, weights)
            self._last_forward_id = data_id
        return self._last_forward


class BinaryObjectiveFunction(ObjectiveFunction):
    def objective(self, w):
        predict = self._forward(w)
        target = np.array(self._y).reshape(predict.shape)
        value = np.sum(self._loss(predict, target))
        return value

    def gradient(self, w):
        output = self._forward(w)
        _, weights_grad = self._neural_network.backward(self._X, w)

        grad = np.zeros((1, self._neural_network.num_weights))
        for i in range(len(self._X)):
            grad += self._loss.gradient(output[i][0], self._y[i]) * weights_grad[i]

        return grad


class MultiClassObjectiveFunction(ObjectiveFunction):
    def objective(self, w):
        val = 0.0
        probs = self._forward(w)
        for i in range(len(self._X)):
            for y_predict, prob in enumerate(probs[i]):
                val += prob * self._loss(y_predict, self._y[i])
        return val

    def gradient(self, w):
        # todo: we don't have _forward here
        num_classes = self._neural_network.output_shape[0]
        grad = np.zeros((1, self._neural_network.num_weights))
        for x, y_target in zip(self._X, self._y):
            # TODO: do batch eval
            _, weight_prob_grad = self._neural_network.backward(x, w)
            for i in range(num_classes):
                grad += weight_prob_grad[0, i, :].reshape(grad.shape) * self._loss(i, y_target)
        return grad


class OneHotObjectiveFunction(ObjectiveFunction):
    def objective(self, w):
        val = 0.0
        probs = self._forward(w)
        # TODO: do batch eval
        for i in range(len(self._X)):
            val += self._loss(probs[i], self._y[i])
        return val

    def gradient(self, w):
        grad = np.zeros(self._neural_network.num_weights)
        for x, y_target in zip(self._X, self._y):
            # TODO: do batch eval
            y_predict = self._neural_network.forward(x, w)
            _, weight_prob_grad = self._neural_network.backward(x, w)
            grad += self._loss.gradient(y_predict[0], y_target) @ weight_prob_grad[0, :]
        return grad


class TrainableModel:
    pass
