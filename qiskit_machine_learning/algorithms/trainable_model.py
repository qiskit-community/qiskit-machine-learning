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
"""A common model for classifiers/regressors."""
from abc import abstractmethod

import numpy as np


class ObjectiveFunction:
    def __init__(self, X: np.ndarray, y: np.ndarray, neural_network, loss) -> None: # pylint: disable=invalid-name
        super().__init__()
        self._X = X
        self._y = y
        self._neural_network = neural_network
        self._loss = loss

    @abstractmethod
    def objective(self, w):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, w):
        raise NotImplementedError


class SingleObjectiveFunction(ObjectiveFunction):

    def objective(self, w):
        predict = self._neural_network.forward(self._X, w)
        target = np.array(self._y).reshape(predict.shape)
        value = np.sum(self._loss(predict, target))
        return value

    def gradient(self, w):
        # TODO should store output from forward pass (implement loss interface?)
        # TODO: need to be able to turn off input grads if not needed.
        output = self._neural_network.forward(self._X, w)
        _, weights_grad = self._neural_network.backward(self._X, w)

        grad = np.zeros((1, self._neural_network.num_weights))
        for i in range(len(self._X)):
            grad += self._loss.gradient(output[i][0], self._y[i]) * weights_grad[i]

        return grad


class OneHotFunction(ObjectiveFunction):
    def objective(self, w):
        val = 0.0
        probs = self._neural_network.forward(self._X, w)
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


class ClassifierFunction(ObjectiveFunction):
    def objective(self, w):
        val = 0.0
        probs = self._neural_network.forward(self._X, w)
        for i in range(len(self._X)):
            for y_predict, prob in enumerate(probs[i]):
                val += prob * self._loss(y_predict, self._y[i])
        return val

    def gradient(self, w):
        num_classes = self._neural_network.output_shape[0]
        grad = np.zeros((1, self._neural_network.num_weights))
        for x, y_target in zip(self._X, self._y):
            # TODO: do batch eval
            _, weight_prob_grad = self._neural_network.backward(x, w)
            for i in range(num_classes):
                grad += weight_prob_grad[
                        0, i, :].reshape(grad.shape) * self._loss(i, y_target)
        return grad


class TrainableModel:
    pass
