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
"""An abstract objective function definition and common objective functions suitable
for classifiers/regressors."""

from abc import abstractmethod
from typing import Optional, Union

import numpy as np

try:
    from sparse import SparseArray
except ImportError:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import Loss


class ObjectiveFunction:
    """An abstract objective function. Provides methods for computing objective value and
    gradients for forward and backward passes."""

    # pylint: disable=invalid-name
    def __init__(
        self, X: np.ndarray, y: np.ndarray, neural_network: NeuralNetwork, loss: Loss
    ) -> None:
        """
        Args:
            X: The input data.
            y: The target values.
            neural_network: An instance of an quantum neural network to be used by this
                objective function.
            loss: A target loss function to be used in training.
        """
        super().__init__()
        self._X = X
        self._y = y
        self._neural_network = neural_network
        self._loss = loss
        self._last_forward_id: Optional[str] = None
        self._last_forward: Optional[Union[np.ndarray, SparseArray]] = None

    @abstractmethod
    def objective(self, weights: np.ndarray) -> float:
        """Computes the value of this objective function given weights.

        Args:
            weights: an array of weights to be used in the objective function.

        Returns:
            Value of the function.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Computes gradients of this objective function given weights.

        Args:
            weights: an array of weights to be used in the objective function.

        Returns:
            Gradients of the function.
        """
        raise NotImplementedError

    def _neural_network_forward(self, weights: np.ndarray) -> Union[np.ndarray, SparseArray]:
        """
        Computes and caches the results of the forward pass. Cached values may be re-used in
        gradient computation.

        Args:
            weights: an array of weights to be used in the forward pass.

        Returns:
            The result of the neural network.
        """
        # we compare weights and input data by identifier to make it faster.
        # input data is added to the comparison for safety reasons only,
        # comparison of weights should be enough.
        data_id = str(id(weights)) + str(id(self._X))
        if data_id != self._last_forward_id:
            # compute forward and cache the results for re-use in backward
            self._last_forward = self._neural_network.forward(self._X, weights)
            self._last_forward_id = data_id
        return self._last_forward


class BinaryObjectiveFunction(ObjectiveFunction):
    """An objective function for binary representation of the output,
    e.g. classes of ``-1`` and ``+1``."""

    def objective(self, weights: np.ndarray) -> float:
        predict = self._neural_network_forward(weights)
        target = np.array(self._y).reshape(predict.shape)
        value = np.sum(self._loss(predict, target))
        return value

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        output = self._neural_network_forward(weights)
        _, weights_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))
        for i in range(len(self._X)):
            grad += self._loss.gradient(output[i][0], self._y[i]) * weights_grad[i]

        return grad


class MultiClassObjectiveFunction(ObjectiveFunction):
    """
    An objective function for multiclass representation of the output,
    e.g. classes of ``0``, ``1``, ``2``, etc.
    """

    def objective(self, weights: np.ndarray) -> float:
        val = 0.0
        probs = self._neural_network_forward(weights)
        for i in range(len(self._X)):
            for y_predict, prob in enumerate(probs[i]):
                val += prob * self._loss(y_predict, self._y[i])
        return val

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        num_classes = self._neural_network.output_shape[0]
        grad = np.zeros((1, self._neural_network.num_weights))
        # TODO: do batch eval instead of zip()
        for x, y_target in zip(self._X, self._y):
            _, weight_prob_grad = self._neural_network.backward(x, weights)
            for i in range(num_classes):
                grad += weight_prob_grad[0, i, :].reshape(grad.shape) * self._loss(i, y_target)
        return grad


class OneHotObjectiveFunction(ObjectiveFunction):
    """
    An objective function for one hot encoding representation of the output,
    e.g. classes like ``[1, 0, 0]``, ``[0, 1, 0]``, ``[0, 0, 1]``.
    """

    def objective(self, weights: np.ndarray) -> float:
        val = 0.0
        probs = self._neural_network_forward(weights)
        # TODO: do batch eval
        for i in range(len(self._X)):
            val += self._loss(probs[i], self._y[i])
        return val

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        grad = np.zeros(self._neural_network.num_weights)
        y_predict = self._neural_network_forward(weights)
        # TODO: do batch eval for backward instead of zip()
        for i, (x, y_target) in enumerate(zip(self._X, self._y)):
            _, weight_prob_grad = self._neural_network.backward(x, weights)
            grad += self._loss.gradient(y_predict[i], y_target) @ weight_prob_grad[0, :]
        return grad
