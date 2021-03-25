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

""" neural network regressor """

from typing import Union
import numpy as np

from qiskit.algorithms.optimizers import Optimizer
from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions.loss import Loss, L1Loss, L2Loss

class NeuralNetworkRegressor:
    """ Quantum neural network regressor"""

    def __init__(self, neural_network: NeuralNetwork, loss, optimizer, warm_start=False,
                 callback=None) -> None:
        """
        Args:
        """
        del callback  # silence pylint until it is handled

        self._neural_network = neural_network
        self._loss = loss
        self._optimizer = optimizer

        self._warm_start = warm_start
        self._fit_result = None

    def fit(self, X, y):  # pylint: disable=invalid-name
        """ fit """
        if self._neural_network.dense:

            def objective(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    # TODO: enable batching / proper handling of batches
                    val += self._loss(self._neural_network.forward(x, w), np.array([y_target]))
                return val

            def objective_grad(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    # TODO: allow setting which gradients to evaluate (input/weights)
                    _, weights_grad = self._neural_network.backward(x, w)
                    # TODO: can we store the forward result and reuse it?
                    val += self._loss.gradient(self._neural_network.forward(x, w)
                                               [0], y_target) * weights_grad
                return val

        else:

            def objective(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    probs = self._neural_network.forward(x, w)
                    for y_predict, prob in probs.items():
                        val += prob * self._loss(y_predict, y_target)
                return val

            def objective_grad(w):
                grad = np.zeros(self._neural_network.num_weights)
                for x, y_target in zip(X, y):
                    _, weight_prob_grad = self._neural_network.backward(x, w)
                    for i in range(self._neural_network.num_weights):
                        for y_predict, p_grad in weight_prob_grad[i].items():
                            grad[i] += p_grad * self._loss(y_predict, y_target)
                return grad

        if self._warm_start and self._fit_result is not None:
            initial_point = self._fit_result[0]
        else:
            initial_point = np.random.rand(self._neural_network.num_weights)

        self._fit_result = self._optimizer.optimize(self._neural_network.num_weights, objective,
                                                    objective_grad, initial_point=initial_point)
        return self

    def predict(self, X):  # pylint: disable=invalid-name
        """ predict """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        # TODO: proper handling of batching
        result = np.zeros(len(X))
        for i, x in enumerate(X):
            # TODO: handle sampling case too
            result[i] = self._neural_network.forward(x, self._fit_result[0])
        return result

    def score(self, X, y):  # pylint: disable=invalid-name
        """ score """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')
        return np.sum(self.predict(X) - y) / len(y)
