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
"""An implementation of quantum neural network classifier."""

import numpy as np
from typing import Union

from qiskit.algorithms.optimizers import Optimizer

from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions.loss import Loss, L1Loss, L2Loss


class NeuralNetworkClassifier:
    """Quantum neural network classifier."""

    def __init__(self, neural_network: NeuralNetwork, loss: Union[str, Loss] = 'l2',
                 optimizer: Optimizer = None, warm_start: bool = False, callback=None):
        """
        Args:
            neural_network: An instance of an quantum neural network.
            loss: A target loss function to be used in training. Default is `l2`, L2 loss.
            optimizer: An instance of an optimizer to be used in training.
            warm_start:
            callback:
        """
        # TODO: callback

        self._neural_network = neural_network
        if isinstance(loss, str):
            if loss.lower() == 'l1':
                self._loss = L1Loss()
            elif loss.lower() == 'l2':
                self._loss = L2Loss()
            else:
                raise QiskitMachineLearningError(f'Unknown loss {loss}!')
        else:
            self._loss = loss
        self._optimizer = optimizer
        self._warm_start = warm_start
        self._fit_result = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to data matrix X and target(s) y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained classifier.

        """
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
                    val += self._loss.gradient(
                        self._neural_network.forward(x, w), y_target) * weights_grad
                return val

        else:

            def objective(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    probs = self._neural_network.forward(x, w)
                    for y_predict, p in probs.items():
                        val += p * self._loss(y_predict, y_target)
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

    def predict(self, X: np.ndarray):
        """
        Predict using the network specified to the classifier.

        Args:
            X: The input data.

        Returns:
            The predicted classes.
        """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        # TODO: proper handling of batching
        result = np.zeros(len(X))
        for i, x in enumerate(X):
            # TODO: handle sampling case too
            result[i] = np.sign(self._neural_network.forward(x, self._fit_result[0]))
        return result

    def score(self, X: np.ndarray, y: np.ndarray):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X: Test samples.
            y: True labels for `X`.

        Returns:
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')
        return np.sum(nnc.predict(X) == y) / len(y)
