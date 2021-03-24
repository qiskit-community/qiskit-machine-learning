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

from typing import Union
import numpy as np

from qiskit.algorithms.optimizers import Optimizer

from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions.loss import Loss, L1Loss, L2Loss


class NeuralNetworkClassifier:
    """Quantum neural network classifier."""

    def __init__(self, neural_network: NeuralNetwork, loss: Union[str, Loss] = 'l2',
                 eval_probabilities: bool = False,  # TODO: better name?
                 optimizer: Optimizer = None, warm_start: bool = False):
        """
        Args:
            neural_network: An instance of an quantum neural network.
            loss: A target loss function to be used in training. Default is `l2`, L2 loss.
            eval_probabilities: Determines in the case of a multi-dimensional result of the
                neural_network how to interpret the result. If True it is interpreted as a single
                sample (a probability vector, e.g. for 'CrossEntropy' loss function), and if False
                as a set of individual predictions with occurance probabilities (the index would be
                the prediction and the value the corresponding frequency).
            optimizer: An instance of an optimizer to be used in training.
            warm_start: Use weights from previous fit to start next fit.

        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """

        # TODO: add getter and some setter (warm_start, loss, optimizer, neural_network?)
        self._neural_network = neural_network
        if len(neural_network.output_shape) > 1:
            raise QiskitMachineLearningError('Invalid neural network output shape!')
        if isinstance(loss, Loss):
            self._loss = loss
        else:
            if loss.lower() == 'l1':
                self._loss = L1Loss()
            elif loss.lower() == 'l2':
                self._loss = L2Loss()
            else:
                raise QiskitMachineLearningError(f'Unknown loss {loss}!')

        self._eval_probabilities = eval_probabilities
        self._optimizer = optimizer
        self._warm_start = warm_start
        self._fit_result = None

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        """
        Fit the model to data matrix X and target(s) y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained classifier.
        """

        if self._neural_network.output_shape == (1,):

            def objective(w):

                predict = self._neural_network.forward(X, w)
                target = np.array(y).reshape(predict.shape)
                value = np.sum(self._loss(predict, target))
                return value

            def objective_grad(w):

                # TODO should store output from forward pass (implement loss interface?)
                output = self._neural_network.forward(X, w)
                # TODO: need to be able to turn off input grads if not needed.
                _, weights_grad = self._neural_network.backward(X, w)

                grad = np.zeros((1, self._neural_network.num_weights))
                for i in range(len(X)):
                    grad += self._loss.gradient(output[i][0], y[i]) * weights_grad[i]

                return grad

        else:

            if self._eval_probabilities:

                def objective(w):
                    val = 0.0
                    probs = self._neural_network.forward(X, w)
                    for i in range(len(X)):
                        val += self._loss(probs[i], y[i])
                    return val

                def objective_grad(w):
                    grad = np.zeros((1, self._neural_network.num_weights))
                    for x, y_target in zip(X, y):
                        # TODO: do batch eval
                        y_predict = self._neural_network.forward(x, w)
                        _, weight_prob_grad = self._neural_network.backward(x, w)
                        grad += self._loss.gradient(y_predict, y_target) @ weight_prob_grad[0, :]
                    return grad

            else:

                def objective(w):
                    val = 0.0
                    probs = self._neural_network.forward(X, w)
                    for i in range(len(X)):
                        for y_predict, prob in enumerate(probs[i]):
                            val += prob * self._loss(y_predict, y[i])
                    return val

                def objective_grad(w):
                    num_classes = self._neural_network.output_shape[0]
                    grad = np.zeros((1, self._neural_network.num_weights))
                    for x, y_target in zip(X, y):
                        # TODO: do batch eval
                        _, weight_prob_grad = self._neural_network.backward(x, w)
                        for i in range(num_classes):
                            grad += weight_prob_grad[
                                0, i, :].reshape(grad.shape) * self._loss(i, y_target)
                    return grad

        if self._warm_start and self._fit_result is not None:
            initial_point = self._fit_result[0]
        else:
            initial_point = np.random.rand(self._neural_network.num_weights)

        self._fit_result = self._optimizer.optimize(self._neural_network.num_weights, objective,
                                                    objective_grad, initial_point=initial_point)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict using the network specified to the classifier.

        Args:
            X: The input data.
        Raises:
            QiskitMachineLearningError: Model needs to be fit to some training data first
        Returns:
            The predicted classes.
        """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        # TODO: currently its either {-1, +1} (sign) or {0, 1} (argmax), needs to be cleaned up.
        if self._neural_network.output_shape == (1,):
            predict = np.sign(self._neural_network.forward(X, self._fit_result[0]))
        else:
            forward = self._neural_network.forward(X, self._fit_result[0])
            predict_ = np.argmax(forward, axis=1)
            if self._eval_probabilities:
                predict = np.zeros(forward.shape)
                for i, v in enumerate(predict_):
                    predict[i, v] = 1
            else:
                predict = predict_
        return predict

    def score(self, X: np.ndarray, y: np.ndarray) -> int:  # pylint: disable=invalid-name
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X: Test samples.
            y: True labels for `X`.
        Raises:
            QiskitMachineLearningError: Model needs to be fit to some training data first
        Returns:
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        predict = self.predict(X)
        score = np.sum(predict == y.reshape(predict.shape)) / len(y)
        if len(predict.shape) == 2:
            score /= predict.shape[1]
        return score
