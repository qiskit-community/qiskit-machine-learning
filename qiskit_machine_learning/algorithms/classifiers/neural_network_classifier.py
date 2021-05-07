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
from sklearn.base import ClassifierMixin

from ..trainable_model import BinaryObjectiveFunction, OneHotObjectiveFunction, MultiClassObjectiveFunction
from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions import (
    Loss,
    L1Loss,
    L2Loss,
    CrossEntropyLoss,
    CrossEntropySigmoidLoss,
)


class NeuralNetworkClassifier(ClassifierMixin):
    """Quantum neural network classifier. Implements Scikit-Learn compatible methods for
    classification and extends ``ClassifierMixin``. See `Scikit-Learn <https://scikit-learn.org>`__
    for more details.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: Union[str, Loss] = "l2",
        one_hot: bool = False,
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
            one_hot: Determines in the case of a multi-dimensional result of the
                neural_network how to interpret the result. If True it is interpreted as a single
                one-hot-encoded sample (e.g. for 'CrossEntropy' loss function), and if False
                as a set of individual predictions with occurrence probabilities (the index would be
                the prediction and the value the corresponding frequency, e.g. for L1/L2 loss).
                This option is ignored in case of a one-dimensional output.
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
            if loss.lower() == "l1":
                self._loss = L1Loss()
            elif loss.lower() == "l2":
                self._loss = L2Loss()
            elif loss.lower() == "cross_entropy":
                self._loss = CrossEntropyLoss()
            elif loss.lower() == "cross_entropy_sigmoid":
                self._loss = CrossEntropySigmoidLoss()
            else:
                raise QiskitMachineLearningError(f"Unknown loss {loss}!")

        self._one_hot = one_hot
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
    def one_hot(self):
        """Returns the underlying neural network."""
        return self._one_hot

    @property
    def warm_start(self) -> bool:
        """Returns the warm start flag."""
        return self._warm_start

    @warm_start.setter
    def warm_start(self, warm_start: bool) -> None:
        """Sets the warm start flag."""
        self._warm_start = warm_start

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        """
        Fit the model to data matrix X and target(s) y.

        Args:
            X: The input data.
            y: The target values.

        Returns:
            self: returns a trained classifier.

        Raises:
            QiskitMachineLearningError: In case of invalid data (e.g. incompatible with network)
        """

        if self._neural_network.output_shape == (1,):

            if len(y.shape) != 1 or len(np.unique(y)) != 2:
                raise QiskitMachineLearningError(
                    f"Current settings only applicable to binary classification! Got labels: {y}"
                )
            # binary classification
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)

            # def objective(w):
            #
            #     predict = self._neural_network.forward(X, w)
            #     target = np.array(y).reshape(predict.shape)
            #     value = np.sum(self._loss(predict, target))
            #     return value
            #
            # def objective_grad(w):
            #
            #     # TODO should store output from forward pass (implement loss interface?)
            #     # TODO: need to be able to turn off input grads if not needed.
            #     output = self._neural_network.forward(X, w)
            #     _, weights_grad = self._neural_network.backward(X, w)
            #
            #     grad = np.zeros((1, self._neural_network.num_weights))
            #     for i in range(len(X)):
            #         grad += self._loss.gradient(output[i][0], y[i]) * weights_grad[i]
            #
            #     return grad

        else:

            if self._one_hot:
                function = OneHotObjectiveFunction(X, y, self._neural_network, self._loss)
                # def objective(w):
                #     val = 0.0
                #     probs = self._neural_network.forward(X, w)
                #     for i in range(len(X)):
                #         val += self._loss(probs[i], y[i])
                #     return val
                #
                # def objective_grad(w):
                #     grad = np.zeros(self._neural_network.num_weights)
                #     for x, y_target in zip(X, y):
                #         # TODO: do batch eval
                #         y_predict = self._neural_network.forward(x, w)
                #         _, weight_prob_grad = self._neural_network.backward(x, w)
                #         grad += self._loss.gradient(y_predict[0], y_target) @ weight_prob_grad[0, :]
                #     return grad

            else:
                function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)
                # def objective(w):
                #     val = 0.0
                #     probs = self._neural_network.forward(X, w)
                #     for i in range(len(X)):
                #         for y_predict, prob in enumerate(probs[i]):
                #             val += prob * self._loss(y_predict, y[i])
                #     return val
                #
                # def objective_grad(w):
                #     num_classes = self._neural_network.output_shape[0]
                #     grad = np.zeros((1, self._neural_network.num_weights))
                #     for x, y_target in zip(X, y):
                #         # TODO: do batch eval
                #         _, weight_prob_grad = self._neural_network.backward(x, w)
                #         for i in range(num_classes):
                #             grad += weight_prob_grad[
                #                 0, i, :].reshape(grad.shape) * self._loss(i, y_target)
                #     return grad

        if self._warm_start and self._fit_result is not None:
            initial_point = self._fit_result[0]
        else:
            initial_point = np.random.rand(self._neural_network.num_weights)

        self._fit_result = self._optimizer.optimize(
            self._neural_network.num_weights,
            function.objective,
            function.gradient,
            initial_point=initial_point,
        )
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
            raise QiskitMachineLearningError("Model needs to be fit to some training data first!")
        if self._neural_network.output_shape == (1,):
            predict = np.sign(self._neural_network.forward(X, self._fit_result[0]))
        else:
            forward = self._neural_network.forward(X, self._fit_result[0])
            predict_ = np.argmax(forward, axis=1)
            if self._one_hot:
                predict = np.zeros(forward.shape)
                for i, v in enumerate(predict_):
                    predict[i, v] = 1
            else:
                predict = predict_
        return predict
