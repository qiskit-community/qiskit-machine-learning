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

from typing import Union, Optional

import numpy as np
from qiskit.algorithms.optimizers import Optimizer
from sklearn.base import ClassifierMixin

from ..objective_functions import (
    BinaryObjectiveFunction,
    OneHotObjectiveFunction,
    MultiClassObjectiveFunction,
    ObjectiveFunction,
)
from ..trainable_model import TrainableModel
from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions import Loss


class NeuralNetworkClassifier(TrainableModel, ClassifierMixin):
    """Quantum neural network classifier. Implements Scikit-Learn compatible methods for
    classification and extends ``ClassifierMixin``. See `Scikit-Learn <https://scikit-learn.org>`__
    for more details.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: Union[str, Loss] = "squared_error",
        one_hot: bool = False,
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
            one_hot: Determines in the case of a multi-dimensional result of the
                neural_network how to interpret the result. If True it is interpreted as a single
                one-hot-encoded sample (e.g. for 'CrossEntropy' loss function), and if False
                as a set of individual predictions with occurrence probabilities (the index would be
                the prediction and the value the corresponding frequency, e.g. for absolute/squared
                loss). This option is ignored in case of a one-dimensional output.
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.

        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        super().__init__(neural_network, loss, optimizer, warm_start, initial_point)
        self._one_hot = one_hot

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        # mypy definition
        function: ObjectiveFunction = None
        if self._neural_network.output_shape == (1,):
            if len(y.shape) != 1 or len(np.unique(y)) != 2:
                raise QiskitMachineLearningError(
                    f"Current settings only applicable to binary classification! Got labels: {y}"
                )
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            if self._one_hot:
                function = OneHotObjectiveFunction(X, y, self._neural_network, self._loss)
            else:
                function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)

        self._fit_result = self._optimizer.optimize(
            self._neural_network.num_weights,
            function.objective,
            function.gradient,
            initial_point=self._choose_initial_point(),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
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

    # pylint: disable=invalid-name
    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        return ClassifierMixin.score(self, X, y, sample_weight)
