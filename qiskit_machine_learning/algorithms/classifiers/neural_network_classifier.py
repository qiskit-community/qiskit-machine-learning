# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network classifier."""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
import scipy.sparse
from qiskit.algorithms.optimizers import Optimizer
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
        loss: str | Loss = "squared_error",
        one_hot: bool = False,
        optimizer: Optimizer | None = None,
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
                'squared_error', 'cross_entropy', 'cross_entropy_sigmoid', or as a loss function
                implementing the Loss interface.
            one_hot: Determines in the case of a multi-dimensional result of the
                neural_network how to interpret the result. If True it is interpreted as a single
                one-hot-encoded sample (e.g. for 'CrossEntropy' loss function), and if False
                as a set of individual predictions with occurrence probabilities (the index would be
                the prediction and the value the corresponding frequency, e.g. for absolute/squared
                loss). In case of a one-dimensional categorical output, this option determines how
                to encode the target data (i.e. one-hot or integer encoding).
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        super().__init__(neural_network, loss, optimizer, warm_start, initial_point, callback)
        self._one_hot = one_hot
        # encodes the target data if categorical
        self._target_encoder = OneHotEncoder(sparse=False) if one_hot else LabelEncoder()

        # For ensuring the number of classes matches those of the previous
        # batch when training from a warm start.
        self._num_classes: int | None = None

    @property
    def num_classes(self) -> int | None:
        """The number of classes found in the most recent fit.

        If called before :meth:`fit`, this will return ``None``.
        """
        # For user checking and validation.
        return self._num_classes

    def fit(self, X: np.ndarray, y: np.ndarray) -> NeuralNetworkClassifier:
        if not self._warm_start:
            self._fit_result = None
        X, y = self._validate_input(X, y)

        # mypy definition
        function: ObjectiveFunction = None
        if self._neural_network.output_shape == (1,):
            self._validate_binary_targets(y)
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            if self._one_hot:
                self._validate_one_hot_targets(y)
                function = OneHotObjectiveFunction(X, y, self._neural_network, self._loss)
            else:
                function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)

        objective = self._get_objective(function)

        self._fit_result = self._optimizer.minimize(
            fun=objective,
            x0=self._choose_initial_point(),
            jac=function.gradient,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        if self._fit_result is None:
            raise QiskitMachineLearningError("Model needs to be fit to some training data first!")

        X, _ = self._validate_input(X)

        if self._neural_network.output_shape == (1,):
            predict = np.sign(self._neural_network.forward(X, self._fit_result.x))
        else:
            forward = self._neural_network.forward(X, self._fit_result.x)
            predict_ = np.argmax(forward, axis=1)
            if self._one_hot:
                predict = np.zeros(forward.shape)
                for i, v in enumerate(predict_):
                    predict[i, v] = 1
            else:
                predict = predict_
        return predict

    # pylint: disable=invalid-name
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        X, y = self._validate_input(X, y)
        return ClassifierMixin.score(self, X, y, sample_weight)

    def _validate_input(self, X: np.ndarray, y: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Validates and transforms if required features and labels. If arrays are sparse, they are
        converted to dense as the numpy math in the loss/objective functions does not work with
        sparse. If one hot encoding is required, then labels are one hot encoded otherwise label
        are encoded via ``LabelEncoder`` from ``SciKit-Learn``. If labels are strings, they
        converted to numerical representation.

        Args:
            X: features
            y: labels

        Returns:
            A tuple with validated features and labels.
        """
        if scipy.sparse.issparse(X):
            # our math does not work with sparse arrays
            X = cast(spmatrix, X).toarray()  # cast is required by mypy

        if y is not None:
            self._num_classes = self._get_num_classes(y)
            if isinstance(y[0], str):
                # string data is assumed to be categorical

                # OneHotEncoder expects data with shape (n_samples, n_features) but
                # LabelEncoder expects shape (n_samples,) so set desired shape
                y = y.reshape(-1, 1) if self._one_hot else y
                if self._fit_result is None:
                    # the model is being trained, fit first
                    self._target_encoder.fit(y)
                y = self._target_encoder.transform(y)
            elif scipy.sparse.issparse(y):
                y = cast(spmatrix, y).toarray()  # cast is required by mypy

        return X, y

    def _validate_binary_targets(self, y: np.ndarray) -> None:
        """Validate binary encoded targets.

        Raises:
            QiskitMachineLearningError: If targets are invalid.
        """
        if len(y.shape) != 1:
            raise QiskitMachineLearningError(
                "The shape of the targets does not match the shape of neural network output."
            )
        if len(np.unique(y)) != 2:
            raise QiskitMachineLearningError(
                "The target values appear to be multi-classified. "
                "The neural network output shape is only suitable for binary classification."
            )

    def _validate_one_hot_targets(self, targets: np.ndarray) -> None:
        """Validate one-hot encoded targets.

        Ensure one-hot encoded data is valid and not multi-label.

        Raises:
            QiskitMachineLearningError: If targets are invalid.
        """
        if not np.isin(targets, [0, 1]).all():
            raise QiskitMachineLearningError(
                "Invalid one-hot targets. The targets must contain only 0's and 1's."
            )

        if not np.isin(np.sum(targets, axis=-1), 1).all():
            raise QiskitMachineLearningError(
                "The target values appear to be multi-labelled. "
                "Multi-label classification is not supported."
            )

    def _get_num_classes(self, y: np.ndarray) -> int:
        """Infers the number of classes from the targets.

        Args:
            y: The target values.

        Raises:
            QiskitMachineLearningError: If the number of classes differs from
            the previous batch when using a warm start.

        Returns:
            The number of inferred classes.
        """
        if self._one_hot:
            num_classes = y.shape[-1]
        else:
            num_classes = len(np.unique(y))

        if self._warm_start and self._num_classes is not None and self._num_classes != num_classes:
            raise QiskitMachineLearningError(
                f"The number of classes ({num_classes}) is different to the previous batch "
                f"({self._num_classes})."
            )
        return num_classes
