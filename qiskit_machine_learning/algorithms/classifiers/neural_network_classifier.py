# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
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
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ..objective_functions import (
    BinaryObjectiveFunction,
    OneHotObjectiveFunction,
    MultiClassObjectiveFunction,
    ObjectiveFunction,
)
from ..trainable_model import TrainableModel
from ...optimizers import Optimizer, OptimizerResult, Minimizer
from ...exceptions import QiskitMachineLearningError
from ...neural_networks import NeuralNetwork
from ...utils.loss_functions import Loss


class NeuralNetworkClassifier(TrainableModel, ClassifierMixin):
    """Implements a basic quantum neural network classifier. Implements Scikit-Learn compatible
    methods for classification and extends ``ClassifierMixin``.
    See `Scikit-Learn <https://scikit-learn.org>`__ for more details.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: str | Loss = "squared_error",
        one_hot: bool = False,
        optimizer: Optimizer | Minimizer | None = None,
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
                'squared_error', 'cross_entropy', or as a loss function
                implementing the Loss interface.
            one_hot: Determines in the case of a multi-dimensional result of the
                neural_network how to interpret the result. If True it is interpreted as a single
                one-hot-encoded sample (e.g. for 'CrossEntropy' loss function), and if False
                as a set of individual predictions with occurrence probabilities (the index would be
                the prediction and the value the corresponding frequency, e.g. for absolute/squared
                loss). In case of a one-dimensional categorical output, this option determines how
                to encode the target data (i.e. one-hot or integer encoding).
            optimizer: An instance of an optimizer or a callable to be used in training.
                Refer to  :class:`~qiskit_machine_learning.optimizers.Minimizer` for more information on
                the callable protocol. When `None` defaults to
                :class:`~qiskit_machine_learning.optimizers.SLSQP`.
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
        self._target_encoder = OneHotEncoder(sparse_output=False) if one_hot else LabelEncoder()

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

    # pylint: disable=invalid-name
    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> OptimizerResult:
        X, y = self._validate_input(X, y)

        function = self._create_objective(X, y)
        return self._minimize(function)

    def _create_objective(self, X: np.ndarray, y: np.ndarray) -> ObjectiveFunction:
        """
        Creates an objective function that depends on the classification we want to solve.

        Args:
            X: The input data.
            y: True values for ``X``.

        Returns:
            An instance of the objective function.
        """
        # mypy definition
        function: ObjectiveFunction = None
        if self._neural_network.output_shape == (1,):
            self._validate_binary_targets(y)
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            if self._one_hot:
                function = OneHotObjectiveFunction(X, y, self._neural_network, self._loss)
            else:
                function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)

        return function

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Args:
            X (np.ndarray): Input features. For a callable kernel (an instance of
                :class:`~qiskit_machine_learning.kernels.BaseKernel`), the shape
                should be ``(m_samples, n_features)``. For a pre-computed kernel, the shape should be
                ``(m_samples, n_samples)``. Here, ``m_*`` denotes the set to be
                predicted, and ``n_*`` denotes the size of the training set.
                In the case of a pre-computed kernel, the kernel values in ``X`` must be calculated
                with respect to the elements of the set to be predicted and the training set.

        Returns:
            np.ndarray: An array of shape ``(n_samples,)``, representing the predicted class labels for
                each sample in ``X``.

        Raises:
            QiskitMachineLearningError:
                - If the :meth:`predict` method is called before the model has been fit.
            ValueError:
                - If the pre-computed kernel matrix has the wrong shape and/or dimension.
        """
        self._check_fitted()
        X, _ = self._validate_input(X)

        if self._neural_network.output_shape == (1,):
            # Binary classification
            raw_output = self._neural_network.forward(X, self._fit_result.x)
            predict = np.sign(raw_output)
        else:
            # Multi-class classification
            forward = self._neural_network.forward(X, self._fit_result.x)
            predict_ = np.argmax(forward, axis=1)

            if self._one_hot:
                # Convert class indices to one-hot encoded format
                predict = np.zeros(forward.shape)
                for i, v in enumerate(predict_):
                    predict[i, v] = 1
            else:
                predict = predict_

        return self._validate_output(predict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts the predicted probabilities for each class based on the output of a neural
        network.

        Args:
            X (np.ndarray): Input features. For a callable kernel (an instance of
                :class:`~qiskit_machine_learning.kernels.BaseKernel`), the shape
                should be ``(m_samples, n_features)``. For a pre-computed kernel, the shape should be
                ``(m_samples, n_samples)``. Here, ``m_*`` denotes the set to be
                predicted, and ``n_*`` denotes the size of the training set. In the case of a
                pre-computed kernel, the kernel values in ``X`` must be calculated with respect to
                the elements of the set to be predicted and the training set.

        Returns:
            np.ndarray: An array of shape ``(n_samples, n_classes)`` representing the predicted class
                probabilities (in the range :math:`[0, 1]`) for each sample in ``X``.
        """
        self._check_fitted()
        X, _ = self._validate_input(X)

        # Assumes an activation function is applied within the forward method
        proba = self._neural_network.forward(X, self._fit_result.x)

        return proba

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
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
            A tuple with validated and transformed features and labels.
        """
        if scipy.sparse.issparse(X):
            # our math does not work with sparse arrays
            X = cast(spmatrix, X).toarray()  # cast is required by mypy

        if y is not None:
            if scipy.sparse.issparse(y):
                y = cast(spmatrix, y).toarray()  # cast is required by mypy

            if isinstance(y[0], str):
                y = self._encode_categorical_labels(y)
            elif self._one_hot and not self._validate_one_hot_targets(y, raise_on_failure=False):
                y = self._encode_one_hot_labels(y)

            self._num_classes = self._get_num_classes(y)

        return X, y

    def _encode_categorical_labels(self, y: np.ndarray):
        # string data is assumed to be categorical

        # OneHotEncoder expects data with shape (n_samples, n_features) but
        # LabelEncoder expects shape (n_samples,) so set desired shape
        y = y.reshape(-1, 1) if self._one_hot else y
        if self._fit_result is None:
            # the model is being trained, fit first
            self._target_encoder.fit(y)
        y = self._target_encoder.transform(y)

        return y

    def _encode_one_hot_labels(self, y: np.ndarray):
        # conversion to one hot of the labels is required
        y = y.reshape(-1, 1)
        if self._fit_result is None:
            # the model is being trained, fit first
            self._target_encoder.fit(y)
        y = self._target_encoder.transform(y)

        return y

    def _validate_output(self, y_hat: np.ndarray) -> np.ndarray:
        try:
            check_is_fitted(self._target_encoder)
            return self._target_encoder.inverse_transform(y_hat).squeeze()
        except NotFittedError:
            return y_hat

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

    def _validate_one_hot_targets(self, y: np.ndarray, raise_on_failure=True) -> bool:
        """
        Validate one-hot encoded labels. Ensure one-hot encoded data is valid and not multi-label.

        Args:
            y: targets
            raise_on_failure: If ``True``, raises :class:`~QiskitMachineLearningError` if the labels
                are not one hot encoded. If set to ``False``, returns ``False`` if labels are not
                one hot encoded and no errors are raised.

        Returns:
            ``True`` when targets are one hot encoded, ``False`` otherwise.

        Raises:
            QiskitMachineLearningError: If targets are invalid.
        """
        if len(y.shape) != 2:
            if raise_on_failure:
                raise QiskitMachineLearningError(
                    f"One hot encoded targets must be of shape (num_samples, num_classes), "
                    f"but found {y.shape}."
                )
            return False

        if not np.isin(y, [0, 1]).all():
            if raise_on_failure:
                raise QiskitMachineLearningError(
                    "Invalid one-hot targets. The targets must contain only 0's and 1's."
                )
            return False

        if not np.isin(np.sum(y, axis=-1), 1).all():
            if raise_on_failure:
                raise QiskitMachineLearningError(
                    "The target values appear to be multi-labelled. "
                    "Multi-label classification is not supported."
                )
            return False

        return True

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
