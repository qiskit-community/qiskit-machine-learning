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

""" Loss utilities """

from abc import ABC, abstractmethod

import numpy as np

from ...exceptions import QiskitMachineLearningError


class Loss(ABC):
    """
    Abstract base class for computing Loss.
    """

    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        This method calls the ``evaluate`` method. This is a convenient method to compute loss.
        """
        return self.evaluate(predict, target)

    @abstractmethod
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        An abstract method for evaluating the loss function. Inputs are expected in a shape
        of ``(N, *)``. Where ``N`` is a number of samples. Loss is computed for each sample
        individually.

        Args:
            predict: an array of predicted values using the model.
            target: an array of the true values.

        Returns:
            An array with values of the loss function of the shape ``(N, 1)``.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match
        """
        raise NotImplementedError

    @staticmethod
    def _validate_shapes(predict: np.ndarray, target: np.ndarray) -> None:
        """
        Validates that shapes of both parameters are identical.

        Args:
            predict: an array of predicted values using the model
            target: an array of the true values

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """

        if predict.shape != target.shape:
            raise QiskitMachineLearningError(
                f"Shapes don't match, predict: {predict.shape}, target: {target.shape}!"
            )

    @abstractmethod
    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        An abstract method for computing the gradient. Inputs are expected in a shape
        of ``(N, *)``. Where ``N`` is a number of samples. Gradient is computed for each sample
        individually.

        Args:
            predict: an array of predicted values using the model.
            target: an array of the true values.

        Returns:
            An array with gradient values of the shape ``(N, *)``. The output shape depends on
            the loss function.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """
        raise NotImplementedError


class L1Loss(Loss):
    r"""
    This class computes the L1 loss (i.e. absolute error) for each sample as:

    .. math::

        \text{L1Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} \left| predict_i -
        target_i \right|.
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(predict.shape) <= 1:
            return np.abs(predict - target)
        else:
            return np.linalg.norm(predict - target, ord=1, axis=tuple(range(1, len(predict.shape))))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        return np.sign(predict - target)


class L2Loss(Loss):
    r"""
    This class computes the L2 loss (i.e. squared error) for each sample as:

    .. math::

        \text{L2Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} (predict_i - target_i)^2.

    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(predict.shape) <= 1:
            return (predict - target) ** 2
        else:
            return np.linalg.norm(predict - target, axis=tuple(range(1, len(predict.shape)))) ** 2

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        return 2 * (predict - target)


class CrossEntropyLoss(Loss):
    r"""
    This class computes the cross entropy loss for each sample as:

    .. math::

        \text{CrossEntropyLoss}(predict, target) = -\sum_{i=0}^{N_{\text{classes}}}
        target_i * log(predict_i).
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)

        # multiply target and log(predict) matrices row by row and sum up each row
        # into a single float, so the output is of shape(N,), where N number or samples.
        # then reshape
        # before taking the log we clip the predicted probabilities at a small positive number. This
        # ensures that in cases where a class is predicted to have 0 probability we don't get `nan`.
        val = -np.einsum(
            "ij,ij->i", target, np.log2(np.clip(predict, a_min=1e-10, a_max=None))
        ).reshape(-1, 1)
        return val

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Assume softmax is used, and target vector may or may not be one-hot encoding"""

        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)

        # sum up target along rows, then multiply predict by this sum element wise,
        # then subtract target
        grad = np.einsum("ij,i->ij", predict, np.sum(target, axis=1)) - target

        return grad


class CrossEntropySigmoidLoss(Loss):
    """
    This class computes the cross entropy sigmoid loss and should be used for binary classification.
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(set(target)) != 2:
            raise QiskitMachineLearningError(
                "Sigmoid Cross Entropy is used for binary classification!"
            )

        x = CrossEntropyLoss()
        return 1.0 / (1.0 + np.exp(-x.evaluate(predict, target)))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        return target * (1.0 / (1.0 + np.exp(-predict)) - 1) + (1 - target) * (
            1.0 / (1.0 + np.exp(-predict))
        )
