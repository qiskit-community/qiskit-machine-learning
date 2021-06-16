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

""" Loss utilities """

from abc import ABC, abstractmethod
from typing import Union, Tuple
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
    def _validate(predict: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            predict: an array of predicted values using the model
            target: an array of the true values

        Returns:
            A tuple of predicted values and true values converted to numpy arrays.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """

        predict = np.array(predict)
        target = np.array(target)
        if predict.shape != target.shape:
            raise QiskitMachineLearningError(
                f"Shapes don't match, predict: {predict.shape}, target: {target.shape}!"
            )
        return predict, target

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
    """
    This class computes the L1 loss for each sample as:

    .. math::
        \text{L1Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} \left| predict_i - target_i \right|.
    """

    def evaluate(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)

        predict = np.array(predict)
        target = np.array(target)
        if len(predict.shape) == 0:
            return np.abs(predict - target)
        elif len(predict.shape) <= 1:
            return np.linalg.norm(predict - target, ord=1)
        elif len(predict.shape) > 1:
            return np.linalg.norm(predict - target, ord=1, axis=tuple(range(1, len(predict.shape))))
        else:
            raise QiskitMachineLearningError(f"Invalid shape {predict.shape}!")

    def gradient(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        return np.sign(predict - target)


class L2Loss(Loss):
    """
    This class computes the L2 loss for each sample as:

    .. math::
        \text{L2Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} (predict_i - target_i)^2.

    """

    def evaluate(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        if len(predict.shape) <= 1:
            return np.linalg.norm(predict - target) ** 2
        elif len(predict.shape) > 1:
            return np.linalg.norm(predict - target, axis=len(predict.shape) - 1) ** 2
        else:
            raise QiskitMachineLearningError(f"Invalid shape {predict.shape}!")

    def gradient(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        return 2 * (predict - target)


class CrossEntropyLoss(Loss):
    """
    This class computes the cross entropy loss for each sample as: -sum target * log(predict)

    .. math::
        \text{CrossEntropyLoss}(predict, target) = -\sum_{i=0}^{N_{\text{classes}}} target_i * log(predict_i).
    """

    def evaluate(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        return float(-np.sum([target[i] * np.log2(predict[i]) for i in range(len(predict))]))

    def gradient(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        return predict * np.sum(target) - target


class CrossEntropySigmoidLoss(Loss):
    """
    This class computes the cross entropy sigmoid loss.

    This is used for binary classification.
    """

    def evaluate(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        x = CrossEntropyLoss()
        return 1.0 / (1.0 + np.exp(-x.evaluate(predict, target)))

    def gradient(self, predict: Union[int, np.ndarray], target: Union[int, np.ndarray]) -> float:
        predict, target = self._validate(predict, target)
        predict = np.array(predict)
        target = np.array(target)

        return target * (1.0 / (1.0 + np.exp(-predict)) - 1) + (1 - target) * (
            1.0 / (1.0 + np.exp(-predict))
        )
