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
import numpy as np

from ...exceptions import QiskitMachineLearningError


class Loss(ABC):
    """
    Abstract base class for Loss.
    """

    def __call__(self, predict, target):
        return self.evaluate(predict, target)

    @abstractmethod
    def evaluate(self, predict, target):
        """evaluate"""
        raise NotImplementedError

    @abstractmethod
    def gradient(self, predict, target):
        """gradient"""
        raise NotImplementedError

    @staticmethod
    def _validate(predict, target):
        predict = np.array(predict)
        target = np.array(target)
        if predict.shape != target.shape:
            raise QiskitMachineLearningError(
                f"Shapes don't match, predict: {predict.shape}, "
                f"target: {target.shape}!"
            )
        return predict, target


class L1Loss(Loss):
    """L1Loss"""

    def evaluate(self, predict, target):
        predict, target = self._validate(predict, target)

        if len(predict.shape) == 0:
            return np.abs(predict - target)
        elif len(predict.shape) <= 1:
            return np.linalg.norm(predict - target, ord=1)
        elif len(predict.shape) > 1:
            return np.linalg.norm(
                predict - target, ord=1, axis=tuple(range(1, len(predict.shape)))
            )
        else:
            raise QiskitMachineLearningError(f"Invalid shape {predict.shape}!")

    def gradient(self, predict, target):
        predict, target = self._validate(predict, target)

        return np.sign(predict - target)


class L2Loss(Loss):
    """L2Loss"""

    def evaluate(self, predict, target):
        predict, target = self._validate(predict, target)

        if len(predict.shape) <= 1:
            return np.linalg.norm(predict - target) ** 2
        elif len(predict.shape) > 1:
            return np.linalg.norm(predict - target, axis=len(predict.shape) - 1) ** 2
        else:
            raise QiskitMachineLearningError(f"Invalid shape {predict.shape}!")

    def gradient(self, predict, target):
        predict, target = self._validate(predict, target)

        return 2 * (predict - target)


class CrossEntropyLoss(Loss):
    """CrossEntropyLoss"""

    def evaluate(self, predict, target):
        predict, target = self._validate(predict, target)

        return -np.sum([target[i] * np.log2(predict[i]) for i in range(len(predict))])

    def gradient(self, predict, target):
        """Assume softmax is used, and target vector may or may not be one-hot encoding"""
        predict, target = self._validate(predict, target)

        return predict * np.sum(target) - target


class CrossEntropySigmoidLoss(Loss):
    """This is used for binary classification"""

    def evaluate(self, predict, target):
        predict, target = self._validate(predict, target)

        if len(set(target)) != 2:
            raise QiskitMachineLearningError(
                "Sigmoid Cross Entropy is used for binary classification!"
            )

        x = CrossEntropyLoss()
        return 1.0 / (1.0 + np.exp(-x.evaluate(predict, target)))

    def gradient(self, predict, target):
        predict, target = self._validate(predict, target)

        return target * (1.0 / (1.0 + np.exp(-predict)) - 1) + (1 - target) * (
            1.0 / (1.0 + np.exp(-predict))
        )
