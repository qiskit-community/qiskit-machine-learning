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
        """ evaluate """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, predict, target):
        """ gradient """
        raise NotImplementedError


class L2Loss(Loss):
    """ L2Loss """

    def evaluate(self, predict, target):
        predict = np.array(predict)
        target = np.array(target)
        if len(predict.shape) <= 1:
            return np.linalg.norm(predict - target)**2
        elif len(predict.shape) > 1:
            return np.linalg.norm(predict - target, axis=len(predict.shape)-1)**2
        else:
            raise QiskitMachineLearningError(f'Invalid shape {predict.shape}!')

    def gradient(self, predict, target):
        predict = np.array(predict)
        target = np.array(target)
        return 2*(predict - target)


class L1Loss(Loss):
    """ L1Loss """

    def evaluate(self, predict, target):
        predict = np.array(predict)
        target = np.array(target)
        if predict.shape != target.shape:
            raise QiskitMachineLearningError(f'Invalid shape {predict.shape}!')
        if len(predict.shape) == 0:
            return np.abs(predict - target)
        elif len(predict.shape) <= 1:
            return np.linalg.norm(predict - target, ord=1)
        elif len(predict.shape) > 1:
            return np.linalg.norm(predict - target, ord=1, axis=tuple(range(1, len(predict.shape))))
        else:
            raise QiskitMachineLearningError(f'Invalid shape {predict.shape}!')

    def gradient(self, predict, target):
        predict = np.array(predict)
        target = np.array(target)
        if predict.shape != target.shape:
            raise QiskitMachineLearningError(f'Invalid shape {predict.shape}!')
        if len(predict.shape) <= 1:
            return np.sign(predict - target)
        elif len(predict.shape) > 1:
            return np.sign(predict - target)
        else:
            raise QiskitMachineLearningError(f'Invalid shape {predict.shape}!')


#################################################
#################################################
# TBD
#################################################
#################################################

class L2LossProbability(Loss):
    """ L2LossProbability """

    def __init__(self, predict, target):  # predict and target are both probabilities
        super().__init__(predict, target)
        self.joint_keys = set(predict.keys())
        self.joint_keys.update(target.keys())

    def evaluate(self):
        val = 0.0
        for k in self.joint_keys:
            val += (self.predict.get(k, 0) - self.target.get(k, 0))**2
        return val

    def gradient(self):
        val = {}
        for k in self.joint_keys:
            val[k] = 2*(self.predict.get(k, 0) - self.target.get(k, 0))
        return val


class CrossEntropyLoss(Loss):
    """ CrossEntropyLoss """

    def evaluate(self, predict, target):
        return -np.sum([target[i]*np.log2(predict[i]) for i in range(len(predict))])

    def gradient(self, predict, target):
        pass  # TODO
    # gradient depends on how to handling softmax


class KLDivergence(Loss):
    """ KLDivergence """

    def __init__(self, predict, target):  # predict and target are both probabilities
        super().__init__(predict, target)
        self.predict = np.array(predict)
        self.target = np.array(target)

    def evaluate(self):
        return sum(predict[i] * np.log2(predict[i]/target[i]) for i in range(len(predict)))
