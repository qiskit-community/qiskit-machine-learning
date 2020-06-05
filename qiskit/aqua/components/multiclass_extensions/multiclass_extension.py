# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Base class for multiclass extension """

from typing import Optional, List, Callable
from abc import ABC, abstractmethod
from .estimator import Estimator


class MulticlassExtension(ABC):
    """
        Base class for multiclass extension.

        This method should initialize the module and
        use an exception if a component of the module is not available.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.estimator_cls = None  # type: Optional[Callable[[List], Estimator]]
        self.params = []  # type: List

    def set_estimator(self,
                      estimator_cls: Callable[[List], Estimator],
                      params: Optional[List] = None) -> None:
        """
        Called internally to set :class:`Estimator` and parameters
        Args:
           estimator_cls: An :class:`Estimator` class
           params: Parameters for the estimator
        """
        self.estimator_cls = estimator_cls
        self.params = params if params is not None else []

    @abstractmethod
    def train(self, x, y):
        """
        Training multiple estimators each for distinguishing a pair of classes.

        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self, x, y):
        """
        Testing multiple estimators each for distinguishing a pair of classes.

        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        """
        Applying multiple estimators for prediction.

        Args:
            x (numpy.ndarray): input points
        """
        raise NotImplementedError()
