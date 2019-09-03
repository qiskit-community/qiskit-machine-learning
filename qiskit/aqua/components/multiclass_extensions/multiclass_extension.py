# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Base class for multiclass extension """

from abc import abstractmethod
from qiskit.aqua import Pluggable


class MulticlassExtension(Pluggable):
    """
        Base class for multiclass extension.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is available.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        """ init params """
        multiclass_extension_params = params.get(Pluggable.SECTION_KEY_MULTICLASS_EXT)
        args = {k: v for k, v in multiclass_extension_params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def train(self, x, y):
        """
        training multiple estimators each for distinguishing a pair of classes.
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self, x, y):
        """
        testing multiple estimators each for distinguishing a pair of classes.
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        """
        applying multiple estimators for prediction
        Args:
            x (numpy.ndarray): input points
        """
        raise NotImplementedError()
