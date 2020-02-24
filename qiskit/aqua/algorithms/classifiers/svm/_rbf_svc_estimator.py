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

"""The estimator that uses the RBF Kernel."""

from sklearn.svm import SVC

from qiskit.aqua.components.multiclass_extensions import Estimator

# pylint: disable=invalid-name


class _RBF_SVC_Estimator(Estimator):
    """The estimator that uses the RBF Kernel."""

    def __init__(self):
        self._estimator = SVC(kernel='rbf', gamma='auto')

    def fit(self, x, y):
        """
        fit values for the points and the labels
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self._estimator.fit(x, y)

    def decision_function(self, x):
        """
        predicted values for the points which account for both the labels and the confidence
        Args:
            x (numpy.ndarray): input points
        Returns:
            numpy.ndarray: decision function
        """
        return self._estimator.decision_function(x)
