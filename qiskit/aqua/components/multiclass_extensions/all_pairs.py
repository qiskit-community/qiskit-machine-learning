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

"""
The All-Pairs multiclass extension.
"""

import logging

import numpy as np
from sklearn.utils.multiclass import _ovr_decision_function

from .multiclass_extension import MulticlassExtension

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class AllPairs(MulticlassExtension):
    """
    The All-Pairs multiclass extension.

    In the **all-pairs** reduction, one trains :math:`k(k−1)/2` binary classifiers for a
    :math:`k`-way multiclass problem; each receives the samples of a pair of classes from the
    original training set, and must learn to distinguish these two classes. At prediction time,
    a **weighted voting scheme** is used: all :math:`k(k−1)/2` classifiers are applied to an unseen
    sample, and each class gets assigned the sum of all the scores obtained by the various
    classifiers. The combined classifier returns as a result the class getting the highest value.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classes_ = None
        self.estimators = None

    def train(self, x, y):
        """
        Training multiple estimators each for distinguishing a pair of classes.

        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        Raises:
            ValueError: can not be fit when only one class is present.
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("can not be fit when only one class is present.")
        n_classes = self.classes_.shape[0]
        self.estimators = {}
        logger.info("Require %s estimators.", n_classes * (n_classes - 1) / 2)
        for i in range(n_classes):
            estimators_from_i = {}
            for j in range(i + 1, n_classes):
                estimator = self.estimator_cls(*self.params)
                cond = np.logical_or(y == i, y == j)
                indcond = np.arange(x.shape[0])[cond]
                x_filtered = x[indcond]
                y_filtered = y[indcond]
                y_filtered[y_filtered == i] = 0
                y_filtered[y_filtered == j] = 1
                estimator.fit(x_filtered, y_filtered)
                estimators_from_i[j] = estimator
            self.estimators[i] = estimators_from_i

    def test(self, x, y):
        """
        Testing multiple estimators each for distinguishing a pair of classes.

        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels

        Returns:
            float: accuracy
        """
        A = self.predict(x)
        B = y
        _l = len(A)
        diff = np.sum(A != B)
        logger.debug("%d out of %d are wrong", diff, _l)
        return 1. - (diff * 1.0 / _l)

    def predict(self, x):
        """
        Applying multiple estimators for prediction.

        Args:
            x (numpy.ndarray): NxD array
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        predictions = []
        confidences = []
        for i in self.estimators:
            estimators_from_i = self.estimators[i]
            for j in estimators_from_i:
                estimator = estimators_from_i[j]
                confidence = np.ravel(estimator.decision_function(x))

                indices = (confidence > 0).astype(np.int)
                prediction = self.classes_[indices]

                predictions.append(prediction.reshape(-1, 1))
                confidences.append(confidence.reshape(-1, 1))

        predictions = np.hstack(predictions)
        confidences = np.hstack(confidences)
        y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        return self.classes_[y.argmax(axis=1)]
