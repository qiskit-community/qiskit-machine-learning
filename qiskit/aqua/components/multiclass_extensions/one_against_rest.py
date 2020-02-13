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
The One Against Rest multiclass extension.
"""

import logging

import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import LabelBinarizer
from .multiclass_extension import MulticlassExtension

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class OneAgainstRest(MulticlassExtension):
    r"""
    The One Against Rest multiclass extension.

    For an :math:`n`-class problem, the **one-against-rest** method constructs :math:`n`
    SVM classifiers, with the :math:`i`-th classifier separating class :math:`i` from all the
    remaining classes, :math:`\forall i \in \{1, 2, \ldots, n\}`. When the :math:`n` classifiers
    are combined to make the final decision, the classifier that generates the highest value from
    its decision function is selected as the winner and the corresponding class label is returned.
    """

    def __init__(self) -> None:
        super().__init__()
        self.label_binarizer_ = None
        self.classes = None
        self.estimators = None

    def train(self, x, y):
        """
        Training multiple estimators each for distinguishing a pair of classes.

        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        Raises:
            Exception: given all data points are assigned to the same class,
                        the prediction would be boring
        """
        self.label_binarizer_ = LabelBinarizer(neg_label=0)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes = self.label_binarizer_.classes_
        columns = (np.ravel(col) for col in Y.T)
        self.estimators = []
        for _, column in enumerate(columns):
            unique_y = np.unique(column)
            if len(unique_y) == 1:
                raise Exception("given all data points are assigned to the same class, "
                                "the prediction would be boring.")
            estimator = self.estimator_cls(*self.params)
            estimator.fit(x, column)
            self.estimators.append(estimator)

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
        return 1 - (diff * 1.0 / _l)

    def predict(self, x):
        """
        Applying multiple estimators for prediction.

        Args:
            x (numpy.ndarray): NxD array
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        n_samples = _num_samples(x)
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)
        for i, e in enumerate(self.estimators):
            pred = np.ravel(e.decision_function(x))
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        return self.classes[np.array(argmaxima.T)]
