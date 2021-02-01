# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Error Correcting Code multiclass extension.
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.multiclass import _ConstantPredictor

from qiskit.aqua import aqua_globals
from qiskit.aqua.utils.validation import validate_min
from .multiclass_extension import MulticlassExtension

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class ErrorCorrectingCode(MulticlassExtension):
    r"""
    The Error Correcting Code multiclass extension.

    Error Correcting Code (ECC) is an ensemble method designed for the multiclass classification
    problem.  As for the other multiclass methods, the task is to decide one label from
    :math:`k > 2` possible choices.

    +-------+------------------------------------------------------------------------+
    |       |                                Code Word                               |
    + Class +-----------+------------+-----------+-----------+-----------+-----------+
    |       |:math:`f_0`|:math:`f_1` |:math:`f_2`|:math:`f_3`|:math:`f_4`|:math:`f_5`|
    +-------+-----------+------------+-----------+-----------+-----------+-----------+
    |   1   |     0     |     1      |     0     |     1     |     0     |     1     |
    +-------+-----------+------------+-----------+-----------+-----------+-----------+
    |   2   |     1     |     0      |     0     |     1     |     0     |     0     |
    +-------+-----------+------------+-----------+-----------+-----------+-----------+
    |   3   |     1     |     1      |     1     |     0     |     0     |     0     |
    +-------+-----------+------------+-----------+-----------+-----------+-----------+

    The table above shows a 6-bit ECC for a 3-class problem. Each class is assigned a unique
    binary string of length 6.  The string is also called  a **codeword**.  For example, class 2
    has codeword ``100100``.  During training, one binary classifier is learned for each column.
    For example, for the first column, ECC builds a binary classifier to separate :math:`\{2, 3\}`
    from :math:`\{1\}`. Thus, 6 binary classifiers are trained in this way.  To classify a
    new data point :math:`\mathbf{x}`, all 6 binary classifiers are evaluated to obtain a 6-bit
    string. Finally, we choose the class whose bitstring is closest to :math:`\mathbf{x}`’s
    output string as the predicted label. This implementation of ECC uses the Euclidean distance.
    """

    def __init__(self, code_size: int = 4):
        """
        Args:
            code_size: Size of error correcting code
        """
        validate_min('code_size', code_size, 1)
        super().__init__()
        self.code_size = code_size
        self.rand = aqua_globals.random
        self.estimators = None
        self.classes = None
        self.codebook = None

    def train(self, x, y):
        """
        Training multiple estimators each for distinguishing a pair of classes.

        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self.estimators = []
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]
        code_size = int(n_classes * self.code_size)
        self.codebook = self.rand.random((n_classes, code_size))
        self.codebook[self.codebook > 0.5] = 1
        self.codebook[self.codebook != 1] = 0
        classes_index = dict((c, i) for i, c in enumerate(self.classes))
        Y = np.array([self.codebook[classes_index[y[i]]]
                      for i in range(x.shape[0])], dtype=int)
        # pylint: disable=unsubscriptable-object
        logger.info("Require %s estimators.", Y.shape[1])
        for i in range(Y.shape[1]):
            y_bit = Y[:, i]
            unique_y = np.unique(y_bit)
            if len(unique_y) == 1:
                estimator = _ConstantPredictor()
                estimator.fit(x, unique_y)
            else:
                estimator = self.estimator_cls(*self.params)
                estimator.fit(x, y_bit)
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
        confidences = []
        for e in self.estimators:
            confidence = np.ravel(e.decision_function(x))
            confidences.append(confidence)
        y = np.array(confidences).T
        pred = euclidean_distances(y, self.codebook).argmin(axis=1)
        return self.classes[pred]
