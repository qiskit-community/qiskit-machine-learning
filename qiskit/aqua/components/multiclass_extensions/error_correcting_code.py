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

"""
the multiclass extension based on the error-correcting-code algorithm.
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.multiclass import _ConstantPredictor

from qiskit.aqua import aqua_globals
from qiskit.aqua.components.multiclass_extensions import MulticlassExtension

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class ErrorCorrectingCode(MulticlassExtension):
    """
      the multiclass extension based on the error-correcting-code algorithm.
    """
    CONFIGURATION = {
        'name': 'ErrorCorrectingCode',
        'description': 'ErrorCorrectingCode extension',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'error_correcting_code_schema',
            'type': 'object',
            'properties': {
                'code_size': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, estimator_cls, params=None, code_size=4):
        self.validate(locals())
        super().__init__()
        self.estimator_cls = estimator_cls
        self.params = params if params is not None else []
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
        self.codebook = self.rand.random_sample((n_classes, code_size))
        self.codebook[self.codebook > 0.5] = 1
        self.codebook[self.codebook != 1] = 0
        classes_index = dict((c, i) for i, c in enumerate(self.classes))
        Y = np.array([self.codebook[classes_index[y[i]]]
                      for i in range(x.shape[0])], dtype=np.int)
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
        Applying multiple estimators for prediction
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
