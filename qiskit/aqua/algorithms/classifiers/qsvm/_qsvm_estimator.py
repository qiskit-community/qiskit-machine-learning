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

"""The estimator that uses the quantum kernel."""

from qiskit.aqua.components.multiclass_extensions import Estimator
from ._qsvm_binary import _QSVM_Binary

# pylint: disable=invalid-name


class _QSVM_Estimator(Estimator):
    """The estimator that uses the quantum kernel."""

    def __init__(self, feature_map, qalgo):  # pylint: disable=unused-argument
        super().__init__()
        self._qsvm_binary = _QSVM_Binary(qalgo)
        self._ret = {}

    def fit(self, x, y):
        """
        Fit values for the points and the labels.

        Args:
            x (numpy.ndarray): input points, NxD array
            y (numpy.ndarray): input labels, Nx1 array
        """
        self._qsvm_binary.train(x, y)
        self._ret = self._qsvm_binary._ret

    def decision_function(self, x):
        """
        Predicted values for the points which account for both the labels and the confidence.

        Args:
            x (numpy.ndarray): NxD array
        Returns:
            numpy.ndarray: predicted confidence, Nx1 array
        """
        confidence = self._qsvm_binary.get_predicted_confidence(x)
        return confidence

    @property
    def ret(self):
        """ returns result """
        return self._ret
