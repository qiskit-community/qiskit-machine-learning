# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network regressor."""

from typing import Optional

import numpy as np
from sklearn.base import RegressorMixin

from ..objective_functions import (
    BinaryObjectiveFunction,
    MultiClassObjectiveFunction,
    ObjectiveFunction,
)
from ..trainable_model import TrainableModel
from ...optimizers import OptimizerResult


class NeuralNetworkRegressor(TrainableModel, RegressorMixin):
    """Implements a basic quantum neural network regressor. Implements Scikit-Learn compatible
    methods for regression and extends ``RegressorMixin``.
    See `Scikit-Learn <https://scikit-learn.org>`__ for more details.
    """

    def _fit_internal(
        self, X: np.ndarray, y: np.ndarray
    ) -> OptimizerResult:  # pylint: disable=invalid-name
        # mypy definition
        function: ObjectiveFunction = None
        if self._neural_network.output_shape == (1,):
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)

        return self._minimize(function)

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        self._check_fitted()

        return self._neural_network.forward(X, self._fit_result.x)

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        return RegressorMixin.score(self, X, y, sample_weight)
