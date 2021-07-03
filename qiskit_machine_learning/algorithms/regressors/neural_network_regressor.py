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
from ...exceptions import QiskitMachineLearningError


class NeuralNetworkRegressor(TrainableModel, RegressorMixin):
    """Quantum neural network regressor. Implements Scikit-Learn compatible methods for
    regression and extends ``RegressorMixin``. See `Scikit-Learn <https://scikit-learn.org>`__
    for more details.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):  # pylint: disable=invalid-name
        # mypy definition
        function: ObjectiveFunction = None
        if self._neural_network.output_shape == (1,):
            function = BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            function = MultiClassObjectiveFunction(X, y, self._neural_network, self._loss)

        self._fit_result = self._optimizer.optimize(
            self._neural_network.num_weights,
            function.objective,
            function.gradient,
            initial_point=self._choose_initial_point(),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        if self._fit_result is None:
            raise QiskitMachineLearningError("Model needs to be fit to some training data first!")

        return self._neural_network.forward(X, self._fit_result[0])

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        return RegressorMixin.score(self, X, y, sample_weight)
