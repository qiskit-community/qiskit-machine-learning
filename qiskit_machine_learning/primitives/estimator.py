# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Qiskit Machine Learning Estimator"""
from __future__ import annotations
from typing import Iterable, Any

from qiskit.primitives import (
    BaseEstimatorV2,
    StatevectorEstimator,
)
from qiskit.transpiler import PassManager


class QMLEstimator(BaseEstimatorV2):
    """Simple EstimatorV2 wrapper that just delegates to a provided estimator.
    This file exists to keep the algorithm structure stable.
    """

    def __init__(self, estimator: BaseEstimatorV2, pass_manager: PassManager | None = None):
        """
        Constructor
        """
        if estimator is None:
            estimator = StatevectorEstimator()
        self._inner = estimator
        self.pass_manager = pass_manager  # stored for algorithms to use if they choose

    def run(self, pubs: Iterable[Any], *, precision: float | None = None):
        # Delegate; if you need to apply a stored pass manager, do it in your pipeline.
        return self._inner.run(pubs, precision=precision)
