# estimator.py
from __future__ import annotations
from typing import Iterable, Any

from qiskit.primitives import (
    BaseEstimatorV2,
    StatevectorEstimator,
)
from qiskit.transpiler import PassManager


class QML_Estimator(BaseEstimatorV2):
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
