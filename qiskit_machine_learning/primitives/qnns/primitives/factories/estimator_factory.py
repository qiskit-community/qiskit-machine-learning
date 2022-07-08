from abc import ABC, abstractmethod

from qiskit.primitives import BaseEstimator

class EstimatorFactory(ABC):
    """Class to construct an estimator, given circuits and observables."""

    def __init__(self, circuits=None, observables=None, parameters=None):
        self.circuits = circuits
        self.observables = observables
        self.parameters = parameters

    @abstractmethod
    def __call__(self, circuits=None, observables=None, parameters=None) -> BaseEstimator:
        pass

