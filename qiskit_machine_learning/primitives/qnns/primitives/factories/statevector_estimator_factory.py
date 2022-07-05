from qiskit.primitives import Estimator
from .estimator_factory import EstimatorFactory

class StatevectorEstimatorFactory(EstimatorFactory):
    """Estimator factory evaluated with statevector simulations."""

    def __call__(self, circuits=None, observables=None, parameters=None) -> Estimator:
        circuits = circuits or self.circuits
        observables = observables or self.observables
        parameters = parameters or self.parameters

        return Estimator(circuits, observables, parameters)

