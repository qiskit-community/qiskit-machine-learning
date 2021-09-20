from functools import partial
from typing import Iterable, Union, Optional, Callable

import numpy as np

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils.loss_functions import Loss
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.utils import algorithm_globals


class WeightedKernelAlignmentClassification():
    """
    This class computes the weighted kernel alignment loss.
    """

    def evaluate(self,
                free_parameters: Iterable[float],
                kernel: QuantumKernel,
                data: np.ndarray,
                labels: np.ndarray) -> np.ndarray:

        # Bind learnable parameters
        kernel.assign_free_parameters(free_parameters)

        # Train a quantum support vector classifier
        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(data, labels)

        # Get dual coefficients
        ay = qsvc.dual_coef_[0]

        # Get support vectors
        sv = qsvc.support_

        # Get estimated kernel matrix
        K = kernel.evaluate(np.array(data))[sv,:][:,sv]

        # Calculate loss
        loss = np.sum(np.abs(ay)) - (.5 * (ay.T @ K @ ay))

        return loss


class QuantumKernelTrainer:
    def __init__(
        self,
        kernel: QuantumKernel,
        loss: Union[str, Loss] = "weighted_alignment",
        optimizer: Optional[Optimizer] = None,
        initial_point: np.ndarray = None
    ):
        """
        Args:
            loss: A target loss function to be used in training. Default is `weighted_alignment`,
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SPSA.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        if isinstance(loss, Loss):
            self._loss = loss
        else:
            loss = loss.lower()
            if loss == "weighted_alignment":
                self._loss = WeightedKernelAlignmentClassification()
            else:
                raise QiskitMachineLearningError(f"Unknown loss {loss}!")

        if optimizer is None:
            optimizer = SPSA(maxiter=10, callback=self.QKTCallback())
        self._optimizer = optimizer
        self._initial_point = initial_point

    class QKTCallback:
        """Callback wrapper class."""
        def __init__(self) -> None:
            self._data = [[] for i in range(5)]

        def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
            """
            x[0]: number of function evaluations
            x[1]: the parameters
            x[2]: the function value
            x[3]: the stepsize
            x[4]: whether the step was accepted
            """
            self._data[0].append(x0)
            self._data[1].append(x1)
            self._data[2].append(x2)
            self._data[3].append(x3)
            self._data[4].append(x4)

        def get_callback_data(self):
            return self._data

        def clear_callback_data(self):
            self._data = [[] for i in range(5)]

    @property
    def loss(self):
        """Returns the underlying neural network."""
        return self._loss

    @property
    def optimizer(self) -> Optimizer:
        """Returns an optimizer to be used in training."""
        return self._optimizer

    @property
    def initial_point(self) -> np.ndarray:
        """Returns current initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray) -> None:
        """Sets the initial point"""
        self._initial_point = initial_point

    def fit_kernel(self,
                data: np.ndarray,
                labels: np.ndarray,
                ) -> VariationalResult:
        """
        Align the quantum kernel by minimizing loss over the kernel parameters.

        Args:
            data (numpy.ndarray): NxD array of training data, where N is the
                              number of samples and D is the feature dimension
            labels (numpy.ndarray): Nx1 array of +/-1 labels of the N training samples

        Returns:
            dict: the results of kernel alignment
        """
        # Bind inputs to objective function
        obj_func = partial(WeightedKernelAlignmentClassification().evaluate,
                            self.kernel,
                            data=data,
                            labels=labels)

        if self.initial_point:
            self.kernel.assign_free_parameters(self.initial_point)
        else:
            num_params = len(self.kernel.free_parameters)
            self.kernel.assign_free_parameters(
                    algorithm_globals.random.random(num_params))

        # Perform kernel alignment
        initial_point = np.random.randint(0, len(self.kernel.feature_map.qubits))
        opt_params, opt_vals, num_optimizer_evals = optimizer.optimize(
                                                                1,
                                                                obj_func,
                                                                initial_point=initial_point)
        # Return kernel alignment results
        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self.kernel.free_parameters, opt_params))

        return result, callback
