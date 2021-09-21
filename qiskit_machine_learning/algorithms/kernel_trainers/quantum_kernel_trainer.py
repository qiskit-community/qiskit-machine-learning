from functools import partial
from typing import Iterable, Union, Optional, Callable

import numpy as np

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.utils.algorithm_globals import QiskitAlgorithmGlobals
from qiskit_machine_learning.utils.loss_functions import KernelLoss, WeightedKernelAlignmentClassification


class QuantumKernelTrainer:
    def __init__(
        self,
        kernel: QuantumKernel,
        loss: Optional[Union[str, Loss]] = "weighted_alignment",
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[np.ndarray] = None,
    ):
        """
        Args:
            loss: A target loss function to be used in training. Default is `weighted_alignment`,
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SPSA.
            initial_point: Initial point for the optimizer to start from.

        Raises:
            QiskitMachineLearningError: unknown loss function
        """
        loss = loss.lower()
        if loss == "weighted_alignment":
            self.loss = WeightedKernelAlignmentClassification()
        else:
            raise QiskitMachineLearningError(f"Unknown loss {loss}!")

        if optimizer is None:
            optimizer = SPSA(maxiter=10, callback=[None] * 5)
        self.optimizer = optimizer
        self.initial_point = initial_point

    @property
    def loss(self):
        """Returns the underlying neural network."""
        return self._loss

    @loss.setter
    def loss(self, loss: KernelLoss) -> None:
        """Sets the loss."""
        self._loss = loss

    @property
    def optimizer(self) -> Optimizer:
        """Returns an optimizer to be used in training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Sets the loss."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> np.ndarray:
        """Returns current initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray) -> None:
        """Sets the initial point"""
        self._initial_point = initial_point

    def fit_kernel(
        self,
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
        obj_func = partial(self.loss.evaluate, self.kernel, data=data, labels=labels)

        # Randomly initialize our free parameters if no initial point was passed
        if not self.initial_point:
            num_params = len(self.kernel.free_parameters)
            algo_globals = QiskitAlgorithmGlobals()
            algo_globals.random_seed = 9195
            self.initial_point = algo_globals.random.random(num_params)

        self.kernel.assign_free_parameters(self.initial_point)

        # Perform kernel alignment
        opt_params, opt_vals, num_optimizer_evals = optimizer.optimize(
            1, obj_func, initial_point=self.initial_point
        )
        # Return kernel alignment results
        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self.kernel.free_parameters, opt_params))

        return result, callback
