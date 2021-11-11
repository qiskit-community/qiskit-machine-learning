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

"""Quantum Kernel Trainer"""
import copy
from typing import Union, Optional, Sequence, Callable, TYPE_CHECKING

import numpy as np

from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit_machine_learning.utils.loss_functions import SVCLoss

# Prevent circular dependencies from type checking
if TYPE_CHECKING:
    from qiskit_machine_learning.kernels import QuantumKernel
else:
    QuantumKernel = object


class QuantumKernelTrainerResult(VariationalResult):
    """Quantum Kernel Trainer Result."""

    def __init__(self) -> None:
        super().__init__()
        self._quantum_kernel = None  # type: QuantumKernel

    @property
    def quantum_kernel(self) -> Optional[QuantumKernel]:
        """Returns the optimized quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        self._quantum_kernel = quantum_kernel


class QuantumKernelTrainer:
    """
    Quantum Kernel Trainer.
    This class provides utility to train ``QuantumKernel`` feature map parameters.

    **Example**

    .. code-block::

        quant_kernel = QuantumKernel(
            feature_map=...,
            user_parameters=...,
            quantum_instance=...
        )

        loss_func = ...
        optimizer = ...
        initial_point = ...

        qk_trainer = QuantumKernelTrainer(
                                        loss=loss_func,
                                        optimizer=optimizer,
                                        initial_point=initial_point,
                                        )

        qsvc = QSVC(quantum_kernel=quant_kernel, kernel_trainer=qk_trainer)
        qsvc.fit(X_train, y_train)
        score = qsvc.score(X_test, y_test)
    """

    def __init__(
        self,
        quantum_kernel: QuantumKernel,
        loss: Union[str, Callable[[Sequence[float]], float]] = "svc_alignment",
        optimizer: Optimizer = SPSA(),
        initial_point: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            quantum_kernel: QuantumKernel to be trained
            loss (Callable[[Sequence[float]], float] or str):
                str: Loss functions available via string: {'svc_alignment: SVCLoss()).
                    If a string is passed as the loss function, then the underlying
                    KernelLoss object will exhibit default behavior.
                Callable[[Sequence[float]], float]: A callable loss function which takes
                    a vector of parameter values as input and returns a loss value (float)
            optimizer: An instance of ``Optimizer`` to be used in training. Since no
                analytical gradient is defined for kernel loss functions, gradient-based
                optimizers are not recommended for training kernels.
            initial_point: Initial point from which the optimizer will begin.

        Raises:
            ValueError: unknown loss function
        """
        # Class fields
        self._quantum_kernel = quantum_kernel
        self._optimizer = optimizer
        self._initial_point = initial_point
        self._loss = None

        # Setters
        self.loss = loss

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns the quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        """Sets the quantum kernel."""
        self._quantum_kernel = quantum_kernel

    @property
    def loss(self) -> Union[str, Callable[[Sequence[float]], float]]:
        """Returns the loss object."""
        return self._loss

    @loss.setter
    def loss(self, loss: Union[str, Callable[[Sequence[float]], float]]) -> None:
        """Sets the loss."""
        if isinstance(loss, str):
            self._loss = loss.lower()
            if self._loss == "svc_alignment":
                pass
            else:
                raise ValueError(f"Unknown loss {loss}!")
        elif callable(loss):
            self._loss = loss  # type: ignore
        else:
            raise ValueError(f"Unknown loss {loss}!")

    @property
    def optimizer(self) -> Optimizer:
        """Returns an optimizer to be used in training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Sets the loss."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[Sequence[float]]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Sequence[float]]) -> None:
        """Sets the initial point"""
        self._initial_point = initial_point

    def fit_kernel(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> QuantumKernelTrainerResult:
        """
        Train the QuantumKernel by minimizing loss over the kernel parameters. The input
        quantum kernel will not be altered, and an optimized quantum kernel will be returned.

        Args:
            data (numpy.ndarray): ``NxD`` array of training data, where ``N`` is the
                              number of samples and ``D`` is the feature dimension
            labels (numpy.ndarray): ``Nx1`` array of +/-1 labels of the ``N`` training samples

        Returns:
            QuantumKernelTrainerResult: the results of kernel training

        Raises:
            ValueError: No trainable user parameters specified in quantum kernel
        """
        # Number of parameters to tune
        num_params = len(self._quantum_kernel.user_parameters)
        if num_params == 0:
            msg = "Quantum kernel cannot be fit because there are no user parameters specified."
            raise ValueError(msg)

        # Bind inputs to objective function
        output_kernel = copy.deepcopy(self._quantum_kernel)
        if isinstance(self._loss, str):
            obj_func = _str_to_variational_callable(
                loss_str=self._loss, quantum_kernel=output_kernel, data=data, labels=labels
            )
            self._loss = obj_func

        # Randomly initialize the initial point if one was not passed
        if self.initial_point is None:
            self.initial_point = algorithm_globals.random.random(num_params)

        # Perform kernel optimization
        opt_results = self._optimizer.minimize(
            fun=self._loss,
            x0=self._initial_point,
        )

        # Return kernel training results
        result = QuantumKernelTrainerResult()
        result.optimizer_evals = opt_results.nfev
        result.optimal_value = opt_results.fun
        result.optimal_point = opt_results.x
        result.optimal_parameters = dict(zip(output_kernel.user_parameters, opt_results.x))

        # Return the QuantumKernel in optimized state
        output_kernel.assign_user_parameters(result.optimal_parameters)
        result.quantum_kernel = output_kernel

        return result


def _str_to_variational_callable(
    loss_str: str,
    quantum_kernel: QuantumKernel = None,
    data: np.ndarray = None,
    labels: np.ndarray = None,
) -> Callable[[Sequence[float]], float]:
    if loss_str == "svc_alignment":
        loss_obj = SVCLoss()
        return loss_obj.get_variational_callable(
            quantum_kernel=quantum_kernel, data=data, labels=labels
        )
    else:
        raise ValueError(f"Unknown loss {loss_str}!")
