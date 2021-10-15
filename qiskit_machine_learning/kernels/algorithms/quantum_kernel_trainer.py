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

from functools import partial
from typing import Union, Optional, Sequence

import numpy as np

from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils.loss_functions import KernelLoss, SVCAlignment


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

        qsvc = QSVC(quantum_kernel=quant_kernel,
                quant_kernel_trainer=qk_trainer)
        qsvc.fit(X_train, y_train)
        score = qsvc.score(X_test, y_test)
    """

    def __init__(
        self,
        loss: Union[str, KernelLoss] = "svc_alignment",
        optimizer: Optimizer = SPSA(),
        initial_point: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            loss: A target loss function to be used in training. Default is `svc_alignment`.
            optimizer: An instance of ``Optimizer`` to be used in training. Defaults to
                ``SPSA``.
            initial_point: Initial point for the optimizer to start from.

        Raises:
            ValueError: unknown loss function
        """
        # Class fields
        self.loss = loss
        self.optimizer = optimizer
        self.initial_point = initial_point

    @property
    def loss(self) -> Union[str, KernelLoss]:
        """Returns the loss object."""
        return self._loss

    @loss.setter
    def loss(self, loss: Union[str, KernelLoss]) -> None:
        """Sets the loss."""
        if isinstance(loss, str):
            loss = loss.lower()
            if loss == "svc_alignment":
                self._loss = SVCAlignment()
            else:
                raise ValueError(f"Unknown loss {loss}!")
        elif isinstance(loss, KernelLoss):
            self._loss = loss # type: ignore
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
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> VariationalResult:
        """
        Train the QuantumKernel by minimizing loss over the kernel parameters.

        Args:
            quantum_kernel (QuantumKernel): ``QuantumKernel`` object to be optimized
            data (numpy.ndarray): ``NxD`` array of training data, where ``N`` is the
                              number of samples and ``D`` is the feature dimension
            labels (numpy.ndarray): ``Nx1`` array of +/-1 labels of the ``N`` training samples

        Returns:
            dict: the results of kernel training
        """
        # Bind inputs to objective function
        obj_func = partial(self._loss.evaluate, kernel=quantum_kernel, data=data, labels=labels)

        # Number of parameters to tune
        num_params = len(quantum_kernel.user_parameters)

        # Randomly initialize the initial point if one was not passed
        if self.initial_point is None:
            self.initial_point = algorithm_globals.random.random(num_params)

        # Perform kernel optimization
        opt_results = self._optimizer.minimize(fun=obj_func, x0=self._initial_point)

        # Return kernel training results
        result = VariationalResult()
        result.optimizer_evals = opt_results.nfev
        result.optimal_value = opt_results.fun
        result.optimal_point = opt_results.x
        result.optimal_parameters = dict(zip(quantum_kernel.user_parameters, opt_results.x))

        # Ensure QuantumKernel is left in optimized state
        quantum_kernel.assign_user_parameters(result.optimal_parameters)

        return result
