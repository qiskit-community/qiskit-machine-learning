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
from typing import Union, Optional

import numpy as np

from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils.loss_functions import KernelLoss, SVCAlignment


class QuantumKernelTrainer:
    """
    Quantum Kernel Trainer.
    This class provides utility to train QuantumKernel feature map parameters.

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
        loss: Optional[Union[str, KernelLoss]] = "svc_alignment",
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[np.ndarray] = None,
    ):
        """
        Args:
            loss: A target loss function to be used in training. Default is `svc_alignment`.
            optimizer: An instance of an optimizer to be used in training. Defaults to
                gradient descent.
            initial_point: Initial point for the optimizer to start from.

        Raises:
            ValueError: unknown loss function
        """
        # Class fields
        self._loss = None
        self._optimizer = None
        self._initial_point = None

        # Setters
        self.initial_point = initial_point
        self.loss = loss if loss else "svc_alignment"
        self.optimizer = optimizer if optimizer else SPSA()

    @property
    def loss(self):
        """Returns the underlying loss function."""
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
        elif callable(loss):
            self._loss = loss
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
    def initial_point(self) -> np.ndarray:
        """Returns current initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray) -> None:
        """Sets the initial point"""
        self._initial_point = initial_point

    def fit_kernel(
        self,
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> VariationalResult:
        """
        Train the quantum kernel by minimizing loss over the kernel parameters.

        Args:
            quantum_kernel (QuantumKernel): QuantumKernel object to be optimized
            data (numpy.ndarray): NxD array of training data, where N is the
                              number of samples and D is the feature dimension
            labels (numpy.ndarray): N x 1 array of +/-1 labels of the N training samples

        Returns:
            dict: the results of kernel training
        """
        # Bind inputs to objective function
        obj_func = partial(self.loss.evaluate, kernel=quantum_kernel, data=data, labels=labels)

        # Number of parameters to tune
        num_params = len(quantum_kernel.user_parameters)

        # Randomly initialize our user parameters if no initial point was passed
        if not self.initial_point:
            self.initial_point = algorithm_globals.random.random(num_params)

        # Perform kernel optimization
        result = self.optimizer.minimize(fun=obj_func, x0=self.initial_point)
        opt_params = result.x
        opt_vals = result.fun
        num_optimizer_evals = result.nfev

        # Return kernel training results
        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimal_value = opt_vals
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(quantum_kernel.user_parameters, opt_params))

        return result
