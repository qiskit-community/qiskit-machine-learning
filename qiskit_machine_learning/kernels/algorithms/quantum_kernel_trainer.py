# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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
from functools import partial
from typing import Union, Optional, Sequence

import numpy as np

from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit_machine_learning.utils.loss_functions import KernelLoss, SVCLoss

from qiskit_machine_learning.kernels import TrainableKernel


class QuantumKernelTrainerResult(VariationalResult):
    """Quantum Kernel Trainer Result."""

    def __init__(self) -> None:
        super().__init__()
        self._quantum_kernel: TrainableKernel = None

    @property
    def quantum_kernel(self) -> Optional[TrainableKernel]:
        """Return the optimized quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: TrainableKernel) -> None:
        self._quantum_kernel = quantum_kernel


class QuantumKernelTrainer:
    """
    Quantum Kernel Trainer.
    This class provides utility to train ``QuantumKernel`` feature map parameters.

    **Example**

    .. code-block::

        # Create 2-qubit feature map
        qc = QuantumCircuit(2)

        # Vectors of input and trainable user parameters
        input_params = ParameterVector("x_par", 2)
        training_params = ParameterVector("θ_par", 2)

        # Create an initial rotation layer of trainable parameters
        for i, param in enumerate(training_params):
            qc.ry(param, qc.qubits[i])

        # Create a rotation layer of input parameters
        for i, param in enumerate(input_params):
            qc.rz(param, qc.qubits[i])

        quant_kernel = TrainableFidelityQuantumKernel(
            feature_map=qc,
            training_parameters=training_params,
        )

        loss_func = ...
        optimizer = ...
        initial_point = ...

        qk_trainer = QuantumKernelTrainer(
                                        quantum_kernel=quant_kernel,
                                        loss=loss_func,
                                        optimizer=optimizer,
                                        initial_point=initial_point,
                                        )
        qkt_results = qk_trainer.fit(X_train, y_train)
        optimized_kernel = qkt_results.quantum_kernel
    """

    def __init__(
        self,
        quantum_kernel: TrainableKernel,
        loss: Optional[Union[str, KernelLoss]] = None,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            quantum_kernel: QuantumKernel to be trained
            loss (str or KernelLoss): Loss functions available via string:
                       {'svc_loss': SVCLoss()}.
                       If a string is passed as the loss function, then the
                       underlying KernelLoss object will exhibit default
                       behavior.
            optimizer: An instance of ``Optimizer`` to be used in training. Since no
                       analytical gradient is defined for kernel loss functions, gradient-based
                       optimizers are not recommended for training kernels.
            initial_point: Initial point from which the optimizer will begin.

        Raises:
            ValueError: unknown loss function
        """
        # Class fields
        self._quantum_kernel = quantum_kernel
        self._initial_point = initial_point
        self._optimizer = optimizer or SPSA()

        # Loss setter
        self._set_loss(loss)

    @property
    def quantum_kernel(self) -> TrainableKernel:
        """Return the quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: TrainableKernel) -> None:
        """Set the quantum kernel."""
        self._quantum_kernel = quantum_kernel

    @property
    def loss(self) -> KernelLoss:
        """Return the loss object."""
        return self._loss

    @loss.setter
    def loss(self, loss: Optional[Union[str, KernelLoss]]) -> None:
        """
        Set the loss.

        Args:
            loss: a loss function to set

        Raises:
            ValueError: Unknown loss function
        """
        self._set_loss(loss)

    @property
    def optimizer(self) -> Optimizer:
        """Return an optimizer to be used in training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[Sequence[float]]:
        """Return initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Sequence[float]]) -> None:
        """Set the initial point"""
        self._initial_point = initial_point

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> QuantumKernelTrainerResult:
        """
        Train the QuantumKernel by minimizing loss over the kernel parameters. The input
        quantum kernel will not be altered, and an optimized quantum kernel will be returned.

        Args:
            data (numpy.ndarray): ``(N, D)`` array of training data, where ``N`` is the
                              number of samples and ``D`` is the feature dimension
            labels (numpy.ndarray): ``(N, 1)`` array of target values for the training samples

        Returns:
            QuantumKernelTrainerResult: the results of kernel training

        Raises:
            ValueError: No trainable user parameters specified in quantum kernel
        """
        # Number of parameters to tune
        num_params = len(self._quantum_kernel.training_parameters)
        if num_params == 0:
            msg = "Quantum kernel cannot be fit because there are no user parameters specified."
            raise ValueError(msg)

        # Bind inputs to objective function
        output_kernel = copy.deepcopy(self._quantum_kernel)

        # Randomly initialize the initial point if one was not passed
        if self._initial_point is None:
            self._initial_point = algorithm_globals.random.random(num_params)

        # Perform kernel optimization
        loss_function = partial(
            self._loss.evaluate, quantum_kernel=self.quantum_kernel, data=data, labels=labels
        )
        opt_results = self._optimizer.minimize(
            fun=loss_function,
            x0=self._initial_point,
        )

        # Return kernel training results
        result = QuantumKernelTrainerResult()
        result.optimizer_evals = opt_results.nfev
        result.optimal_value = opt_results.fun
        result.optimal_point = opt_results.x
        result.optimal_parameters = dict(zip(output_kernel.training_parameters, opt_results.x))

        # Return the QuantumKernel in optimized state
        output_kernel.assign_training_parameters(result.optimal_parameters)
        result.quantum_kernel = output_kernel

        return result

    def _set_loss(self, loss: Optional[Union[str, KernelLoss]]) -> None:
        """Internal setter."""
        if loss is None:
            loss = SVCLoss()
        elif isinstance(loss, str):
            loss = self._str_to_loss(loss)

        self._loss = loss

    def _str_to_loss(self, loss_str: str) -> KernelLoss:
        """Function which maps strings to default KernelLoss objects."""
        if loss_str == "svc_loss":
            loss_obj = SVCLoss()
        else:
            raise ValueError(f"Unknown loss {loss_str}!")

        return loss_obj
