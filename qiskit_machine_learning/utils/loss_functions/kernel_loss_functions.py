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

""" Kernel Loss utilities """

from functools import partial
from abc import ABC, abstractmethod
from typing import Sequence, Iterable, Callable

import numpy as np
from sklearn.svm import SVC

# Prevent circular dependencies caused from type checking
from ...kernels import QuantumKernel


class KernelLoss(ABC):
    """
    Abstract base class for computing the loss of a kernel function.
    Unlike many loss functions, which only take into account the labels and predictions
    of a model, kernel loss functions may be a function of internal model parameters or
    quantities that are generated during training. For this reason, extensions of this
    class may find it necessary to introduce additional inputs.
    """

    def __call__(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: "QuantumKernel",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        This method calls the ``evaluate`` method. This is a convenient method to compute loss.
        """
        return self.evaluate(parameter_values, quantum_kernel, data, labels)

    def get_variational_callable(
        self, quantum_kernel: QuantumKernel, data: Iterable, labels: Iterable
    ) -> Callable[[Sequence[float]], float]:
        """
        Return a callable variational loss function given some inputs. The sole input to the
        callable should be an array of numerical feature map parameter values, and the output should
        be a numerical loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: "QuantumKernel",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        An abstract method for evaluating the loss of a kernel function on a labeled dataset.

        Args:
            parameter_values: an array of values to assign to the user params
            quantum_kernel: A ``QuantumKernel`` object to evaluate
            data: An ``(N, M)`` matrix containing the data
                    ``N = # samples, M = dimension of data``
            labels: A length-N array containing the truth labels

        Returns:
            A loss value
        """
        raise NotImplementedError


class SVCLoss(KernelLoss):
    r"""
    .. math::

        \text{This class provides a kernel loss function for classification tasks by
        fitting an ``SVC`` model from sklearn. Given training samples, x_{i}, with binary
        labels, y_{i}, and a kernel, K_{θ}, parameterized by values, θ, the loss is defined as:}

        SVCLoss = \sum_{i} a_i - 0.5 \sum_{i,j} a_i a_j y_{i} y_{j} K_θ(x_i, x_j)

        \text{where a_i are the optimal Lagrange multipliers found by solving the standard
        SVM quadratic program. Note that the hyper-parameter C for the soft-margin penalty can
        be specified through the keyword args.}

    Minimizing this loss over the parameters, θ, of the kernel is equivalent to maximizing a
    weighted kernel alignment, which in turn yields the smallest upper bound to the SVM
    generalization error for a given parametrization.

    See https://arxiv.org/abs/2105.03406 for further details.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_variational_callable(
        self, quantum_kernel: QuantumKernel, data: Iterable, labels: Iterable
    ) -> Callable[[Sequence[float]], float]:
        return partial(
            self.evaluate,
            quantum_kernel=quantum_kernel,
            data=data,
            labels=labels,
        )

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: "QuantumKernel",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_user_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))

        # Train a quantum support vector classifier
        svc = SVC(kernel="precomputed", **self.kwargs)
        svc.fit(kmatrix, labels)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Prune kernel matrix of non-support-vector entries
        kmatrix = kmatrix[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (0.5 * (dual_coefs.T @ kmatrix @ dual_coefs))

        return loss
