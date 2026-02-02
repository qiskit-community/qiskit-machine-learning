# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Kernel Loss utilities"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from sklearn.svm import SVC, SVR

# Prevent circular dependencies caused from type checking
from ...kernels import TrainableKernel


class KernelLoss(ABC):
    """
    Abstract base class for computing the loss of a kernel function.
    Unlike many loss functions, which only take into account the labels and predictions
    of a model, kernel loss functions may be a function of internal model parameters or
    quantities that are generated during training.
    """

    def __call__(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        This method calls the ``evaluate`` method. This is a convenient method to compute loss.
        """
        return self.evaluate(parameter_values, quantum_kernel, data, labels)

    @abstractmethod
    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        An abstract method for evaluating the loss of a kernel function on a labeled dataset.

        Args:
            parameter_values: An array of values to assign to the user params
            quantum_kernel: A trainable quantum kernel object to evaluate
            data: An ``(N, M)`` matrix containing the data
                    ``N = # samples, M = dimension of data``
            labels: A length-N array containing the truth labels

        Returns:
            A loss value
        """
        raise NotImplementedError


class SVCLoss(KernelLoss):
    r"""
    This class provides a kernel loss function for classification tasks by fitting an ``SVC`` model
    from scikit-learn. Given training samples, :math:`x_{i}`, with binary labels, :math:`y_{i}`,
    and a kernel, :math:`K_{θ}`, parameterized by values, :math:`θ`, the loss is defined as:

    .. math::

        SVCLoss = \sum_{i} a_i - 0.5 \sum_{i,j} a_i a_j y_{i} y_{j} K_θ(x_i, x_j)

    where :math:`a_i` are the optimal Lagrange multipliers found by solving the standard SVM
    quadratic program. Note that the hyper-parameter ``C`` for the soft-margin penalty can be
    specified through the keyword args.

    Minimizing this loss over the parameters, :math:`θ`, of the kernel is equivalent to maximizing a
    weighted kernel alignment, which in turn yields the smallest upper bound to the SVM
    generalization error for a given parameterization.

    See https://arxiv.org/abs/2105.03406 for further details.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor within
                      SVCLoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)

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


class SVRLoss(KernelLoss):
    r"""
    This class provides a kernel loss function for regression tasks by fitting an ``SVR`` model
    from scikit-learn. Given training samples, :math:`x_{i}`, with labels, :math:`y_{i}`,
    and a kernel, :math:`K_{θ}`, parameterized by values, :math:`θ`, the loss is defined as:

    .. math::

        SVRLoss = -0.5 \sum_{i,j} \beta_i \beta_j K_θ(x_i, x_j)
                  - \epsilon \sum_{i} |\beta_i| + \sum_{i} y_i \beta_i

    where :math:`\beta_i = \alpha_i - \alpha_i^*` are the optimal Lagrange multipliers found by
    solving the standard SVR quadratic program. Note that the hyper-parameters ``C`` and
    ``epsilon`` can be specified through the keyword args.

    Minimizing this loss over the parameters, :math:`θ`, of the kernel is equivalent to minimizing
    the optimized dual objective of the SVR, which is a proxy for the primal objective
    (a combination of the model complexity and the training error).

    See https://arxiv.org/abs/2105.03406 for further details on kernel training (though it focuses
    on classification, the principle applies to regression).
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVR constructor within
                      SVRLoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))

        # Train a quantum support vector regressor
        svr = SVR(kernel="precomputed", **self.kwargs)
        svr.fit(kmatrix, labels)

        # Get dual coefficients (alpha_i - alpha_i^*)
        dual_coefs = svr.dual_coef_[0]

        # Get support vectors
        support_vecs = svr.support_

        # Get epsilon
        epsilon = svr.epsilon

        # Prune kernel matrix of non-support-vector entries
        kmatrix_support = kmatrix[support_vecs, :][:, support_vecs]

        # Calculate loss (dual objective)
        # L = -0.5 * beta^T * K * beta - epsilon * sum|beta| + y^T * beta
        loss = (
            -0.5 * (dual_coefs.T @ kmatrix_support @ dual_coefs)
            - epsilon * np.sum(np.abs(dual_coefs))
            + (labels[support_vecs].T @ dual_coefs)
        )

        return loss


class MSRLoss(KernelLoss):
    """
    This class provides a simple mean squared regression loss function by fitting an ``SVR`` model
    from scikit-learn and computing the mean squared error on the training set.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVR constructor within
                      MSRLoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))

        # Train a quantum support vector regressor
        svr = SVR(kernel="precomputed", **self.kwargs)
        svr.fit(kmatrix, labels)

        # Predict on training data
        predictions = svr.predict(kmatrix)

        # Calculate mean squared error
        loss = np.mean(np.square(predictions - labels))

        return loss


class MARLoss(KernelLoss):
    """
    This class provides a mean absolute regression loss function by fitting an ``SVR`` model
    from scikit-learn and computing the mean absolute error on the training set.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVR constructor within
                      MARLoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))

        # Train a quantum support vector regressor
        svr = SVR(kernel="precomputed", **self.kwargs)
        svr.fit(kmatrix, labels)

        # Predict on training data
        predictions = svr.predict(kmatrix)

        # Calculate mean absolute error
        loss = np.mean(np.abs(predictions - labels))

        return loss


class HuberLoss(KernelLoss):
    """
    This class provides a Huber loss function for regression. It is robust to outliers by
    using a combination of squared error for small errors and absolute error for large errors.
    """

    def __init__(self, delta: float = 1.0, **kwargs):
        """
        Args:
            delta: The threshold at which to change from squared to linear loss.
            **kwargs: Arbitrary keyword arguments to pass to SVR constructor.
        """
        self.delta = delta
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))

        # Train a quantum support vector regressor
        svr = SVR(kernel="precomputed", **self.kwargs)
        svr.fit(kmatrix, labels)

        # Predict on training data
        predictions = svr.predict(kmatrix)

        # Calculate Huber loss
        error = predictions - labels
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = np.mean(0.5 * quadratic**2 + self.delta * linear)

        return loss
