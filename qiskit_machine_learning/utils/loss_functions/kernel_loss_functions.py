# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2024, 2026.
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
from typing import Sequence, Optional

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


class BatchedSubKernelSVCLoss(SVCLoss):
    r"""
    This class evaluates an average SVC loss over batches of randomly sampled sub-kernels.

    Each evaluation samples ``batch_size`` number of sub-kernels with each sub-kernel containing
    ``sub_kernel_size`` number of data points. Each sub-kernel aims to preserve the same class
    ratio as the full dataset while still ensuring each class is represented within a kernel at
    all times. Data points are sampled without replacement within each class group separately and
    once all data points have been sampled within an individual group then that individual group
    is reset.

    See https://arxiv.org/abs/2401.02879 for further details.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        sub_kernel_size: Optional[int] = None,
        batch_size: Optional[int] = 1,
        encoder=None,
        **kwargs,
    ):
        """
        Args:
            data (np.ndarray): The data to evaluate the loss on.
            labels (np.ndarray): The corresponding labels for the data.
            sub_kernel_size (int, optional): The size of the sub-kernel batches to split the data into.
                If not provided, the entire data set is used in a single batch.
            batch_size (int, optional): The number of sub-kernels per evaluation.
            encoder (torch.nn): An instance to optionally reduce dimension before calculating loss
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor within
                      SVCLoss evaluation.
        """
        super().__init__(**kwargs)
        # Split data into batches
        self.data = data
        self.labels = labels
        self.sub_kernel_size = sub_kernel_size
        self.batch_size = batch_size
        self.encoder = encoder
        self.loss_arr: list[float] = []
        self.data_idxs = list(range(len(data)))

        self.unique_labels, self.label_counts = np.unique(labels, return_counts=True)
        self.class_idxs = {
            label: np.flatnonzero(self.labels == label) for label in self.unique_labels
        }

        self.unused_idxs = {
            label: np.random.permutation(idxs).tolist() for label, idxs in self.class_idxs.items()
        }

        label_freqs = self.label_counts / np.sum(self.label_counts)

        if sub_kernel_size is not None:
            # Ensure each class is represented by at least 1 sample for SVC to work
            class1_samples = round(sub_kernel_size * label_freqs[0])
            clipped1_samples = np.clip(class1_samples, 1, sub_kernel_size - 1)
            self.sample_counts = np.array([clipped1_samples, sub_kernel_size - clipped1_samples])

    def _batch_subkernels(self):

        subkernels = []

        for _ in range(self.batch_size):
            subkernel_idxs = []

            for label, num_samples in zip(self.unique_labels, self.sample_counts):
                for _ in range(num_samples):
                    if not self.unused_idxs[label]:
                        self.unused_idxs[label] = np.random.permutation(
                            self.class_idxs[label]
                        ).tolist()

                    available_idxs = [
                        idx for idx in self.unused_idxs[label] if idx not in subkernel_idxs
                    ]  # prevents sampling the same point twice in the same kernel

                    idx = np.random.choice(available_idxs)
                    subkernel_idxs.append(idx)
                    self.unused_idxs[label].remove(idx)

            subkernel_idxs = np.array(subkernel_idxs)
            subkernels.append((self.data[subkernel_idxs], self.labels[subkernel_idxs]))

        return subkernels

    def evaluate(
        self,
        parameters: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Wrapper function for loss evaluation with batches of sun kernels.
        If sub_kernel_size is None, it will execute SVCLoss() on full dataset.

        Args:
            parameters (Sequence[float]): The parameter values to evaluate the loss with.
            quantum_kernel (TrainableKernel): The quantum kernel to use for evaluation.
            data (np.ndarray): The data to evaluate the loss on.
            labels (np.ndarray): The corresponding labels for the data.
        Returns:
            loss (float): the loss value for the given parameters and quantum kernel.
        """
        if self.sub_kernel_size is None:
            if self.encoder is not None:
                weights = parameters[: self.encoder.num_weights]
                variational_params = parameters[self.encoder.num_weights :]
                self.encoder.set_weights(weights)
                encoded_data = self.encoder.encode(data)
                return super().evaluate(variational_params, quantum_kernel, encoded_data, labels)
            else:
                loss = super().evaluate(parameters, quantum_kernel, data, labels)
                self.loss_arr.append(loss)
                return loss

        subkernel_batches = self._batch_subkernels()

        # Evaluate the loss for each batch and accumulate the total loss
        total_loss = 0.0

        for subkernel_data, subkernel_labels in subkernel_batches:
            if self.encoder is not None:
                weights = parameters[: self.encoder.num_weights]
                variational_params = parameters[self.encoder.num_weights :]
                self.encoder.set_weights(weights)
                subkernel_data = self.encoder.encode(subkernel_data)
            else:
                variational_params = parameters
            loss = super().evaluate(
                variational_params, quantum_kernel, subkernel_data, subkernel_labels
            )
            total_loss += loss

        param_loss = total_loss / self.batch_size
        self.loss_arr.append(param_loss)

        return param_loss


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
