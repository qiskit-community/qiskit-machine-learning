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

""" Loss utilities """

from functools import partial
from abc import ABC, abstractmethod
from typing import Sequence, Callable, TYPE_CHECKING

import numpy as np
from sklearn.svm import SVC

from ...exceptions import QiskitMachineLearningError

# Prevent circular dependencies caused from type checking
if TYPE_CHECKING:
    from ...kernels import QuantumKernel
else:
    QuantumKernel = object


class Loss(ABC):
    """
    Abstract base class for computing Loss.
    """

    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        This method calls the ``evaluate`` method. This is a convenient method to compute loss.
        """
        return self.evaluate(predict, target)

    @abstractmethod
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        An abstract method for evaluating the loss function. Inputs are expected in a shape
        of ``(N, *)``. Where ``N`` is a number of samples. Loss is computed for each sample
        individually.

        Args:
            predict: an array of predicted values using the model.
            target: an array of the true values.

        Returns:
            An array with values of the loss function of the shape ``(N, 1)``.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match
        """
        raise NotImplementedError

    @staticmethod
    def _validate_shapes(predict: np.ndarray, target: np.ndarray) -> None:
        """
        Validates that shapes of both parameters are identical.

        Args:
            predict: an array of predicted values using the model
            target: an array of the true values

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """

        if predict.shape != target.shape:
            raise QiskitMachineLearningError(
                f"Shapes don't match, predict: {predict.shape}, target: {target.shape}!"
            )

    @abstractmethod
    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        An abstract method for computing the gradient. Inputs are expected in a shape
        of ``(N, *)``. Where ``N`` is a number of samples. Gradient is computed for each sample
        individually.

        Args:
            predict: an array of predicted values using the model.
            target: an array of the true values.

        Returns:
            An array with gradient values of the shape ``(N, *)``. The output shape depends on
            the loss function.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """
        raise NotImplementedError


class KernelLoss(ABC):
    """
    Abstract base class for computing the loss of a kernel function.
    Unlike many loss functions, which only take into account the labels and predictions
    of a model, kernel loss functions may be a function of internal model parameters or 
    quantities that are generated during training. For this reason, extensions of this
    class may find it neccesary to introduce additional inputs. 
    """

    def __call__(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: QuantumKernel,
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
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        An abstract method for evaluating the loss of a kernel function on a labeled dataset.

        Args:
            parameter_values: an array of values to assign to the user params
            quantum_kernel: A ``QuantumKernel`` object to evaluate
            data: An ``NxM`` matrix containing the data
                    ``N = # samples, M = dimension of data``
            labels: A length-N array containing the truth labels

        Returns:
            A loss value
        """
        raise NotImplementedError

    @abstractmethod
    def get_variational_callable(self) -> Callable[[Sequence[float]], float]:
        """
        Return a callable variational loss function given a quantum kernel and labeled dataset.
        The sole input to the callable will be an array of feature map parameter values, and the
        output will be a loss value.
        """
        raise NotImplementedError


class L1Loss(Loss):
    r"""
    This class computes the L1 loss (i.e. absolute error) for each sample as:

    .. math::

        \text{L1Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} \left| predict_i -
        target_i \right|.
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(predict.shape) <= 1:
            return np.abs(predict - target)
        else:
            return np.linalg.norm(predict - target, ord=1, axis=tuple(range(1, len(predict.shape))))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        return np.sign(predict - target)


class L2Loss(Loss):
    r"""
    This class computes the L2 loss (i.e. squared error) for each sample as:

    .. math::

        \text{L2Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} (predict_i - target_i)^2.

    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(predict.shape) <= 1:
            return (predict - target) ** 2
        else:
            return np.linalg.norm(predict - target, axis=tuple(range(1, len(predict.shape)))) ** 2

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        return 2 * (predict - target)


class CrossEntropyLoss(Loss):
    r"""
    This class computes the cross entropy loss for each sample as:

    .. math::

        \text{CrossEntropyLoss}(predict, target) = -\sum_{i=0}^{N_{\text{classes}}}
        target_i * log(predict_i).
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)

        # multiply target and log(predict) matrices row by row and sum up each row
        # into a single float, so the output is of shape(N,), where N number or samples.
        # then reshape
        val = -np.einsum("ij,ij->i", target, np.log2(predict)).reshape(-1, 1)

        return val

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Assume softmax is used, and target vector may or may not be one-hot encoding"""

        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)

        # sum up target along rows, then multiply predict by this sum element wise,
        # then subtract target
        grad = np.einsum("ij,i->ij", predict, np.sum(target, axis=1)) - target

        return grad


class CrossEntropySigmoidLoss(Loss):
    """
    This class computes the cross entropy sigmoid loss and should be used for binary classification.
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(set(target)) != 2:
            raise QiskitMachineLearningError(
                "Sigmoid Cross Entropy is used for binary classification!"
            )

        x = CrossEntropyLoss()
        return 1.0 / (1.0 + np.exp(-x.evaluate(predict, target)))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)

        return target * (1.0 / (1.0 + np.exp(-predict)) - 1) + (1 - target) * (
            1.0 / (1.0 + np.exp(-predict))
        )


class SVCLoss(KernelLoss):
    """
    This class computes the inverse of the margin obtained after training a SKLearn
    ``SVC`` class on a given quantum kernel and data set. The dual-form of the
    SVC objective, which gives the inverse margin, is equal to the negative
    weighted kernel alignment with an added regularization term.

    SVCLoss = (\sum_i^N a_i)   - (\sum_{i,j}^N a_i•a_j•y_i•y_j•K_ij)
            = (regularization) - (weighted alignment)

    See https://arxiv.org/pdf/2105.03406.pdf for further details.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_user_parameters(parameter_values)

        # Train a quantum support vector classifier
        svc = SVC(kernel=quantum_kernel.evaluate, **self.kwargs)
        svc.fit(data, labels)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (0.5 * (dual_coefs.T @ kmatrix @ dual_coefs))

        return loss

    def get_variational_callable(
        self,
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Callable[[Sequence[float]], float]:

        return partial(self.evaluate, quantum_kernel=quantum_kernel, data=data, labels=labels)
