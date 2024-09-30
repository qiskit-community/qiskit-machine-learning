# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pegasos Quantum Support Vector Classifier."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

import numpy as np
from sklearn.base import ClassifierMixin

from ...algorithms.serializable_model import SerializableModelMixin
from ...exceptions import QiskitMachineLearningError
from ...kernels import BaseKernel, FidelityQuantumKernel
from ...utils import algorithm_globals


logger = logging.getLogger(__name__)


class PegasosQSVC(ClassifierMixin, SerializableModelMixin):
    r"""
    Implements Pegasos Quantum Support Vector Classifier algorithm. The algorithm has been
    developed in [1] and includes methods ``fit``, ``predict`` and ``decision_function`` following
    the signatures
    of `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    This implementation is adapted to work with quantum kernels.

    **Example**

    .. code-block:: python

        quantum_kernel = FidelityQuantumKernel()

        pegasos_qsvc = PegasosQSVC(quantum_kernel=quantum_kernel)
        pegasos_qsvc.fit(sample_train, label_train)
        pegasos_qsvc.predict(sample_test)

    **References**
        [1]: Shalev-Shwartz et al., Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
            `Pegasos for SVM <https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf>`_

    """

    FITTED = 0
    UNFITTED = 1

    # pylint: disable=too-many-positional-arguments
    # pylint: disable=invalid-name
    def __init__(
        self,
        quantum_kernel: BaseKernel | None = None,
        C: float = 1.0,
        num_steps: int = 1000,
        precomputed: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            quantum_kernel: A quantum kernel to be used for classification.
                Has to be ``None`` when a precomputed kernel is used. If None,
                and ``precomputed`` is ``False``, the quantum kernel will default to
                :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`.
            C: Positive regularization parameter. The strength of the regularization is inversely
                proportional to C. Smaller ``C`` induce smaller weights which generally helps
                preventing overfitting. However, due to the nature of this algorithm, some of the
                computation steps become trivial for larger ``C``. Thus, larger ``C`` improve
                the performance of the algorithm drastically. If the data is linearly separable
                in feature space, ``C`` should be chosen to be large. If the separation is not
                perfect, ``C`` should be chosen smaller to prevent overfitting.
            num_steps: The number of steps in the Pegasos algorithm. There is no early stopping
                criterion. The algorithm iterates over all steps.
            precomputed: A boolean flag indicating whether a precomputed kernel is used. Set it to
                ``True`` in case of precomputed kernel.
            seed: A seed for the random number generator.

        Raises:
            ValueError:
                - if ``quantum_kernel`` is passed and ``precomputed`` is set to ``True``. To use
                a precomputed kernel, ``quantum_kernel`` has to be of the ``None`` type.
                - if C is not a positive number.
        """

        if precomputed:
            if quantum_kernel is not None:
                raise ValueError("'quantum_kernel' has to be None to use a precomputed kernel")
        else:
            if quantum_kernel is None:
                quantum_kernel = FidelityQuantumKernel()

        self._quantum_kernel = quantum_kernel
        self._precomputed = precomputed
        self._num_steps = num_steps
        if seed is not None:
            algorithm_globals.random_seed = seed

        if C > 0:
            self.C = C
        else:
            raise ValueError(f"C has to be a positive number, found {C}.")

        # these are the parameters being fit and are needed for prediction
        self._alphas: Dict[int, int] | None = None
        self._x_train: np.ndarray | None = None
        self._n_samples: int | None = None
        self._y_train: np.ndarray | None = None
        self._label_map: Dict[int, int] | None = None
        self._label_pos: int | None = None
        self._label_neg: int | None = None

        # added to all kernel values to include an implicit bias to the hyperplane
        self._kernel_offset = 1

        # for compatibility with the base SVC class. Set as unfitted.
        self.fit_status_ = PegasosQSVC.UNFITTED

    # pylint: disable=invalid-name
    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "PegasosQSVC":
        """Fit the model according to the given training data.

        Args:
            X: Train features. For a callable kernel (an instance of
               :class:`~qiskit_machine_learning.kernels.BaseKernel`) the shape
               should be ``(n_samples, n_features)``, for a precomputed kernel the shape should be
               ``(n_samples, n_samples)``.
            y: shape (n_samples), train labels . Must not contain more than two unique labels.
            sample_weight: this parameter is not supported, passing a value raises an error.

        Returns:
            ``self``, Fitted estimator.

        Raises:
            ValueError:
                - X and/or y have the wrong shape.
                - X and y have incompatible dimensions.
                - y includes more than two unique labels.
                - Pre-computed kernel matrix has the wrong shape and/or dimension.

            NotImplementedError:
                - when a sample_weight which is not None is passed.
        """
        # check whether the data have the right format
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if np.ndim(y) != 1:
            raise ValueError("y has to be a 1D array")
        if len(np.unique(y)) != 2:
            raise ValueError("Only binary classification is supported")
        if X.shape[0] != y.shape[0]:
            raise ValueError("'X' and 'y' have to contain the same number of samples")
        if self._precomputed and X.shape[0] != X.shape[1]:
            raise ValueError(
                "For a precomputed kernel, X should be in shape (n_samples, n_samples)"
            )
        if sample_weight is not None:
            raise NotImplementedError(
                "Parameter 'sample_weight' is not supported. All samples have to be weighed equally"
            )
        # reset the fit state
        self.fit_status_ = PegasosQSVC.UNFITTED

        # the algorithm works with labels in {+1, -1}
        self._label_pos = np.unique(y)[0]
        self._label_neg = np.unique(y)[1]
        self._label_map = {self._label_pos: +1, self._label_neg: -1}

        # the training data are later needed for prediction
        self._x_train = X
        self._y_train = y
        self._n_samples = X.shape[0]

        # empty dictionary to represent sparse array
        self._alphas = {}

        t_0 = datetime.now()
        # training loop
        for step in range(1, self._num_steps + 1):
            # for every step, a random index (determining a random datum) is fixed
            i = algorithm_globals.random.integers(0, len(y))

            value = self._compute_weighted_kernel_sum(i, X, training=True)

            if (self._label_map[y[i]] * self.C / step) * value < 1:
                # only way for a component of alpha to become non zero
                self._alphas[i] = self._alphas.get(i, 0) + 1

        self.fit_status_ = PegasosQSVC.FITTED

        logger.debug("fit completed after %s", str(datetime.now() - t_0)[:-7])

        return self

    # pylint: disable=invalid-name
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Args:
            X: Features. For a callable kernel (an instance of
               :class:`~qiskit_machine_learning.kernels.BaseKernel`) the shape
               should be ``(m_samples, n_features)``, for a precomputed kernel the shape should be
               ``(m_samples, n_samples)``. Where ``m`` denotes the set to be predicted and ``n`` the
               size of the training set. In that case, the kernel values in X have to be calculated
               with respect to the elements of the set to be predicted and the training set.

        Returns:
            An array of the shape (n_samples), the predicted class labels for samples in X.

        Raises:
            QiskitMachineLearningError:
                - predict is called before the model has been fit.
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension.
        """

        t_0 = datetime.now()
        values = self.decision_function(X)
        y = np.array([self._label_pos if val > 0 else self._label_neg for val in values])
        logger.debug("prediction completed after %s", str(datetime.now() - t_0)[:-7])

        return y

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.

        Args:
            X: Features. For a callable kernel (an instance of
               :class:`~qiskit_machine_learning.kernels.BaseKernel`) the shape
               should be ``(m_samples, n_features)``, for a precomputed kernel the shape should be
               ``(m_samples, n_samples)``. Where ``m`` denotes the set to be predicted and ``n`` the
               size of the training set. In that case, the kernel values in X have to be calculated
               with respect to the elements of the set to be predicted and the training set.

        Returns:
            An array of the shape (n_samples), the decision function of the sample.

        Raises:
            QiskitMachineLearningError:
                - the method is called before the model has been fit.
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension.
        """
        if self.fit_status_ == PegasosQSVC.UNFITTED:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if self._precomputed and self._n_samples != X.shape[1]:
            raise ValueError(
                "For a precomputed kernel, X should be in shape (m_samples, n_samples)"
            )

        values = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            values[i] = self._compute_weighted_kernel_sum(i, X, training=False)

        return values

    def _compute_weighted_kernel_sum(self, index: int, X: np.ndarray, training: bool) -> float:
        """Helper function to compute the weighted sum over support vectors used for both training
        and prediction with the Pegasos algorithm.

        Args:
            index: fixed index distinguishing some datum
            X: Features
            training: flag indicating whether the loop is used within training or prediction

        Returns:
            Weighted sum of kernel evaluations employed in the Pegasos algorithm
        """
        # non-zero indices corresponding to the support vectors
        support_indices = list(self._alphas.keys())

        # for training
        if training:
            # support vectors
            x_supp = X[support_indices]
        # for prediction
        else:
            x_supp = self._x_train[support_indices]
        if not self._precomputed:
            # evaluate kernel function only for the fixed datum and the support vectors
            kernel = self._quantum_kernel.evaluate(X[index], x_supp) + self._kernel_offset
        else:
            kernel = X[index, support_indices]

        # map the training labels of the support vectors to {-1,1}
        y = np.array(list(map(self._label_map.get, self._y_train[support_indices])))
        # weights for the support vectors
        alphas = np.array(list(self._alphas.values()))
        # this value corresponds to a sum of kernel values weighted by their labels and alphas
        value = np.sum(alphas * y * kernel)

        return value

    @property
    def quantum_kernel(self) -> BaseKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: BaseKernel):
        """
        Sets quantum kernel. If previously a precomputed kernel was set, it is reset to ``False``.
        """

        self._quantum_kernel = quantum_kernel
        # quantum kernel is set, so we assume the kernel is not precomputed
        self._precomputed = False

        # reset training status
        self._reset_state()

    @property
    def num_steps(self) -> int:
        """Returns number of steps in the Pegasos algorithm."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps: int):
        """Sets the number of steps to be used in the Pegasos algorithm."""
        self._num_steps = num_steps

        # reset training status
        self._reset_state()

    @property
    def precomputed(self) -> bool:
        """Returns a boolean flag indicating whether a precomputed kernel is used."""
        return self._precomputed

    @precomputed.setter
    def precomputed(self, precomputed: bool):
        """Sets the pre-computed kernel flag. If ``True`` is passed then the previous kernel is
        cleared. If ``False`` is passed then a new instance of
        :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel` is created."""
        self._precomputed = precomputed
        if precomputed:
            # remove the kernel, a precomputed will
            self._quantum_kernel = None
        else:
            # re-create a new default quantum kernel
            self._quantum_kernel = FidelityQuantumKernel()

        # reset training status
        self._reset_state()

    def _reset_state(self):
        """Resets internal data structures used in training."""
        self.fit_status_ = PegasosQSVC.UNFITTED
        self._alphas = None
        self._x_train = None
        self._n_samples = None
        self._y_train = None
        self._label_map = None
        self._label_pos = None
        self._label_neg = None
