# pylint: disable=invalid-name
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

"""Quantum Pegasos Support Vector Classifier"""

from typing import Optional
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


class PegasosQSVC(SVC):
    r"""Quantum Pegasos Support Vector Classifier
    This class implements the algorithm developed in
    https://link.springer.com/content/pdf/10.1007/s10107-010-0420-4.pdf
    and implements some of the methods like ``fit`` and ``predict`` like in QSVC.
    **Example**
    .. code-block::
        qkernel = QuantumKernel()

        qpsvc = PegasosQSVC(quantum_kernel=qkernel)
        qpsvc.fit(sample_train,label_train)
        qpsvc.predict(sample_test)
    """

    def __init__(
        self,
        quantum_kernel: Optional[QuantumKernel] = None,
        C: float = 1,
        num_steps: int = 1000,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            quantum_kernel: QuantumKernel to be used for classification.
            C: positive regularization parameter
            num_steps: number of steps in the Pegasos algorithm
            verbose: whether the time taken for train and predict is printed
        """
        self._quantum_kernel = quantum_kernel if quantum_kernel is not None else QuantumKernel()

        super().__init__(C=C)
        self._num_steps = num_steps
        self._verbose = verbose

        # these are the parameters being fit and are needed for prediction
        self._fit_status = False
        self._alphas = None
        self._x_train = None
        self._y_train = None

        # added to all kernel values  to include an implicit bias to the hyperplane
        self._kernel_offset = 1

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: None = None) -> None:
        """Implementation of the kernalized Pegasos algorithm to fit the SVM
        Args:
            X: shape (x_samples, s), train features
            y: shape (x_samples) train labels
            sample_weight: None

        Raises:
            NotImplementedError:
                - when a sample_weight which is not None is passed
        """
        if sample_weight is not None:
            raise NotImplementedError("all samples have to be weighed equally")

        self._fit_internal(X, y)

    def fit_precomputed(self, X: np.ndarray, y: np.ndarray, precomputed_kernel: np.ndarray) -> None:
        """Implementation of the kernalized Pegasos algorithm to fit the SVM using a precomputed kernel
        Args:
            X: shape (x_samples, s), train features
            y: shape (x_samples) train labels
            precomputed_kernel: shape (x_samples, x_samples) optional pre computed kernel matrix
        """
        self._fit_internal(X, y, precomputed_kernel)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X.
        Args:
            X: shape (x_samples, s)

        Returns:
            y_pred: Shape (x_samples), the class labels in {-1, +1} for samples in X.

        Raises:
            QiskitMachineLearningError:
                - predict is called before the model has been fit
        """
        if not self._fit_status:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")

        t_0 = datetime.now()
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            value = 0.0
            for j in self._alphas:  # only loop over the non zero alphas
                value += (
                    self._alphas[j]
                    * self._y_train[j]
                    * (self._quantum_kernel.evaluate(X[i], self._x_train[j]) + self._kernel_offset)
                )
            y[i] = np.sign(value)

        if self._verbose:
            print(f"prediction completed after {str(datetime.now() - t_0)[:-7]}")

        return y

    def _fit_internal(
        self, X: np.ndarray, y: np.ndarray, precomputed_kernel: Optional[np.ndarray] = None
    ) -> None:
        """Helper function implementing the kernalized Pegasos algorithm to fit the SVM for both the
        pre computed and non pre computed version.
        Args:
            X: shape (x_samples, s), train features
            y: shape (x_samples) train labels
            precomputed_kernel: shape (x_samples, x_samples) optional pre computed kernel matrix

        Raises:
            ValueError:
                - X and/or y have the wrong shape
                - X and y have incompatible dimensions
                - Pre-computed kernel matrix precomputed_kernel has the wrong shape and/or dimension
                - y contains incompatible labels
        """
        # check whether the data has the right format
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if np.ndim(y) != 1:
            raise ValueError("y has to be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have to contain the same number of samples")
        if not np.all(np.unique(y) == np.array([-1, 1])):
            raise ValueError("the labels in y have to be in {-1, +1}")
        if (precomputed_kernel is not None) and (
            not precomputed_kernel.shape == (X.shape[0], X.shape[0])
        ):
            raise ValueError(
                f"precomputed_kernel has the wrong shape {precomputed_kernel.shape}, \
                it should be {(X.shape[0], X.shape[0])}"
            )

        self._x_train = X
        self._y_train = y

        # empty dictionaries to represent sparse arrays
        self._alphas = {}
        kernel = {}

        t_0 = datetime.now()
        for step in range(1, self._num_steps + 1):
            i = np.random.randint(0, len(y))

            value = 0.0
            for j in self._alphas:  # only loop over the non zero alphas
                if precomputed_kernel is None:
                    kernel[(i, j)] = kernel.get((i, j), self._quantum_kernel.evaluate(X[i], X[j]))
                    value += self._alphas[j] * y[j] * (kernel[(i, j)] + self._kernel_offset)
                else:
                    value += (
                        self._alphas[j] * y[j] * (precomputed_kernel[i, j] + self._kernel_offset)
                    )

            if (y[i] * self.C / step) * value < 1:
                self._alphas[i] = self._alphas.get(i, 0) + 1

        self._fit_status = True

        if self._verbose:
            print(f"fit completed after {str(datetime.now() - t_0)[:-7]}")
