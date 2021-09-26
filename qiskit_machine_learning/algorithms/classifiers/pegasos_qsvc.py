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

import numpy as np
from typing import Optional
from datetime import datetime
from sklearn.svm import SVC
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

# TODO should this inherit from some class that exists in Qiskit already?
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
        self._quantum_kernel = (
            quantum_kernel if quantum_kernel is not None else QuantumKernel()
        )

        super().__init__(C=C)
        self._num_steps = num_steps
        self._verbose = verbose

        # these are the parameters being fit and are needed for prediction
        self._fit_status = False
        self._alphas = None
        self._X_train = None
        self._y_train = None

        # added to all kernel values  to include an implicit bias to the hyperplane
        self._c = 1

    def fit(self, X: np.ndarray, y: np.ndarray, K: Optional[np.ndarray] = None) -> None:
        """Implementation of the kernalized Pegasos algorithm to fit the SVM
        Args:
            X: shape (x_samples, s), train features
            y: shape (x_samples) train labels
            K: shape (x_samples, x_samples) optional pre computed kernel matrix
        """
        # check whether the data has the right format
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if np.ndim(y) != 1:
            raise ValueError("y has to be a 1D array")
        if not np.all(np.unique(y) == np.array([-1, 1])):
            raise ValueError("the labels in y have to be in {-1, +1}")
        if (K is not None) and (not (K.shape == (X.shape[0], X.shape[0]))):
            raise ValueError(
                f"K has the wrong shape {K.shape}, it should be {(X.shape[0], X.shape[0])}"
            )

        self._X_train = X
        self._y_train = y

        # empty dictionaries to represent sparse arrays
        self._alphas = {}
        K_dict = {}

        t0 = datetime.now()
        for t in range(1, self._num_steps + 1):
            i = np.random.randint(0, len(y))

            value = 0.0
            for j in self._alphas.keys():  # only loop over the non zero alphas
                if K is None:
                    K_dict[(i, j)] = K_dict.get(
                        (i, j), self._quantum_kernel.evaluate(X[i], X[j])
                    )
                    value += self._alphas[j] * y[j] * (K_dict[(i, j)] + self._c)
                else:
                    value += self._alphas[j] * y[j] * (K[i, j] + self._c)

            if (y[i] * self.C / t) * value < 1:
                self._alphas[i] = self._alphas.get(i, 0) + 1

        self._fit_status = True

        if self._verbose:
            print(f"fit completed after {str(datetime.now() - t0)[:-7]}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X.
        Args:
            X: shape (x_samples, s)
        Returns:
            y_pred: Shape (x_samples), the class labels in {-1, +1} for samples in X.
        """
        if not self._fit_status:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")

        t0 = datetime.now()
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sum = 0.0
            for j in self._alphas.keys():  # only loop over the non zero alphas
                sum += (
                    self._alphas[j]
                    * self._y_train[j]
                    * (self._quantum_kernel.evaluate(X[i], X[j]) + self._c)
                )
            y[i] = np.sign(sum)

        if self._verbose:
            print(f"prediction completed after {str(datetime.now() - t0)[:-7]}")

        return y
