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

from typing import Optional, Dict, Tuple
from datetime import datetime
import logging
import numpy as np
from sklearn.svm import SVC
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

logger = logging.getLogger(__name__)


class PegasosQSVC(SVC):
    r"""Quantum Pegasos Support Vector Classifier
    This class implements the algorithm developed in [1] and includes some of the methods like
    ``fit`` and ``predict`` like in QSVC.

    **Example**

    .. code-block::
        quantum_kernel = QuantumKernel()

        pegasos_qsvc = PegasosQSVC(quantum_kernel=quantum_kernel)
        pegasos_qsvc.fit(sample_train, label_train)
        pegasos_qsvc.predict(sample_test)

    **References**
        [1]: Shalev-Shwartz et al., Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
            `Pegasos for SVM <https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf>`_

    """

    def __init__(
        self,
        quantum_kernel: Optional[QuantumKernel] = None,
        C: float = 1,
        num_steps: int = 1000,
        precomputed: bool = False,
    ) -> None:
        """
        Args:
            quantum_kernel: QuantumKernel to be used for classification
            C: positive regularization parameter
            num_steps: number of steps in the Pegasos algorithm. There is no early stopping criterion
            precomputed: flag indicating whether a precomputed kernel is used

        Raises:
            TypeError:
                - To use a precomputed kernel, 'quantum_kernel' has to be of None type
        """
        if quantum_kernel is None:
            if not precomputed:
                self._quantum_kernel = QuantumKernel()
            else:
                self._quantum_kernel = "precomputed"
        elif isinstance(quantum_kernel, QuantumKernel):
            if not precomputed:
                self._quantum_kernel = quantum_kernel
            else:
                raise TypeError("'quantum_kernel' has to be None to use a precomputed kernel")
        else:
            TypeError("'quantum_kernel' has to be of type None or QuantumKernel")

        super().__init__(C=C)
        self._num_steps = num_steps
        self._precomputed = precomputed

        # these are the parameters being fit and are needed for prediction
        self._fit_status = False
        self._alphas: Optional[Dict[int, int]] = None
        self._x_train: Optional[np.ndarray] = None
        self._n_samples: Optional[int] = None
        self._y_train: Optional[np.ndarray] = None
        self._label_map: Optional[Dict[int, int]] = {}
        self._label_pos = None
        self._label_neg = None

        # added to all kernel values  to include an implicit bias to the hyperplane
        self._kernel_offset = 1

    # pylint: disable=invalid-name
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        """Implementation of the kernelized Pegasos algorithm to fit the QSVC.
        Args:
            X: Train features. For a callable kernel shape (n_samples, n_features), for a precomputed
               kernel shape (n_samples, n_samples).
            y: shape (n_samples), train labels.
            sample_weight: this parameter is not supported, passing a value raises an error.

        Returns:
            ``self``, a trained model.

        Raises:
            ValueError:
                - X and/or y have the wrong shape
                - X and y have incompatible dimensions
                - y includes more than two unique labels
                - Pre-computed kernel matrix has the wrong shape and/or dimension
            NotImplementedError:
                - when a sample_weight which is not None is passed
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

        # the algorithm works with labels in {+1, -1}
        self._label_pos = np.unique(y)[0]
        self._label_neg = np.unique(y)[1]
        self._label_map[self._label_pos] = +1
        self._label_map[self._label_neg] = -1

        # the training data are later needed for prediction
        self._x_train = X
        self._y_train = y
        self._n_samples = X.shape[0]

        # empty dictionaries to represent sparse arrays
        self._alphas = {}
        kernel: Dict[Tuple, np.ndarray] = {}

        t_0 = datetime.now()
        for step in range(1, self._num_steps + 1):
            # for every step, a random index (determining a random datum) is fixed
            i = np.random.randint(0, len(y))

            # this value corresponds to a sum of kernel values weighted by their labels and alphas
            value = 0.0
            # only loop over the non zero alphas (preliminary support vectors)
            for j in self._alphas:
                if not self._precomputed:
                    # evaluate kernel function only for the fixed datum and the data with non zero alpha
                    kernel[(i, j)] = kernel.get((i, j), self._quantum_kernel.evaluate(X[i], X[j]))
                    value += (
                        # alpha weights the contribution of the associated datum
                        self._alphas[j]
                        # the class membership labels have to be in {-1, +1}
                        * self._label_map[y[j]]
                        # the offset to the kernel function leads to an implicit bias term
                        * (kernel[(i, j)] + self._kernel_offset)
                    )
                else:
                    # analogous to the block following the if statement, but for a precomputed kernel
                    value += (
                        self._alphas[j] * self._label_map[y[j]] * (X[i, j] + self._kernel_offset)
                    )

            if (self._label_map[y[i]] * self.C / step) * value < 1:
                # only way for a component of alpha to become non zero
                self._alphas[i] = self._alphas.get(i, 0) + 1

        self._fit_status = True

        logger.debug("fit completed after %s", str(datetime.now() - t_0)[:-7])

        return self

    # pylint: disable=invalid-name
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X.
        Args:
            X: Train features. For a callable kernel shape (m_samples, n_features), for a precomputed
                kernel shape (m_samples, n_samples), where m denotes the set to be predicted and n the
                size of the training set.

        Returns:
            y_pred: Shape (n_samples), the predicted class labels for samples in X.

        Raises:
            QiskitMachineLearningError:
                - predict is called before the model has been fit
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension
        """
        if not self._fit_status:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if self._precomputed and self._n_samples != X.shape[1]:
            raise ValueError(
                "For a precomputed kernel, X should be in shape (m_samples, n_samples)"
            )

        t_0 = datetime.now()
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            value = 0.0
            for j in self._alphas:  # only loop over the non zero alphas
                if not self._precomputed:
                    value += (
                        self._alphas[j]
                        * self._label_map[self._y_train[j]]
                        * (
                            self._quantum_kernel.evaluate(X[i], self._x_train[j])
                            + self._kernel_offset
                        )
                    )
                else:
                    value += (
                        self._alphas[j]
                        * self._label_map[self._y_train[j]]
                        * (X[i, j] + self._kernel_offset)
                    )

            if value > 0:
                y[i] = self._label_pos
            else:
                y[i] = self._label_neg

        logger.debug("prediction completed after %s", str(datetime.now() - t_0)[:-7])

        return y

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
