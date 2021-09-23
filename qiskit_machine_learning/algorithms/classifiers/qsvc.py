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

"""Quantum Support Vector Classifier"""

import warnings
from typing import Optional, Sequence

import numpy as np
from sklearn.svm import SVC
import numpy as np

from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from qiskit_machine_learning.algorithms.kernel_trainers import QuantumKernelTrainer


class QSVC(SVC):
    r"""Quantum Support Vector Classifier.

    This class shows how to use a quantum kernel for classification. The class extends
    `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_,
    and thus inherits its methods like ``fit`` and ``predict`` used in the example below.
    Read more in the `sklearn user guide
    <https://scikit-learn.org/stable/modules/svm.html#svm-classification>`_.

    **Example**

    .. code-block::python

        qsvc = QSVC(quantum_kernel=quant_kernel, kernel_trainer=qkt)
        qsvc.fit(sample_train,label_train)
        qsvc.predict(sample_test)
    """

    def __init__(
        self,
        quantum_kernel: Optional[QuantumKernel] = None,
        kernel_trainer: Optional[QuantumKernelTrainer] = None,
        **kwargs,
    ):
        r"""
        Args:
            quantum_kernel:``QuantumKernel`` to be used for classification
            kernel_trainer: ``QuantumKernelTrainer`` to be used for kernel optimization
            **kwargs: Arbitrary keyword arguments to pass to ``SVC`` constructor
        """
        self._quantum_kernel = None
        self._kernel_trainer = None

        if "random_state" not in kwargs:
            kwargs["random_state"] = algorithm_globals.random_seed

        super().__init__(kernel=self._quantum_kernel.evaluate, **kwargs)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets quantum kernel"""
        # QuantumKernel passed
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate

    @property
    def kernel_trainer(self) -> Optional[QuantumKernelTrainer]:
        """Returns quantum kernel trainer"""
        return self._kernel_trainer

    @kernel_trainer.setter
    def kernel_trainer(self, qk_trainer: Optional[QuantumKernelTrainer]) -> None:
        """Returns quantum kernel trainer"""
        self._kernel_trainer = qk_trainer

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[Sequence[float]] = None
    ) -> SVC:
        # Ensure a backend was specified
        if self._quantum_kernel.quantum_instance is None:
            raise QiskitMachineLearningError(
                """
            Error fitting QSVC. No quantum instance was specified.
            """
            )

        # Ensure there are no unbound user parameters
        unbound_user_params = self._quantum_kernel.unbound_user_parameters()
        if (len(unbound_user_params) > 0) and (self._kernel_trainer is None):
            raise QiskitMachineLearningError(
                f"""
            Cannot fit QSVC while feature map has unbound user parameters ({unbound_user_params}).
            """
            )

        # Conduct kernel optimization, if required
        if self._kernel_trainer:
            results = self._kernel_trainer.fit_kernel(self._quantum_kernel, X, y)
            self._quantum_kernel.assign_user_parameters(results.optimal_parameters)

        return super().fit(X=X, y=y, sample_weight=sample_weight)

    # we override this method to be able to pretty print this instance
    @classmethod
    def _get_param_names(cls):
        names = SVC._get_param_names()
        names.remove("kernel")
        return sorted(names + ["quantum_kernel"])

    @property
    def kernel_trainer(self) -> QuantumKernelTrainer:
        """Returns quantum kernel trainer"""
        return self._kernel_trainer

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Wrapper method for SVC.fit which optimizes the quantum kernel's
        user parameters before fitting the SVC.
        """
        if self._kernel_trainer:
            results = self._kernel_trainer.fit_kernel(X, y)
            self.quantum_kernel.assign_user_parameters(results.optimal_parameters)

        return super().fit(X=X, y=y, sample_weight=sample_weight)
