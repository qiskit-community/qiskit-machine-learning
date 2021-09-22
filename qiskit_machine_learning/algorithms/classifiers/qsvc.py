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

from typing import Optional, Union

import numpy as np
from sklearn.svm import SVC

from qiskit import Aer
from qiskit.utils.algorithm_globals import algorithm_globals
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

    .. code-block::

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(sample_train,label_train)
        qsvc.predict(sample_test)
    """

    def __init__(
        self,
        *args,
        quantum_kernel: Optional[Union[QuantumKernel, QuantumKernelTrainer]] = None,
        **kwargs,
    ):
        """
        Args:
            quantum_kernel: QuantumKernel or QuantumKernelTrainer to be used for classification.
            *args: Variable length argument list to pass to SVC constructor.
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor.
        """
        # Class fields
        self._quantum_kernel = None
        self._kernel_trainer = None

        self.quantum_kernel = quantum_kernel

        if "random_state" not in kwargs:
            kwargs["random_state"] = algorithm_globals.random_seed

        super().__init__(
            kernel=self.quantum_kernel.evaluate,
            *args,
            **kwargs,
        )

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: Union[QuantumKernel, QuantumKernelTrainer]):
        """Sets quantum kernel"""
        self._kernel_trainer = None

        # If no quantum kernel was passed, instantiate a default QuantumKernel
        if not quantum_kernel:
            backend = Aer.get_backend("qasm_simulator")
            self._quantum_kernel = QuantumKernel(quantum_instance=backend)

        # If the input QuantumKernel has unbound user params, set the
        # kernel_trainer field
        elif isinstance(quantum_kernel, QuantumKernel):
            self._quantum_kernel = quantum_kernel
            if quantum_kernel.unbound_user_parameters():
                self._kernel_trainer = QuantumKernelTrainer(quantum_kernel)

        # If the input is a QuantumKernelTrainer, set the quantum_kernel and
        # kernel_trainer fields
        elif isinstance(quantum_kernel, QuantumKernelTrainer):
            self._quantum_kernel = quantum_kernel.quantum_kernel
            self._kernel_trainer = quantum_kernel

        else:
            raise ValueError(
                f"""
            Error setting quantum_kernel field. Expected type QuantumKernel or
            QuantumKernelTrainer. Got {quantum_kernel}.
            """
            )

        # SVC kernel field just needs the 2D kernel matrix
        self.kernel = self._quantum_kernel.evaluate

    @property
    def kernel_trainer(self) -> QuantumKernelTrainer:
        """Returns quantum kernel trainer"""
        return self._kernel_trainer

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Wrapper method for SVC.fit which optimizes the quantum kernel's
        user parameters before fitting the SVC.
        """
        if self.kernel_trainer:
            results = self.kernel_trainer.fit_kernel(X, y)
            self.quantum_kernel.assign_user_parameters(results.optimal_parameters)

        super().fit(X=X, y=y, sample_weight=sample_weight)
