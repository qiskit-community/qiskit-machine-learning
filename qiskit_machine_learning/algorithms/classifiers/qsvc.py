# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
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
from typing import Optional

from sklearn.svm import SVC

from qiskit_machine_learning.algorithms.serializable_model import SerializableModelMixin
from qiskit_machine_learning.exceptions import QiskitMachineLearningWarning
from qiskit_machine_learning.kernels import BaseKernel, FidelityQuantumKernel

from ...utils import algorithm_globals


class QSVC(SVC, SerializableModelMixin):
    r"""Quantum Support Vector Classifier that extends the scikit-learn
    `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
    classifier and introduces an additional `quantum_kernel` parameter.

    This class shows how to use a quantum kernel for classification. The class inherits its methods
    like ``fit`` and ``predict`` from scikit-learn, see the example below.
    Read more in the `scikit-learn user guide
    <https://scikit-learn.org/stable/modules/svm.html#svm-classification>`_.

    **Example**

    .. code-block::

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(sample_train,label_train)
        qsvc.predict(sample_test)
    """

    def __init__(self, *, quantum_kernel: Optional[BaseKernel] = None, **kwargs):
        """
        Args:
            quantum_kernel: A quantum kernel to be used for classification.
                Has to be ``None`` when a precomputed kernel is used. If None,
                default to :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`.
            *args: Variable length argument list to pass to SVC constructor.
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor.
        """
        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
            # if we don't delete, then this value clashes with our quantum kernel
            del kwargs["kernel"]

        self._quantum_kernel = quantum_kernel if quantum_kernel else FidelityQuantumKernel()

        if "random_state" not in kwargs:
            kwargs["random_state"] = algorithm_globals.random_seed

        super().__init__(kernel=self._quantum_kernel.evaluate, **kwargs)

    @property
    def quantum_kernel(self) -> BaseKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: BaseKernel):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate

    # we override this method to be able to pretty print this instance
    @classmethod
    def _get_param_names(cls):
        names = SVC._get_param_names()
        names.remove("kernel")
        return sorted(names + ["quantum_kernel"])
