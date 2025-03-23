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

"""Quantum Support Vector Regressor"""

import warnings
from typing import Optional

from sklearn.svm import SVR

from ...algorithms.serializable_model import SerializableModelMixin
from ...exceptions import QiskitMachineLearningWarning
from ...kernels import BaseKernel, FidelityQuantumKernel


class QSVR(SVR, SerializableModelMixin):
    r"""Quantum Support Vector Regressor that extends the scikit-learn
    `sklearn.svm.SVR <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_
    regressor and introduces an additional `quantum_kernel` parameter.

    This class shows how to use a quantum kernel for regression. The class inherits its methods
    like ``fit`` and ``predict`` from scikit-learn, see the example below.
    Read more in the
    `scikit-learn user guide <https://scikit-learn.org/stable/modules/svm.html#svm-regression>`_.

    **Example**

    .. code-block::

        qsvr = QSVR(quantum_kernel=qkernel)
        qsvr.fit(sample_train,label_train)
        qsvr.predict(sample_test)
    """

    def __init__(self, *, quantum_kernel: Optional[BaseKernel] = None, **kwargs):
        """
        Args:
            quantum_kernel: A quantum kernel to be used for regression. If None,
                default to :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`.
            *args: Variable length argument list to pass to SVR constructor.
            **kwargs: Arbitrary keyword arguments to pass to SVR constructor.
        """
        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
            # if we don't delete, then this value clashes with our quantum kernel
            del kwargs["kernel"]
        if quantum_kernel is None:
            msg = "No quantum kernel is provided, SamplerV1 based quantum kernel will be used."
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
        self._quantum_kernel = quantum_kernel if quantum_kernel else FidelityQuantumKernel()

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
        names = SVR._get_param_names()
        names.remove("kernel")
        return sorted(names + ["quantum_kernel"])
