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

"""Quantum Support Vector Regressor"""

import warnings
from typing import Optional

from sklearn.svm import SVR

from ...exceptions import QiskitMachineLearningWarning
from ...kernels.quantum_kernel import QuantumKernel


class QSVR(SVR):
    r"""Quantum Support Vector Regressor.

    This class shows how to use a quantum kernel for regression. The class extends
    `sklearn.svm.SVR <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_,
    and thus inherits its methods like ``fit`` and ``predict`` used in the example below.
    Read more in the
    `sklearn user guide <https://scikit-learn.org/stable/modules/svm.html#svm-regression>`_.

    **Example**

    .. code-block::

        qsvr = QSVR(quantum_kernel=qkernel)
        qsvr.fit(sample_train,label_train)
        qsvr.predict(sample_test)
    """

    def __init__(self, *args, quantum_kernel: Optional[QuantumKernel] = None, **kwargs):
        """
        Args:
            quantum_kernel: QuantumKernel to be used for regression.
            *args: Variable length argument list to pass to SVR constructor.
            **kwargs: Arbitrary keyword arguments to pass to SVR constructor.
        """
        if (len(args)) != 0:
            msg = (
                f"Positional arguments ({args}) are deprecated as of version 0.3.0 and "
                f"will be removed no sooner than 3 months after the release. Instead use "
                f"keyword arguments."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
            # if we don't delete, then this value clashes with our quantum kernel
            del kwargs["kernel"]

        self._quantum_kernel = quantum_kernel if quantum_kernel else QuantumKernel()

        super().__init__(kernel=self._quantum_kernel.evaluate, *args, **kwargs)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate

    # we override this method to be able to pretty print this instance
    @classmethod
    def _get_param_names(cls):
        names = SVR._get_param_names()
        names.remove("kernel")
        return sorted(names + ["quantum_kernel"])
