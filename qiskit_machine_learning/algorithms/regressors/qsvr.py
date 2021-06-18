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

from typing import Optional

from sklearn.svm import SVR

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
