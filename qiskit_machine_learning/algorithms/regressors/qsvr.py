# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Support Vector Regressor"""

from __future__ import annotations
import warnings

from sklearn.svm import SVR

from ...algorithms.serializable_model import SerializableModelMixin
from ...exceptions import QiskitMachineLearningWarning
from ...kernels import BaseKernel, FidelityQuantumKernel


class QSVR(SVR, SerializableModelMixin):
    r"""Quantum Support Vector Regressor.

    It extends scikit-learn's
    `sklearn.svm.SVR <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_
    by introducing a ``quantum_kernel`` parameter for computing similarity between samples using
    a quantum kernel.

    The class follows scikit-learn conventions and inherits methods such as :meth:`fit` and
    :meth:`predict`. For general SVR usage and parameters, refer to the
    `scikit-learn user guide <https://scikit-learn.org/stable/modules/svm.html#svm-regression>`_.

    Notes:
        - Passing ``kernel=...`` to the constructor is not supported; use ``quantum_kernel``.
          If ``kernel`` is provided, it is discarded and a
          :class:`~qiskit_machine_learning.exceptions.QiskitMachineLearningWarning` is emitted.
        - If ``quantum_kernel`` is ``None``, a
          :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel` is created. A
          ``feature_map`` may be provided via ``kwargs`` and will be forwarded to the default
          fidelity kernel.
        - If ``quantum_kernel == "precomputed"``, the estimator is configured for scikit-learn's
          precomputed-kernel mode and expects kernel matrices as input.


    **Example**

    .. code-block:: python

        qsvr = QSVR(quantum_kernel=qkernel)
        qsvr.fit(sample_train, label_train)
        y_pred = qsvr.predict(sample_test)

    """

    def __init__(self, *, quantum_kernel: BaseKernel | None = None, **kwargs):
        """Create a quantum-kernel SVR estimator.

        Args:
            quantum_kernel: Quantum kernel configuration.

                - If ``None``, defaults to
                  :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`.
                  In this case, an optional ``feature_map`` may be provided via ``kwargs`` and
                  is forwarded to the default fidelity kernel.
                - If equal to the string ``"precomputed"``, the estimator is configured for
                  scikit-learn's precomputed-kernel mode (i.e. it expects kernel matrices
                  rather than raw samples).
                - Otherwise, it must be a :class:`~qiskit_machine_learning.kernels.BaseKernel`
                  instance and its :meth:`~qiskit_machine_learning.kernels.BaseKernel.evaluate`
                  method will be used as the callable kernel by scikit-learn.
            **kwargs: Keyword arguments forwarded to :class:`sklearn.svm.SVR`.

                The ``kernel`` keyword is not supported and will be discarded (use
                ``quantum_kernel`` instead). If provided, a warning is emitted.

        Warns:
            QiskitMachineLearningWarning: If a ``kernel`` argument is provided in ``kwargs`` and
                is discarded.
        """
        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
            # if we don't delete, then this value clashes with our quantum kernel
            del kwargs["kernel"]

        feature_map = kwargs.pop("feature_map", None)

        # Important: when quantum_kernel == "precomputed" we intentionally store the string
        # and configure SVR accordingly.
        self._quantum_kernel = (
            quantum_kernel if quantum_kernel else FidelityQuantumKernel(feature_map=feature_map)
        )

        if quantum_kernel == "precomputed":
            super().__init__(kernel=self._quantum_kernel, **kwargs)
        else:
            super().__init__(kernel=self._quantum_kernel.evaluate, **kwargs)

    @property
    def quantum_kernel(self) -> BaseKernel:
        """Quantum kernel configuration for this estimator.

        Returns:
            BaseKernel: a :class:`~qiskit_machine_learning.kernels.BaseKernel`
            instance.
        """
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: BaseKernel) -> None:
        """Set the quantum kernel used by this estimator.

        This updates the underlying scikit-learn ``kernel`` callable to use
        ``quantum_kernel.evaluate``.

        Args:
            quantum_kernel: The new quantum kernel.

        Notes:
            Setting this always switches the estimator to callable-kernel mode (i.e. away from
            ``"precomputed"``), because a :class:`~qiskit_machine_learning.kernels.BaseKernel`
            is required to provide :meth:`~qiskit_machine_learning.kernels.BaseKernel.evaluate`.
        """
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate

    # we override this method to be able to pretty print this instance
    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Return estimator parameter names for scikit-learn compatibility.

        This removes scikit-learn's ``kernel`` parameter from the public parameter list and
        exposes ``quantum_kernel`` instead, so that :func:`sklearn.base.clone` and
        :meth:`get_params` / :meth:`set_params` behave as expected.

        Returns:
            list[str]: Sorted list of parameter names.
        """
        names = SVR._get_param_names()
        names.remove("kernel")
        return sorted(names + ["quantum_kernel"])
