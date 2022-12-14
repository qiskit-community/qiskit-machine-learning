# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Statevector Quantum Kernel"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .base_kernel import BaseKernel


class StatevectorKernel(BaseKernel):
    r"""
    An implementation of the quantum kernel interface using ``Statevector`` features.

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    .. math::

        K(x, y) = \langle f(x), f(y)\rangle.

    Here :math:`K` is the kernel function, :math:`x`, :math:`y` are :math:`n` dimensional inputs.
    :math:`f` is a map from :math:`n`-dimension to :math:`m`-dimension space. :math:`\langle x, y
    \rangle` denotes the dot product. Usually m is much larger than :math:`n`.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints :math:`x` and
    :math:`y` and feature map :math:`f`, all of :math:`n` dimension. This kernel matrix can then be
    used in classical machine learning algorithms such as support vector classification, spectral
    clustering or ridge regression.

    Here, the kernel function is defined as the overlap of two quantum states defined by a
    parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2

    In this implementation, :math:`\phi` is represented by a ``Statevector.data`` array,
    thus the kernel function is simply:

    .. math::

        K(x,y) = (\phi(x)^\dagger \phi(y))^2

    These are stored in a statevector cache for reuse to avoid repeated computation. This stash
    can be cleared using :meth:`clear_cache`.
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            enforce_psd: Project to the closest positive semidefinite matrix if ``x = y``.
                Default ``True``.
            evaluate_duplicates: Defines a strategy how kernel matrix elements are evaluated if
               duplicate samples are found. Possible values are:

                    - ``all`` means that all kernel matrix elements are evaluated, even the diagonal
                      ones when training. This may introduce additional noise in the matrix.
                    - ``off_diagonal`` when training the matrix diagonal is set to `1`, the remaining
                      elements are fully evaluated, e.g., for two identical samples in the
                      dataset. When inferring, all elements are evaluated. This is the default
                      value.
                    - ``none`` when training the diagonal is set to `1` and if two identical samples
                      are found in the dataset the corresponding matrix element is set to `1`.
                      When inferring, matrix elements for identical samples are set to `1`.

        Raises:
            ValueError: When unsupported value is passed to `evaluate_duplicates`.
        """
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)

        eval_duplicates = evaluate_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Unsupported value passed as evaluate_duplicates: {evaluate_duplicates}"
            )
        self._evaluate_duplicates = eval_duplicates

        self._statevector_cache = {}

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_statevector(x) for x in x_vec]
        y_svs = [self._get_statevector(y) for y in y_vec]

        kernel_matrix = np.ones(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                if self._is_trivial(i, j, x, y, False):
                    continue
                kernel_matrix[i, j] =  self._compute_kernel_element(x, y)

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)

        # due to truncation and rounding errors we may get complex numbers
        kernel_matrix = np.real(kernel_matrix)

        return kernel_matrix

    @staticmethod
    def _compute_kernel_element(x: Statevector, y: Statevector) -> float:
        return np.abs(np.conj(x) @ y) ** 2

    def _get_statevector(self, param_values) -> Statevector:
        param_values = tuple(param_values)
        statevector = self._statevector_cache.get(param_values, None)

        if statevector is None:
            qc = self._feature_map.bind_parameters(param_values)
            statevector = Statevector(qc).data
            self._statevector_cache[param_values] = statevector

        return statevector

    @property
    def evaluate_duplicates(self):
        """Returns the strategy used by this kernel to evaluate kernel matrix elements if duplicate
        samples are found."""
        return self._evaluate_duplicates

    def _is_trivial(
        self, i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool
    ) -> bool:
        """
        Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

        Args:
            i: row index of the entry in the kernel matrix.
            j: column index of the entry in the kernel matrix.
            x_i: a sample from the dataset that corresponds to the row in the kernel matrix.
            y_j: a sample from the dataset that corresponds to the column in the kernel matrix.
            symmetric: whether it is a symmetric case or not.

        Returns:
            `True` if the entry is trivial, `False` otherwise.
        """
        # if we evaluate all combinations, then it is non-trivial
        if self._evaluate_duplicates == "all":
            return False

        # if we are on the diagonal and we don't evaluate it, it is trivial
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True

        # if don't evaluate any duplicates
        if np.array_equal(x_i, y_j) and self._evaluate_duplicates == "none":
            return True

        # otherwise evaluate
        return False

    def clear_cache(self):
        """Clear the statevector cache."""
        self._statevector_cache = {}
