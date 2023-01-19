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
"""Fidelity Statevector Kernel"""

from __future__ import annotations

from typing import Type, TypeVar

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .base_kernel import BaseKernel

SV = TypeVar("SV", bound=Statevector)


class FidelityStatevectorKernel(BaseKernel):
    r"""
    A reference implementation of the quantum kernel interface limited to classically simulated
    statevectors.

    Here, the kernel function is defined as the overlap of two simulated quantum statevectors
    produced by a parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2.

    In this implementation, :math:`\phi` is represented by a ``Statevector.data`` array,
    thus the kernel function is given simply by

    .. math::

        K(x,y) = |\phi(x)^\dagger \phi(y)|^2.

    These arrays are stored in a statevector cache for reuse to avoid repeated computation.
    This stash can be cleared using :meth:`clear_cache`. By default, the cache is cleared
    when :meth:`evaluate` is called.

    """

    def __init__(
        self, *, feature_map: QuantumCircuit | None = None, statevector_type: Type[SV] = Statevector
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            statevector_type: The type of Statevector that will be instantiated using the
                ``feature_map`` quantum circuit and used to compute the fidelity kernel. This type
                should inherit from and defaults to :class:`~qiskit.quantum_info.Statevector`.
        """
        super().__init__(feature_map=feature_map)

        self._statevector_type = statevector_type
        self._statevector_cache: dict[tuple[float, ...], np.ndarray] = {}

    def evaluate(
        self, x_vec: np.ndarray, y_vec: np.ndarray | None = None, clear_cache: bool = True
    ) -> np.ndarray:
        r"""
        Construct kernel matrix for given data.

        If ``y_vec`` is ``None``, self inner product is calculated.

        Args:
            x_vec: 1D or 2D array of datapoints, :math:`N\times D`, where :math:`N` is the number of
                datapoints, :math:`D` is the feature dimension.
            y_vec: 1D or 2D array of datapoints, :math:`M\times D`, where :math:`M` is the number of
                datapoints, :math:`D` is the feature dimension.
            clear_cache: Boolean that determines whether the statevector cache is cleared when
                evaluate is called. Defaults to ``True``.

        Returns:
            2D matrix, :math:`N\times M`.
        """
        if clear_cache:
            self.clear_cache()

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        if y_vec is None:
            y_vec = x_vec

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_statevector(x) for x in x_vec]
        y_svs = [self._get_statevector(y) for y in y_vec]

        kernel_matrix = np.ones(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                if np.array_equal(x, y):
                    continue
                kernel_matrix[i, j] = self._compute_kernel_element(x, y)

        return kernel_matrix

    @staticmethod
    def _compute_kernel_element(x: np.ndarray, y: np.ndarray) -> float:
        return np.abs(np.conj(x) @ y) ** 2

    def _get_statevector(self, param_values) -> np.ndarray:
        param_values = tuple(param_values)
        statevector = self._statevector_cache.get(param_values, None)

        if statevector is None:
            qc = self._feature_map.bind_parameters(param_values)
            statevector = self._statevector_type(qc).data
            self._statevector_cache[param_values] = statevector

        return statevector

    def clear_cache(self):
        """Clear the statevector cache."""
        self._statevector_cache.clear()
