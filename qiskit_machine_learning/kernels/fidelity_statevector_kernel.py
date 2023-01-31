# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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

import functools
from typing import Type, TypeVar

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .base_kernel import BaseKernel

SV = TypeVar("SV", bound=Statevector)


class FidelityStatevectorKernel(BaseKernel):
    r"""
    A reference implementation of the quantum kernel interface optimized for (and limited to)
    classically simulated statevectors.

    Here, the kernel function is defined as the overlap of two simulated quantum statevectors
    produced by a parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2.

    In this implementation, :math:`|\phi(y)\rangle` is given by a ``Statevector.data`` array. These
    arrays are stored in a statevector cache to avoid repeated evaluation of the quantum circuit.
    This cache can be cleared using :meth:`clear_cache`. By default the cache is cleared when
    :meth:`evaluate` is called, unless ``auto_clear_cache`` is ``False``.

    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        statevector_type: Type[SV] = Statevector,
        cache_size: int | None = None,
        auto_clear_cache: bool = True,
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
                should inherit from (and defaults to) :class:`~qiskit.quantum_info.Statevector`.
            cache_size: Maximum size of the statevector cache. When ``None`` this is unbounded.
            auto_clear_cache: Determines whether the statevector cache is retained when
                :meth:`evaluate` is called. The cache is automatically cleared by default.
        """
        super().__init__(feature_map=feature_map)

        self._statevector_type = statevector_type
        self._auto_clear_cache = auto_clear_cache

        # Create the statevector cache at the instance level.
        self._get_statevector = functools.lru_cache(maxsize=cache_size)(self._get_statevector_)

    def evaluate(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._auto_clear_cache:
            self.clear_cache()

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        if y_vec is None:
            y_vec = x_vec

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_statevector(tuple(x)) for x in x_vec]
        y_svs = [self._get_statevector(tuple(y)) for y in y_vec]

        kernel_matrix = np.ones(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                if np.array_equal(x, y):
                    continue
                kernel_matrix[i, j] = self._compute_kernel_entry(x, y)

        return kernel_matrix

    def _get_statevector_(self, param_values: tuple[float]) -> np.ndarray:
        # lru_cache requires hashable function arguments
        qc = self._feature_map.bind_parameters(param_values)
        return self._statevector_type(qc).data

    @staticmethod
    def _compute_kernel_entry(x: np.ndarray, y: np.ndarray) -> float:
        return np.abs(np.conj(x) @ y) ** 2

    def clear_cache(self):
        """Clear the statevector cache."""
        # pylint: disable=no-member
        self._get_statevector.cache_clear()
