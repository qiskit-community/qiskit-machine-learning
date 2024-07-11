# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
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

from functools import lru_cache
from typing import Type, TypeVar, Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from ..utils import algorithm_globals


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

    In this implementation, :math:`|\phi(y)\rangle` is given by the ``data`` attribute of a
    :class:`~qiskit.quantum_info.Statevector` object or one of its subclasses. These
    arrays are stored in a statevector cache to avoid repeated evaluation of the quantum circuit.
    This cache can be cleared using :meth:`clear_cache`. By default the cache is cleared when
    :meth:`evaluate` is called, unless ``auto_clear_cache`` is ``False``.

    Shot noise emulation can also be added. If ``shots`` is ``None``, the exact fidelity is used.
    Otherwise, the mean is taken of samples drawn from a binomial distribution with probability
    equal to the exact fidelity. This model assumes that the fidelity is determined via the
    compute-uncompute method. I.e., the fidelity is given by the probability of measuring
    :math:`0` after preparing the state :math:`U(x)^\dagger U(y) | 0 \rangle`.

    With the addition of shot noise, the kernel matrix may no longer be positive semi-definite. With
    ``enforce_psd`` set to ``True`` this condition is enforced.

    **References:**
    [1] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala,
    A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning
    with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
    `arXiv:1804.11326v2 [quant-ph] <https://arxiv.org/pdf/1804.11326.pdf>`_
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        statevector_type: Type[SV] = Statevector,
        cache_size: int | None = None,
        auto_clear_cache: bool = True,
        shots: int | None = None,
        enforce_psd: bool = True,
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
            shots: The number of shots. If ``None``, the exact fidelity is used. Otherwise, the
                mean is taken of samples drawn from a binomial distribution with probability equal
                to the exact fidelity.
            enforce_psd: Project to the closest positive semidefinite matrix if ``x = y``.
                This is only used when number of shots given is not ``None``.
        """
        super().__init__(feature_map=feature_map)

        self._statevector_type = statevector_type
        self._auto_clear_cache = auto_clear_cache
        self._shots = shots
        self._enforce_psd = enforce_psd
        self._cache_size = cache_size
        # Create the statevector cache at the instance level.
        self._get_statevector = lru_cache(maxsize=cache_size)(self._get_statevector_)

    def evaluate(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._auto_clear_cache:
            self.clear_cache()

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # Determine if calculating self inner product.
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        return self._evaluate(x_vec, y_vec, is_symmetric)

    def _evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool):
        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_statevector(tuple(x)) for x in x_vec]
        y_svs = [self._get_statevector(tuple(y)) for y in y_vec]

        kernel_matrix = np.ones(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                if np.array_equal(x, y):
                    continue
                kernel_matrix[i, j] = self._compute_kernel_entry(x, y)

        if self._enforce_psd and is_symmetric and self._shots is not None:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

    def _get_statevector_(self, param_values: tuple[float]) -> np.ndarray:
        # lru_cache requires hashable function arguments.
        qc = self._feature_map.assign_parameters(param_values)
        return self._statevector_type(qc).data

    def _compute_kernel_entry(self, x: np.ndarray, y: np.ndarray) -> float:
        fidelity = self._compute_fidelity(x, y)
        if self._shots is not None:
            fidelity = self._add_shot_noise(fidelity)
        return fidelity

    @staticmethod
    def _compute_fidelity(x: np.ndarray, y: np.ndarray) -> float:
        return np.abs(np.conj(x) @ y) ** 2

    def _add_shot_noise(self, fidelity: float) -> float:
        return algorithm_globals.random.binomial(n=self._shots, p=fidelity) / self._shots

    def clear_cache(self):
        """Clear the statevector cache."""
        # pylint: disable=no-member
        self._get_statevector.cache_clear()

    def __getstate__(self) -> dict[str, Any]:
        kernel = dict(self.__dict__)
        kernel["_get_statevector"] = None
        return kernel

    def __setstate__(self, kernel: dict[str, Any]):
        self.__dict__ = kernel
        self._get_statevector = lru_cache(maxsize=self._cache_size)(self._get_statevector_)
