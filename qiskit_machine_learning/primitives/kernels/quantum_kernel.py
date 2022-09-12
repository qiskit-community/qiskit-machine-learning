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
"""Overlap Quantum Kernel"""

from __future__ import annotations

import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.state_fidelities import BaseStateFidelity, ComputeUncompute
from qiskit.primitives import Sampler

from .base_kernel import BaseKernel


class QuantumKernel(BaseKernel):
    r"""
    QuantumKernel

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    .. math::

        K(x, y) = \langle f(x), f(y)\rangle.

    Here K is the kernel function, x, y are n dimensional inputs. f is a map from n-dimension
    to m-dimension space. :math:`\langle x, y \rangle` denotes the dot product.
    Usually m is much larger than n.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y and feature
    map f, all of n dimension. This kernel matrix can then be used in classical machine learning
    algorithms such as support vector classification, spectral clustering or ridge regression.

    Here, the kernel function is defined as the overlap of two quantum states defined by a
    parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2
    """

    def __init__(
        self,
        sampler: Sampler | None = None,
        feature_map: QuantumCircuit | None = None,
        fidelity: str | BaseStateFidelity = "zero_prob",
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
    ) -> None:
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)

        # if feature_map is None:
        #     feature_map = ZZFeatureMap(2)
        # self._feature_map = feature_map
        #
        # self._num_features = self._feature_map.num_parameters

        eval_duplicates = evaluate_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Unsupported value passed as evaluate_duplicates: {evaluate_duplicates}"
            )
        self._evaluate_duplicates = eval_duplicates

        if isinstance(fidelity, str):
            if sampler is None:
                raise ValueError(
                    "If the fidelity is passed as a string, a sampler has to be provided"
                    "(currently set to None)."
                )
            if fidelity == "zero_prob":
                self._fidelity = ComputeUncompute(sampler=sampler)
            else:
                raise ValueError(
                    f"{fidelity} is not a valid string for a fidelity. Currently supported: 'zero_prob'."
                )
        else:
            if sampler is not None:
                warnings.warn(
                    "Passed both a sampler and a fidelity instance. If passing a fidelity instance"
                    " for `fidelity`, the sampler will not be used.",
                )
            self._fidelity = fidelity

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        if is_symmetric:
            left_parameters, right_parameters, indices = self._get_symmetric_parameterization(x_vec)
            kernel_matrix = self._get_symmetric_kernel_matrix(
                kernel_shape, left_parameters, right_parameters, indices
            )
        else:
            left_parameters, right_parameters, indices = self._get_parameterization(
                x_vec, y_vec
            )
            kernel_matrix = self._get_kernel_matrix(
                kernel_shape, left_parameters, right_parameters, indices
            )

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

    def _get_parameterization(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = []
        for i, x_i in enumerate(x_vec):
            for j, y_j in enumerate(y_vec):
                if self._is_trivial(i, j, x_i, y_j, False):
                    continue

                left_parameters = np.vstack((left_parameters, x_i))
                right_parameters = np.vstack((right_parameters, y_j))
                indices.append((i, j))

        return left_parameters, right_parameters, indices

    def _get_symmetric_parameterization(
        self, x_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = []
        for i, x_i in enumerate(x_vec):
            for j, x_j in enumerate(x_vec[i:]):
                if self._is_trivial(i, i + j, x_i, x_j, True):
                    continue

                left_parameters = np.vstack((left_parameters, x_i))
                right_parameters = np.vstack((right_parameters, x_j))
                indices.append((i, i + j))

        return left_parameters, right_parameters, indices

    # def _get_kernel_matrix(
    #     self, left_parameters: np.ndarray, right_parameters: np.ndarray, trivial_entries: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Given a parameterization, this computes the symmetric kernel matrix.
    #     """
    #     if np.all(trivial_entries):
    #         return trivial_entries
    #     # todo: a quick fix
    #     size = left_parameters.shape[0]
    #     job = self._fidelity.run(
    #         [self._feature_map] * size,
    #         [self._feature_map] * size,
    #         left_parameters,
    #         right_parameters,
    #     )
    #     kernel_entries = job.result().fidelities
    #     kernel_matrix = np.zeros(trivial_entries.shape)
    #
    #     index = 0
    #     for i in range(trivial_entries.shape[0]):
    #         for j in range(trivial_entries.shape[1]):
    #             if trivial_entries[i, j]:
    #                 kernel_matrix[i, j] = 1.0
    #             else:
    #                 kernel_matrix[i, j] = kernel_entries[index]
    #                 index += 1
    #     return kernel_matrix

    def _get_kernel_matrix(
        self, kernel_shape, left_parameters: np.ndarray, right_parameters: np.ndarray, indices
    ) -> np.ndarray:
        """
        Given a parameterization, this computes the symmetric kernel matrix.
        """
        # todo: a quick fix
        size = left_parameters.shape[0]
        job = self._fidelity.run(
            [self._feature_map] * size,
            [self._feature_map] * size,
            left_parameters,
            right_parameters,
        )
        kernel_entries = job.result().fidelities

        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    # def _get_symmetric_kernel_matrix(
    #     self, left_parameters: np.ndarray, right_parameters: np.ndarray, trivial_entries: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Given a set of parameterization, this computes the kernel matrix.
    #     """
    #     if np.all(trivial_entries):
    #         # todo: this returns a matrix of bool values!
    #         return trivial_entries
    #     # todo: a quick fix
    #     size = left_parameters.shape[0]
    #     job = self._fidelity.run(
    #         [self._feature_map] * size,
    #         [self._feature_map] * size,
    #         left_parameters,
    #         right_parameters,
    #     )
    #     kernel_entries = job.result().fidelities
    #     kernel_matrix = np.zeros(trivial_entries.shape)
    #     index = 0
    #     for i in range(trivial_entries.shape[0]):
    #         for j in range(i, trivial_entries.shape[1]):
    #             if trivial_entries[i, j]:
    #                 kernel_matrix[i, j] = 1.0
    #             else:
    #                 kernel_matrix[i, j] = kernel_entries[index]
    #                 index += 1
    #
    #     kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())
    #     return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self, kernel_shape, left_parameters: np.ndarray, right_parameters: np.ndarray, indices
    ) -> np.ndarray:
        """
        Given a set of parameterization, this computes the kernel matrix.
        """
        # todo: a quick fix
        num_circuits = left_parameters.shape[0]
        if num_circuits != 0:
            job = self._fidelity.run(
                [self._feature_map] * num_circuits,
                [self._feature_map] * num_circuits,
                left_parameters,
                right_parameters,
            )
            kernel_entries = job.result().fidelities
        else:
            # trivial case, only identical samples
            kernel_entries = []

        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]
            kernel_matrix[row, col] = kernel_entries[i]

        return kernel_matrix

    def _is_trivial(self, i, j, x_i, y_j, symmetric):
        # ("all", "off_diagonal", "none")

        # if we evaluate all combinations, then non trivial
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
